"""
Procurement priorities (with conformal uncertainty), fairness and stability audit, recalibration.
Reads the saved model + prediction files; every number goes to reports/procurement_fairness_results.json.
"""
import sys, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED
from src.models import TARGET, ANCHOR, FIRST_TRAIN_YEAR, TRAIN_END, WEATHER
FIG = REPORTS_DIR / "figures"
COVER = 0.40          # ASSUMPTION: early forward contracts secure 40% of a forecast shortfall that would otherwise be bought at spot
SPOT_PREMIUM = 0.15   # ASSUMPTION: spot buying in a shortfall year costs 15% above contract price
ALPHA = 0.10          # 90% conformal intervals
rng = np.random.default_rng(RANDOM_SEED)


def mape(y, p): return float(np.mean(np.abs(p - y) / y) * 100)


def psi(a, b, bins=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3: return 0.0
    q[0], q[-1] = -np.inf, np.inf
    pa = np.histogram(a, q)[0] / len(a) + 1e-6; pb = np.histogram(b, q)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def main():
    df = pd.read_csv(PROCESSED_DATA_DIR / "agri_yield_features.csv")
    df = df[(df.year >= FIRST_TRAIN_YEAR) & (df.is_outlier == 0)].dropna(subset=[ANCHOR]).reset_index(drop=True)
    model = joblib.load(MODELS_DIR / "final_model.joblib"); cols = list(model.feature_names_in_)
    train = df[df.year <= TRAIN_END]; val = df[df.year.between(2019, 2021)].copy(); test = df[df.year >= 2022].copy()
    val["pred"] = model.predict(val[cols]); test["pred"] = model.predict(test[cols])
    out = {}

    # ================= 1. Procurement plan =================
    # split-conformal interval at (district, crop, year) level, calibrated on validation seasons
    def agg(d, predcol="pred"):
        g = d.assign(pa=d[predcol] * d.area_ha, ya=d[TARGET] * d.area_ha, an=d[ANCHOR] * d.area_ha, pr=d.price_lag_1yr * d.area_ha).groupby(["district", "crop", "year"])
        s = g.agg(area=("area_ha", "sum"), pa=("pa", "sum"), ya=("ya", "sum"), an=("an", "sum"), pr=("pr", "sum"), n=("pa", "size")).reset_index()
        s["pred_yield"] = s.pa / s.area; s["actual_yield"] = s.ya / s.area; s["normal_yield"] = s.an / s.area; s["price"] = s.pr / s.area
        return s
    av = agg(val); score = np.abs(av.actual_yield - av.pred_yield) / av.pred_yield
    q = float(np.quantile(score, min(1.0, (1 - ALPHA) * (1 + 1 / len(score)))))
    out["conformal"] = dict(level=1 - ALPHA, relative_half_width=q, n_calibration_groups=int(len(av)))
    at = agg(test); at["covered"] = (np.abs(at.actual_yield - at.pred_yield) <= q * at.pred_yield)
    out["conformal"]["test_group_coverage"] = float(at.covered.mean())

    def plan(a, scenario_label):
        a = a.copy(); a["change_vs_normal_pct"] = 100 * (a.pred_yield / a.normal_yield - 1)
        a["volume_t"] = a.pred_yield * a.area / 10; a["lo_t"] = a.volume_t * (1 - q); a["hi_t"] = a.volume_t * (1 + q)
        a["shortfall_t"] = np.maximum(0, (a.normal_yield - a.pred_yield) * a.area / 10)
        a["risk_flag"] = np.where(a.change_vs_normal_pct <= -15, "HIGH", np.where(a.change_vs_normal_pct <= -7.5, "WATCH", "NORMAL"))
        a["value_at_risk_rs_lakh"] = a.shortfall_t * 10 * a.price * SPOT_PREMIUM / 1e5     # tonnes->quintal, Rs/quintal
        a["tonnes_secured_by_early_contract"] = a.shortfall_t * COVER
        a["priority_score"] = a.shortfall_t * a.price                                     # tonnes short x price level
        a["scenario"] = scenario_label
        return a.sort_values("priority_score", ascending=False).reset_index(drop=True)
    p24 = plan(at[at.year == 2024], "Forecast for 2024 (test season)")
    # Drought stress scenario: replace the weather of 2024 rows with weather of past drought seasons in the SAME district
    sc = test[test.year == 2024].copy(); dr = train[train.year.isin([2009, 2014, 2015, 2018])]
    for d_name, g in sc.groupby("district"):
        pool = dr[dr.district == d_name]
        pick = pool.sample(len(g), replace=True, random_state=RANDOM_SEED)
        for c in WEATHER: sc.loc[g.index, c] = pick[c].values
    sc["pred_dry"] = model.predict(sc[cols]); sd = agg(sc, "pred_dry")
    sd = sd.assign(pred_yield=sd.pa / sd.area); pdry = plan(sd, "SIMULATED drought (2009/14/15/18-type weather)")
    pd.concat([p24, pdry]).drop(columns=["pa", "ya", "an", "pr"]).round(2).to_csv(REPORTS_DIR / "procurement_plan.csv", index=False)
    keep = ["district", "crop", "pred_yield", "normal_yield", "change_vs_normal_pct", "volume_t", "lo_t", "hi_t", "shortfall_t", "risk_flag", "value_at_risk_rs_lakh"]
    out["plan_2024_top10"] = p24[keep].head(10).round(2).to_dict("records")
    out["plan_drought_top10"] = pdry[keep].head(10).round(2).to_dict("records")
    out["plan_2024_flags"] = p24.risk_flag.value_counts().to_dict(); out["plan_drought_flags"] = pdry.risk_flag.value_counts().to_dict()
    out["drought_total_shortfall_t"] = float(pdry.shortfall_t.sum()); out["drought_tonnes_secured"] = float(pdry.tonnes_secured_by_early_contract.sum())
    out["drought_value_at_risk_rs_lakh"] = float(pdry.value_at_risk_rs_lakh.sum() * COVER)
    out["assumptions"] = dict(coverage_of_shortfall=COVER, spot_premium=SPOT_PREMIUM, note="Volumes are for the sampled farm records in the dataset, not whole-district production.")
    out["crop_ranking_drought"] = pdry.groupby("crop").agg(shortfall_t=("shortfall_t", "sum"), mean_change_pct=("change_vs_normal_pct", "mean")).sort_values("shortfall_t", ascending=False).round(1).reset_index().to_dict("records")
    out["district_ranking_drought"] = pdry.groupby("district").agg(shortfall_t=("shortfall_t", "sum"), mean_change_pct=("change_vs_normal_pct", "mean")).sort_values("shortfall_t", ascending=False).round(1).reset_index().to_dict("records")
    # risk matrix figure
    fig, ax = plt.subplots(figsize=(9, 5.5)); pv = pdry.pivot_table(index="district", columns="crop", values="change_vs_normal_pct")
    im = ax.imshow(pv.values, cmap="RdYlGn", vmin=-70, vmax=10); ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns); ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]): ax.text(j, i, f"{pv.values[i, j]:.0f}%", ha="center", va="center", fontsize=8)
    plt.colorbar(im, label="Forecast change vs normal (%)"); ax.set_title("Risk matrix under a SIMULATED drought: forecast yield change vs normal"); plt.tight_layout(); plt.savefig(FIG / "18_procurement_risk_matrix.png", dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(10, 4.5)); t = at.sort_values(["district", "crop"]).head(14)
    ax.errorbar(range(len(t)), t.pred_yield, yerr=q * t.pred_yield, fmt="o", capsize=3, label="forecast with 90% interval"); ax.plot(range(len(t)), t.actual_yield, "rx", label="actual")
    ax.set_xticks(range(len(t))); ax.set_xticklabels([f"{d[:3]}-{c[:3]}-{y}" for d, c, y in zip(t.district, t.crop, t.year)], rotation=60, fontsize=7); ax.set_ylabel("q/ha"); ax.legend(); ax.set_title("Conformal 90% intervals vs actual (test, district-crop-year)")
    plt.tight_layout(); plt.savefig(FIG / "21_conformal_uncertainty_bands.png", dpi=150); plt.close()

    # ================= 2. Fairness =================
    overall_mae = mean_absolute_error(test[TARGET], test.pred); overall_mape = mape(test[TARGET].values, test.pred.values)
    test["farm_size"] = pd.qcut(test.area_ha, 3, labels=["small", "medium", "large"])
    ref = test["hist5_mean_yield"]; test["risk"] = (test[TARGET] < 0.85 * ref).astype(int)
    rows = []
    for gcol in ["district", "crop", "season", "farm_size"]:
        for k, g in test.groupby(gcol, observed=True):
            m_ = mean_absolute_error(g[TARGET], g.pred); mp = mape(g[TARGET].values, g.pred.values); b = float((g.pred - g[TARGET]).mean())
            rows.append(dict(group_type=gcol, group=str(k), n=int(len(g)), mae=float(m_), mape_pct=mp, bias_q_ha=b, bias_pct=100 * b / float(g[TARGET].mean()),
                             flag_abs_mae=bool(m_ > 1.2 * overall_mae), flag_rel_error=bool(mp > 1.2 * overall_mape), flag_bias=bool(abs(100 * b / float(g[TARGET].mean())) > 5)))
    fair = pd.DataFrame(rows); out["fairness"] = dict(overall_mae=float(overall_mae), overall_mape_pct=overall_mape, table=fair.round(3).to_dict("records"))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, col, ttl, ov in [(axs[0], "mae", "Absolute MAE (q/ha)", overall_mae), (axs[1], "mape_pct", "Relative error MAPE (%)", overall_mape)]:
        ax.bar(fair.group_type + ":" + fair.group, fair[col], color=["#c0392b" if v > 1.2 * ov else "#2980b9" for v in fair[col]]); ax.axhline(ov, color="k", ls="--"); ax.axhline(1.2 * ov, color="r", ls=":")
        ax.set_title(ttl + " (red line = 20% worse than average)"); ax.tick_params(axis="x", rotation=75, labelsize=7)
    plt.tight_layout(); plt.savefig(FIG / "19_fairness_subgroup_disparity.png", dpi=150); plt.close()

    # ================= 3. Stability =================
    boot = [mean_absolute_error(*(lambda s: (test[TARGET].values[s], test.pred.values[s]))(rng.integers(0, len(test), len(test)))) for _ in range(300)]
    pert = test.copy(); m50 = rng.random(len(pert)) < 0.5
    for c in ["rainfall_last_14d", "rainfall_last_30d", "cum_rainfall_june_aug", "sowing_window_rainfall"]: pert.loc[m50, c] = np.nan
    pert.loc[m50, "rain_missing_aug"] = 1
    pm = mean_absolute_error(pert[TARGET], model.predict(pert[cols]))
    noisy = test.copy()
    for c in ["cum_rainfall_june_aug", "rainfall_last_30d"]: noisy[c] = noisy[c] * rng.normal(1, 0.10, len(noisy))
    nm = mean_absolute_error(noisy[TARGET], model.predict(noisy[cols]))
    psis = {c: psi(train[c].dropna().values, test[c].dropna().values) for c in WEATHER + ["soil_organic_carbon_pct", "soil_ph", "area_ha"] if c in train.columns}
    out["stability"] = dict(by_year={int(y): float(mean_absolute_error(g[TARGET], g.pred)) for y, g in test.groupby("year")}, bootstrap_mae_ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                            mae_50pct_rain_missing=float(pm), mae_change_missing_pct=float(100 * (pm - overall_mae) / overall_mae),
                            mae_10pct_rain_noise=float(nm), mae_change_noise_pct=float(100 * (nm - overall_mae) / overall_mae), psi=psis)
    fig, ax = plt.subplots(figsize=(9, 4)); s = pd.Series(psis).sort_values(); ax.barh(s.index, s.values, color=["#c0392b" if v > .25 else "#e67e22" if v > .1 else "#27ae60" for v in s]); ax.axvline(.1, ls=":"); ax.axvline(.25, ls="--")
    ax.set_title("PSI: train vs test feature drift (<0.1 stable, >0.25 major)"); plt.tight_layout(); plt.savefig(FIG / "20_psi_feature_drift.png", dpi=150); plt.close()

    # ================= 4. Recalibration (fit on validation, judged on test) =================
    lin = LinearRegression().fit(val[["pred"]], val[TARGET]); pl = lin.predict(test[["pred"]])
    iso = IsotonicRegression(out_of_bounds="clip").fit(val.pred, val[TARGET]); pi = iso.predict(test.pred)
    cal = dict(before_mae=float(overall_mae), linear_mae=float(mean_absolute_error(test[TARGET], pl)), isotonic_mae=float(mean_absolute_error(test[TARGET], pi)),
               linear_coef=[float(lin.intercept_), float(lin.coef_[0])], before_bias=float((test.pred - test[TARGET]).mean()), linear_bias=float((pl - test[TARGET]).mean()))
    cal["linear_change_pct"] = 100 * (cal["linear_mae"] - cal["before_mae"]) / cal["before_mae"]; cal["isotonic_change_pct"] = 100 * (cal["isotonic_mae"] - cal["before_mae"]) / cal["before_mae"]
    cal["adopt_recalibration"] = bool(min(cal["linear_change_pct"], cal["isotonic_change_pct"]) < -1.0)
    out["recalibration"] = cal
    fig, ax = plt.subplots(figsize=(5.5, 5)); bins = pd.qcut(test.pred, 15, duplicates="drop"); c_ = test.groupby(bins, observed=True).agg(p=("pred", "mean"), a=(TARGET, "mean"))
    ax.plot(c_.p, c_.a, "o-"); ax.plot([c_.p.min(), c_.p.max()], [c_.p.min(), c_.p.max()], "k--"); ax.set_xlabel("mean predicted"); ax.set_ylabel("mean actual"); ax.set_title("Calibration on test (binned)")
    plt.tight_layout(); plt.savefig(FIG / "22_calibration_curve.png", dpi=150); plt.close()
    json.dump(out, open(REPORTS_DIR / "procurement_fairness_results.json", "w"), indent=2, default=float)
    print("conformal q", round(q, 4), "coverage", out["conformal"]["test_group_coverage"]); print("flags 2024:", out["plan_2024_flags"], "drought:", out["plan_drought_flags"])
    print(fair[["group_type", "group", "mae", "mape_pct", "bias_pct", "flag_abs_mae", "flag_rel_error", "flag_bias"]].round(2).to_string(index=False)); print(cal)


if __name__ == "__main__":
    main()
