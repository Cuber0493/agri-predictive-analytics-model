"""
Final evaluation (the ONLY place the test period 2022-2024 is used), PASS/FAIL checks, high-risk view and explainability.
Every number is computed here and saved to reports/eval_results.json; nothing is typed in by hand.
"""
import sys, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, precision_recall_curve, average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED
from src.models import (TARGET, ANCHOR, CV_FOLDS, FIRST_TRAIN_YEAR, TRAIN_END, VAL_YEARS, DEFAULT_SET, cols_for, make_model, baseline_preds)
FIG = REPORTS_DIR / "figures"; FIG.mkdir(parents=True, exist_ok=True)
sns_ok = True

EXPECTED_SIGN = {  # agronomic expectation: +1 higher feature -> higher yield ratio ; -1 lower
    "cum_rainfall_june_aug": +1, "rainfall_vs_normal_ratio": +1, "rainfall_last_30d": +1, "rainfall_last_14d": +1,
    "sowing_window_rainfall": +1, "rainy_weeks_count": +1, "soil_organic_carbon_pct": +1,
    "longest_dry_spell_aug": -1, "heat_stress_days": -1}
CONTEXT_OK = {"crop", "district", "season", "area_ha", "lag1_vs_anchor", "std3_vs_anchor", "soil_n_kg_ha", "soil_p_kg_ha",
              "soil_k_kg_ha", "soil_ph", "soil_n_p_ratio", "soil_n_k_ratio", "soil_p_k_ratio", "rain_missing_aug"}


def smape(a, p): return float(np.mean(2 * np.abs(p - a) / (np.abs(a) + np.abs(p) + 1e-9)) * 100)


def metrics(d, p):
    y = d[TARGET].values
    return dict(mae=float(mean_absolute_error(y, p)), rmse=float(root_mean_squared_error(y, p)), r2=float(r2_score(y, p)),
                smape=smape(y, p), bias=float(np.mean(p - y)), n=int(len(d)))


def recall_at_precision(y_true, score, min_prec=0.70):
    if y_true.sum() == 0: return dict(recall=None, precision=None, pr_auc=None, roc_auc=None, threshold=None, n_pos=0)
    pr, rc, th = precision_recall_curve(y_true, score)
    ok = np.where(pr[:-1] >= min_prec)[0]
    if len(ok): i = ok[np.argmax(rc[ok])]; r, p, t = float(rc[i]), float(pr[i]), float(th[i])
    else: r, p, t = 0.0, None, None
    return dict(recall=r, precision=p, pr_auc=float(average_precision_score(y_true, score)), roc_auc=float(roc_auc_score(y_true, score)),
                threshold=t, n_pos=int(y_true.sum()), prevalence=float(y_true.mean()))


def load():
    df = pd.read_csv(PROCESSED_DATA_DIR / "agri_yield_features.csv")
    df = df[(df.year >= FIRST_TRAIN_YEAR) & (df.is_outlier == 0)].dropna(subset=[ANCHOR]).reset_index(drop=True)
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    return df, model


def main():
    df, model = load()
    cols = list(model.feature_names_in_)
    train = df[df.year <= TRAIN_END]; val = df[df.year.isin(VAL_YEARS)]; test = df[df.year >= 2022].copy(); val = val.copy()
    res = {}
    for name, d in [("val", val), ("test", test)]:
        p = model.predict(d[cols]); d["pred"] = p
        m = metrics(d, p)
        for b, kind in [("baseline_last_year", "naive_last_year"), ("baseline_normal", "normal")]:
            bm = metrics(d, baseline_preds(d, kind)); m[b] = bm
            m[f"gain_vs_{b}_pct"] = 100 * (bm["mae"] - m["mae"]) / bm["mae"]
        m["by_year"] = {int(y): dict(model=float(mean_absolute_error(g[TARGET], g.pred)), last_year_baseline=float(mean_absolute_error(g[TARGET], baseline_preds(g, "naive_last_year"))),
                                     normal_baseline=float(mean_absolute_error(g[TARGET], baseline_preds(g, "normal")))) for y, g in d.groupby("year")}
        # tonnes of procurement error (sampled farm records, 1 quintal = 0.1 t)
        d["act_t"] = d[TARGET] * d.area_ha / 10; d["pred_t"] = d.pred * d.area_ha / 10
        yrs = d.year.nunique()
        agg = d.groupby("district").apply(lambda g: pd.Series(dict(actual_t=g.act_t.sum() / yrs, pred_t=g.pred_t.sum() / yrs,
                        abs_err_t=(g.pred_t - g.act_t).abs().sum() / yrs, net_err_t=(g.pred_t - g.act_t).sum() / yrs)))
        agg["abs_err_pct"] = 100 * agg.abs_err_t / agg.actual_t; agg["net_err_pct"] = 100 * agg.net_err_t / agg.actual_t
        m["procurement_by_district_t_per_year"] = agg.round(2).reset_index().to_dict("records")
        m["procurement_total_abs_err_t_per_year"] = float(agg.abs_err_t.sum()); m["procurement_total_net_err_pct"] = float(100 * agg.net_err_t.sum() / agg.actual_t.sum())
        res[name] = m
    val_mae, test_mae = res["val"]["mae"], res["test"]["mae"]
    res["val_test_gap_pct"] = 100 * (test_mae - val_mae) / val_mae

    # ---------------- High-risk view: yield > 15% below own 5-season average ----------------
    def risk_view(d, p):
        ref = d["hist5_mean_yield"].values; lab = (d[TARGET].values < 0.85 * ref).astype(int)
        return recall_at_precision(lab, 1 - p / ref), recall_at_precision(lab, 1 - d["yield_lag_1yr"].fillna(d[ANCHOR]).values / ref)
    hr = {}
    hr["val"], hr["val_baseline"] = risk_view(val, val.pred.values)
    hr["test"], hr["test_baseline"] = risk_view(test, test.pred.values)
    # out-of-fold CV predictions cover drought years 2014, 2015, 2018 (the test period has no drought)
    meta = json.load(open(MODELS_DIR / "model_meta.json")); params = meta["params"]; oof = []
    for tr_y, va_y in CV_FOLDS:
        tr = df[df.year.isin(list(tr_y))]; va = df[df.year.isin(va_y)].copy()
        m = make_model(meta["model_kind"], meta["feature_set"], lgb_params=params); m.fit(tr[cols], tr[TARGET]); va["pred"] = m.predict(va[cols]); oof.append(va)
    oof = pd.concat(oof)
    hr["oof_cv_drought_years"], hr["oof_cv_baseline"] = risk_view(oof, oof.pred.values)
    res["high_risk"] = hr
    # PR curve figure
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab, d, p, c in [("Model, CV folds incl. droughts 2014/15/18", oof, oof.pred.values, "#27ae60"), ("Model, test 2022-24", test, test.pred.values, "#2980b9")]:
        ref = d["hist5_mean_yield"].values; y_ = (d[TARGET].values < 0.85 * ref).astype(int)
        if y_.sum():
            pr, rc, _ = precision_recall_curve(y_, 1 - p / ref); ax.plot(rc, pr, label=lab, color=c)
    ref = oof["hist5_mean_yield"].values; y_ = (oof[TARGET].values < 0.85 * ref).astype(int)
    pr, rc, _ = precision_recall_curve(y_, 1 - oof.yield_lag_1yr.fillna(oof[ANCHOR]).values / ref); ax.plot(rc, pr, "--", color="#7f8c8d", label="Last-year baseline, CV folds")
    ax.axhline(0.7, color="red", ls=":", label="Precision = 70%"); ax.set_xlabel("Recall (share of bad seasons caught)"); ax.set_ylabel("Precision")
    ax.set_title("High-risk detection: yield >15% below 5-season average"); ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / "13_precision_recall_curve.png", dpi=150); plt.close()

    # ---------------- SHAP (inner LightGBM; values are in yield-ratio units, converted to q/ha with the anchor) ----------------
    inner = model.estimator; prep, lgb = inner.named_steps["prep"], inner.named_steps["model"]
    assert hasattr(lgb, "estimators_") or hasattr(lgb, "booster_"), "SHAP expects a tree model"
    samp = test.sample(4000, random_state=RANDOM_SEED)
    Xt = pd.DataFrame(prep.transform(samp[cols]), columns=[c.split("__")[-1] for c in prep.get_feature_names_out()])
    expl = shap.TreeExplainer(lgb); sv = expl.shap_values(Xt)
    sv_q = sv * samp[ANCHOR].values[:, None]
    imp = pd.Series(np.abs(sv_q).mean(0), index=Xt.columns).sort_values(ascending=False)
    top = list(imp.index[:5]); drivers = []
    for f in imp.index[:10]:
        rho = float(spearmanr(Xt[f], sv_q[:, list(Xt.columns).index(f)])[0])
        drivers.append(dict(feature=f, mean_abs_shap_q_ha=float(imp[f]), direction_spearman=rho, expected_sign=EXPECTED_SIGN.get(f)))
    ok = []
    for d_ in drivers[:5]:
        e = d_["expected_sign"]
        d_["sensible"] = bool((e is not None and np.sign(d_["direction_spearman"]) == e) or (e is None and d_["feature"] in CONTEXT_OK))
        ok.append(d_["sensible"])
    res["shap_top_drivers"] = drivers; res["criterion_c_pass"] = bool(all(ok))
    plt.figure(); shap.summary_plot(sv_q, Xt, show=False, max_display=12); plt.title("SHAP summary (impact on yield, q/ha)"); plt.tight_layout()
    plt.savefig(FIG / "14_shap_summary.png", dpi=140, bbox_inches="tight"); plt.close()
    plt.figure(); shap.summary_plot(sv_q, Xt, plot_type="bar", show=False, max_display=12); plt.title("Mean |SHAP| (q/ha)"); plt.tight_layout()
    plt.savefig(FIG / "14_shap_summary_and_bar.png", dpi=140, bbox_inches="tight"); plt.close()
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    for ax, f in zip(axs, top[:4]):
        j = list(Xt.columns).index(f); ax.scatter(Xt[f], sv_q[:, j], s=4, alpha=0.3); ax.set_title(f); ax.set_xlabel("feature value (encoded/scaled as in model)"); ax.set_ylabel("SHAP (q/ha)")
    plt.tight_layout(); plt.savefig(FIG / "15_shap_dependence_top_features.png", dpi=140); plt.close()

    # ---------------- PDP + ICE in business units (manual, uses the full pipeline) ----------------
    base = test.sample(1500, random_state=RANDOM_SEED)
    pdp_feats = [f for f in top if f in base.columns and base[f].nunique() > 6][:5]
    fig, axs = plt.subplots(1, len(pdp_feats), figsize=(4 * len(pdp_feats), 3.8))
    axs = np.atleast_1d(axs)
    for ax, f in zip(axs, pdp_feats):
        grid = np.unique(np.quantile(train[f].dropna(), np.linspace(0.02, 0.98, 25)))
        ice = np.array([model.predict(base[cols].assign(**{f: g})) for g in grid]).T
        for r in ice[:60]: ax.plot(grid, r, color="#95a5a6", alpha=0.25, lw=0.7)
        ax.plot(grid, ice.mean(0), color="#c0392b", lw=2.5, label="average (PDP)"); ax.set_xlabel(f); ax.set_ylabel("predicted yield (q/ha)")
    axs[0].legend(); plt.suptitle("Partial dependence (red) and ICE (grey) of the top drivers"); plt.tight_layout(); plt.savefig(FIG / "16_pdp_ice_top_features.png", dpi=140); plt.close()
    # waterfalls for the best-explained and worst-error rows
    errs = (samp.pred - samp[TARGET]).abs() if "pred" in samp else None
    samp = samp.assign(err=(samp["pred"] - samp[TARGET]))
    idxs = [int(np.argmin(samp.err.abs().values)), int(np.argmax(samp.err.abs().values))]
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    for ax, i, t in zip(axs, idxs, ["Typical (small error)", "Large error"]):
        plt.sca(ax); e = shap.Explanation(values=sv_q[i], base_values=float(expl.expected_value * samp[ANCHOR].values[i]), data=Xt.iloc[i].values, feature_names=list(Xt.columns))
        shap.plots.waterfall(e, max_display=8, show=False); plt.title(f"{t}: actual {samp[TARGET].values[i]:.1f}, predicted {samp.pred.values[i]:.1f} q/ha")
    plt.tight_layout(); plt.savefig(FIG / "17_shap_waterfall_examples.png", dpi=130, bbox_inches="tight"); plt.close()

    # ---------------- PASS / FAIL ----------------
    prim_val, prim_test = res["val"]["gain_vs_baseline_last_year_pct"], res["test"]["gain_vs_baseline_last_year_pct"]
    nor_val, nor_test = res["val"]["gain_vs_baseline_normal_pct"], res["test"]["gain_vs_baseline_normal_pct"]
    pf = [
        dict(criterion="(a) Final model beats baseline by >= 15% (MAE)", result="PASS" if min(prim_val, prim_test) >= 15 and min(nor_val, nor_test) >= 15 else "FAIL",
             detail=f"vs last-year baseline: val {prim_val:+.1f}%, test {prim_test:+.1f}% | vs stronger 5-season-normal baseline: val {nor_val:+.1f}%, test {nor_test:+.1f}%"),
        dict(criterion="(b) Test MAE within 5% of validation MAE", result="PASS" if abs(res["val_test_gap_pct"]) <= 5 else "FAIL",
             detail=f"val {val_mae:.3f} vs test {test_mae:.3f} q/ha (gap {res['val_test_gap_pct']:+.1f}%)"),
        dict(criterion="(c) Top-5 SHAP drivers make agronomic sense", result="PASS" if res["criterion_c_pass"] else "FAIL",
             detail="; ".join(f"{d['feature']} ({'ok' if d['sensible'] else 'CHECK'})" for d in drivers[:5]))]
    res["pass_fail"] = pf
    print(pd.DataFrame(pf).to_string(index=False))
    test[["record_id", "year", "district", "crop", "taluka_village_id", "area_ha", TARGET, "pred", ANCHOR, "hist5_mean_yield", "yield_lag_1yr", "price_lag_1yr"]].to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    val[["record_id", "year", "district", "crop", "area_ha", TARGET, "pred", ANCHOR, "hist5_mean_yield"]].to_csv(REPORTS_DIR / "val_predictions.csv", index=False)
    json.dump(res, open(REPORTS_DIR / "eval_results.json", "w"), indent=2, default=float)
    print(f"val MAE {val_mae:.4f} | test MAE {test_mae:.4f} | gap {res['val_test_gap_pct']:+.2f}%")


if __name__ == "__main__":
    main()
