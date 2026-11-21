"""
Model comparison, feature-set / encoder benchmarks, Optuna tuning and final model export.

STRICT RULE: this script only ever loads seasons <= 2021 (train 2008-2018 + validation 2019-2021).
The test period (2022-2024) is dropped immediately after loading and is first used in evaluate.py.

Outputs (all numbers used in reports come from these files, nothing is hard-coded):
    models/final_model.joblib, models/model_meta.json, reports/model_results.json,
    reports/figures/11_expanding_window_cv.png, reports/figures/12_model_comparison_mae.png
"""
import sys, json, time, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED
from src.utils import set_seed
from src.anchored import AnchoredRegressor

FIG_DIR = REPORTS_DIR / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = "yield_quintal_per_ha"
ANCHOR = "yield_anchor_5yr_p75"
FIRST_TRAIN_YEAR = 2008          # 2006-07 have no real history for lags (see model card)
TRAIN_END, VAL_YEARS = 2018, [2019, 2020, 2021]
CV_FOLDS = [(range(2008, 2013), [2013, 2014]), (range(2008, 2015), [2015, 2016]), (range(2008, 2017), [2017, 2018])]

WEATHER = ["cum_rainfall_june_aug", "rainfall_last_14d", "rainfall_last_30d", "sowing_window_rainfall",
           "rainfall_vs_normal_ratio", "longest_dry_spell_aug", "rainy_weeks_count", "heat_stress_days", "rain_missing_aug"]
SOIL = ["soil_organic_carbon_pct", "soil_n_kg_ha", "soil_p_kg_ha", "soil_k_kg_ha", "soil_ph",
        "soil_n_p_ratio", "soil_n_k_ratio", "soil_p_k_ratio", "area_ha"]
LAGS_RATIO = ["lag1_vs_anchor", "std3_vs_anchor"]
ORIGINAL_LAGS = ["yield_lag_1yr", "yield_lag_2yr", "price_lag_1yr", "price_lag_2yr", "area_lag_1yr",
                 "yield_rolling_3yr_mean", "yield_rolling_3yr_std", "price_vs_prev_year_ratio", "yield_vs_district_avg_lag"]
WEEKLY = [f"rainfall_w{i:02d}" for i in range(1, 13)]
CAT_LOW = ["district", "crop", "season"]
HIGH_CARD = "village_grouped"

FEATURE_SETS = {
    "A: weather + soil + crop/district": dict(num=WEATHER + SOIL, cat=CAT_LOW, high=None),
    "B: A + lag ratios": dict(num=WEATHER + SOIL + LAGS_RATIO, cat=CAT_LOW, high=None),
    "C: B + village (target enc.)": dict(num=WEATHER + SOIL + LAGS_RATIO, cat=CAT_LOW, high="te"),
    "D: all 40 original features": dict(num=WEATHER + SOIL + LAGS_RATIO + ORIGINAL_LAGS + WEEKLY, cat=CAT_LOW, high="te"),
}
DEFAULT_SET = "B: A + lag ratios"


def mae(a, b): return float(mean_absolute_error(a, b))


def build_prep(num, cat, high=None, linear=False):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if linear else []))
    cat_enc = (OneHotEncoder(handle_unknown="ignore", sparse_output=False) if linear
               else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    parts = [("num", num_pipe, num), ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", cat_enc)]), cat)]
    if high == "te":      # smoothed target encoding, fitted only on training rows (internal 5-fold)
        parts.append(("hc", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                      ("enc", TargetEncoder(smooth="auto", cv=5, target_type="continuous", random_state=RANDOM_SEED))]), [HIGH_CARD]))
    elif high == "ohe":
        parts.append(("hc", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), [HIGH_CARD]))
    return ColumnTransformer(parts)


def lgbm(**kw):
    p = dict(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=300, subsample=0.7,
             subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, verbose=-1, n_jobs=1, random_state=RANDOM_SEED)
    p.update(kw); return LGBMRegressor(**p)


def make_model(kind, fs=DEFAULT_SET, lgb_params=None, anchored=True, high_override=None):
    spec = FEATURE_SETS[fs]; high = spec["high"] if high_override is None else high_override
    if kind == "ridge":
        est = Ridge(**{**dict(alpha=10.0), **(lgb_params or {})}); prep = build_prep(spec["num"], spec["cat"], high, linear=True)
    elif kind == "rf":
        est = RandomForestRegressor(**{**dict(n_estimators=60, max_depth=12, min_samples_leaf=50, max_features=0.5,
                                    max_samples=0.3, n_jobs=1, random_state=RANDOM_SEED), **(lgb_params or {})})
        prep = build_prep(spec["num"], spec["cat"], high)
    elif kind == "lgbm":
        est = lgbm(**(lgb_params or {})); prep = build_prep(spec["num"], spec["cat"], high)
    pipe = Pipeline([("prep", prep), ("model", est)])
    return AnchoredRegressor(pipe, ANCHOR) if anchored else pipe


def cols_for(fs, extra=()):
    s = FEATURE_SETS[fs]
    return list(dict.fromkeys(s["num"] + s["cat"] + ([HIGH_CARD] if s["high"] else []) + [ANCHOR] + list(extra)))


def baseline_preds(d, kind):
    if kind == "naive_last_year":
        return d["yield_lag_1yr"].fillna(d[ANCHOR]).values
    return d[ANCHOR].values            # "historical normal": 5-season 75th percentile


def fit_predict(model, tr, te, cols, anchored=True):
    if anchored:
        model.fit(tr[cols], tr[TARGET]); return model.predict(te[cols])
    model.fit(tr[cols], tr[TARGET]); return model.predict(te[cols])


def cv_scores(df, builder, cols):
    out = []
    for tr_years, va_years in CV_FOLDS:
        tr = df[df.year.isin(list(tr_years))]; va = df[df.year.isin(va_years)]
        m = builder(); m.fit(tr[cols], tr[TARGET]); out.append(mae(va[TARGET], m.predict(va[cols])))
    return out


def main(n_trials=15):
    set_seed(RANDOM_SEED); t0 = time.time()
    df = pd.read_csv(PROCESSED_DATA_DIR / "agri_yield_features.csv")
    df = df[(df.year <= 2021) & (df.year >= FIRST_TRAIN_YEAR) & (df.get("is_outlier", 0) == 0)].copy()   # test period never loaded
    df = df.dropna(subset=[ANCHOR]).reset_index(drop=True)
    train = df[df.year <= TRAIN_END]; val = df[df.year.isin(VAL_YEARS)]
    print(f"train rows {len(train):,} | val rows {len(val):,} | years {train.year.min()}-{val.year.max()}")
    res = {"split": {"train_years": f"{FIRST_TRAIN_YEAR}-{TRAIN_END}", "val_years": "2019-2021", "test_years": "2022-2024",
                     "n_train": int(len(train)), "n_val": int(len(val))}}

    # ---------- 1. Baselines ----------
    rows = []
    for name, kind in [("Baseline 1: last year's yield", "naive_last_year"), ("Baseline 2: 5-season normal (p75)", "normal")]:
        folds = []
        for tr_years, va_years in CV_FOLDS:
            v = df[df.year.isin(va_years)]; folds.append(mae(v[TARGET], baseline_preds(v, kind)))
        vm = mae(val[TARGET], baseline_preds(val, kind)); vp = baseline_preds(val, kind)
        rows.append(dict(model=name, cv_folds=folds, cv_mean=float(np.mean(folds)), cv_std=float(np.std(folds)), val_mae=vm,
                         val_rmse=float(root_mean_squared_error(val[TARGET], vp)), val_r2=float(r2_score(val[TARGET], vp))))
    # ---------- 2. Feature-set + high-cardinality encoder benchmark (LightGBM, CV on 2008-2018 only) ----------
    fs_rows = []
    for fs in FEATURE_SETS:
        print(f"[*] feature set {fs}", flush=True)
        c = cols_for(fs); f = cv_scores(df, lambda fs=fs: make_model("lgbm", fs), c)
        fs_rows.append(dict(feature_set=fs, n_features=len(FEATURE_SETS[fs]["num"]) + len(FEATURE_SETS[fs]["cat"]) + (1 if FEATURE_SETS[fs]["high"] else 0),
                            cv_folds=f, cv_mean=float(np.mean(f)), cv_std=float(np.std(f))))
    res["feature_set_comparison"] = fs_rows
    sel = min(fs_rows, key=lambda r: r["cv_mean"])["feature_set"]; res["selected_feature_set"] = sel
    print(f"[✓] feature set chosen by CV: {sel}")
    enc_rows = []; spec_b = FEATURE_SETS[sel]
    for label, high in [("No village feature", None), ("Smoothed target encoding", "te"), ("One-hot (min 50 per level)", "ohe")]:
        c = list(dict.fromkeys(spec_b["num"] + spec_b["cat"] + ([HIGH_CARD] if high else []) + [ANCHOR]))
        f = cv_scores(df, lambda h=high: make_model("lgbm", sel, high_override=h), c)
        enc_rows.append(dict(encoder=label, cv_mean=float(np.mean(f)), cv_std=float(np.std(f))))
    f = []
    for tr_years, va_years in CV_FOLDS:   # frequency encoding fitted on the training part of each fold only
        tr = df[df.year.isin(list(tr_years))].copy(); va = df[df.year.isin(va_years)].copy()
        fm = tr[HIGH_CARD].value_counts(normalize=True)
        tr["village_freq"] = tr[HIGH_CARD].map(fm).fillna(0); va["village_freq"] = va[HIGH_CARD].map(fm).fillna(0)
        c = spec_b["num"] + ["village_freq"] + spec_b["cat"] + [ANCHOR]
        pipe = Pipeline([("prep", build_prep(spec_b["num"] + ["village_freq"], spec_b["cat"])), ("model", lgbm())])
        m = AnchoredRegressor(pipe, ANCHOR); m.fit(tr[c], tr[TARGET]); f.append(mae(va[TARGET], m.predict(va[c])))
    enc_rows.append(dict(encoder="Frequency encoding", cv_mean=float(np.mean(f)), cv_std=float(np.std(f))))
    res["village_encoder_comparison"] = enc_rows

    # ---------- 3. Model comparison on the selected feature set ----------
    cols = cols_for(sel)
    for name, kind in [("Ridge", "ridge"), ("Random Forest", "rf"), ("LightGBM", "lgbm")]:
        print(f"[*] {name} ...", flush=True)
        folds = cv_scores(df, lambda k=kind: make_model(k, sel), cols)
        m = make_model(kind, sel); m.fit(train[cols], train[TARGET]); vp = m.predict(val[cols])
        rows.append(dict(model=name, cv_folds=folds, cv_mean=float(np.mean(folds)), cv_std=float(np.std(folds)),
                         val_mae=mae(val[TARGET], vp), val_rmse=float(root_mean_squared_error(val[TARGET], vp)), val_r2=float(r2_score(val[TARGET], vp))))
    print("[*] LightGBM without anchor (ablation) ...", flush=True)
    raw_cols = cols_for("D: all 40 original features")
    raw_builder = lambda: make_model("lgbm", "D: all 40 original features", anchored=False)
    folds = cv_scores(df, raw_builder, raw_cols)
    m = raw_builder(); m.fit(train[raw_cols], train[TARGET]); vp = m.predict(val[raw_cols])
    rows.append(dict(model="Ablation: LightGBM, raw yield, 40 features", cv_folds=folds, cv_mean=float(np.mean(folds)), cv_std=float(np.std(folds)),
                     val_mae=mae(val[TARGET], vp), val_rmse=float(root_mean_squared_error(val[TARGET], vp)), val_r2=float(r2_score(val[TARGET], vp))))
    res["model_comparison"] = rows
    for r in rows: print(f"   {r['model']:<45} CV {r['cv_mean']:.3f} ± {r['cv_std']:.3f} | VAL {r['val_mae']:.3f}")

    # ---------- 4. Select best by CV (never by validation/test) and tune it with Optuna ----------
    cand = [r for r in rows if r["model"] in ("Ridge", "Random Forest", "LightGBM")]
    best = min(cand, key=lambda r: r["cv_mean"]); res["best_model_by_cv"] = best["model"]
    kind = {"Ridge": "ridge", "Random Forest": "rf", "LightGBM": "lgbm"}[best["model"]]
    print(f"[✓] best by CV: {best['model']}  -> tuning it", flush=True)
    def objective(trial):
        if kind == "lgbm":
            p = dict(n_estimators=trial.suggest_int("n_estimators", 150, 450), learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                     num_leaves=trial.suggest_int("num_leaves", 6, 48), min_child_samples=trial.suggest_int("min_child_samples", 100, 900),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                     reg_lambda=trial.suggest_float("reg_lambda", 1.0, 50.0, log=True))
        elif kind == "rf":
            p = dict(n_estimators=trial.suggest_int("n_estimators", 40, 100), max_depth=trial.suggest_int("max_depth", 5, 16),
                     min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 300), max_features=trial.suggest_float("max_features", 0.3, 0.9),
                     max_samples=trial.suggest_float("max_samples", 0.1, 0.3))
        else:
            p = dict(alpha=trial.suggest_float("alpha", 0.1, 500, log=True))
        return float(np.mean(cv_scores(df, lambda: make_model(kind, sel, lgb_params=p), cols)))
    print(f"[*] Optuna {n_trials} trials (time-aware CV on 2008-2018 only) ...", flush=True)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials)
    bp = study.best_params
    final = make_model(kind, sel, lgb_params=bp); final.fit(train[cols], train[TARGET]); vp = final.predict(val[cols])
    res["tuning"] = dict(model=best["model"], n_trials=n_trials, best_params=bp, cv_mae_untuned=best["cv_mean"], cv_mae_tuned=float(study.best_value),
                         val_mae_untuned=best["val_mae"], val_mae_tuned=mae(val[TARGET], vp))
    res["final_val"] = dict(mae=mae(val[TARGET], vp), rmse=float(root_mean_squared_error(val[TARGET], vp)), r2=float(r2_score(val[TARGET], vp)))
    print(f"    tuned CV {study.best_value:.4f} (untuned {best['cv_mean']:.4f}) | val {res['final_val']['mae']:.4f}")

    joblib.dump(final, MODELS_DIR / "final_model.joblib")
    json.dump(dict(features=cols, anchor=ANCHOR, target=TARGET, model_kind=kind, feature_set=sel, params=bp, trained_on=res["split"]["train_years"]),
              open(MODELS_DIR / "model_meta.json", "w"), indent=2)
    json.dump(res, open(REPORTS_DIR / "model_results.json", "w"), indent=2)

    # ---------- figures ----------
    names = [r["model"].replace("Ablation: ", "Abl: ") for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w/2, [r["cv_mean"] for r in rows], w, yerr=[r["cv_std"] for r in rows], capsize=4, label="CV mean ± std (2013-2018 folds)", color="#2980b9")
    ax.bar(x + w/2, [r["val_mae"] for r in rows], w, label="Validation 2019-2021", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels([n.replace(": ", ":\n") for n in names], fontsize=8); ax.set_ylabel("MAE (quintal/ha)")
    ax.set_title("Model comparison: MAE in business units (lower is better)"); ax.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "12_model_comparison_mae.png", dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(9, 3.2)); yrs = list(range(2008, 2019))
    ax.broken_barh([(2008, 5)], (0.7, 0.5), color="#2980b9"); ax.broken_barh([(2013, 2)], (0.7, 0.5), color="#e67e22")
    ax.broken_barh([(2008, 7)], (1.7, 0.5), color="#2980b9"); ax.broken_barh([(2015, 2)], (1.7, 0.5), color="#e67e22")
    ax.broken_barh([(2008, 9)], (2.7, 0.5), color="#2980b9"); ax.broken_barh([(2017, 2)], (2.7, 0.5), color="#e67e22")
    ax.set_yticks([0.95, 1.95, 2.95]); ax.set_yticklabels(["Fold 1", "Fold 2", "Fold 3"]); ax.set_xlim(2008, 2019); ax.set_title("Expanding-window CV (blue = train, orange = validate)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "11_expanding_window_cv.png", dpi=150); plt.close()
    print(f"[✓] models.py finished in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(n_trials=int(sys.argv[1]) if len(sys.argv) > 1 else 15)
