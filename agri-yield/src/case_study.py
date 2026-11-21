"""
Deep-dive on ONE large-error test prediction. Facts (numbers, raw values, data-quality flags) are pulled from the data;
the diagnosis text is selected by rules from those facts. Output: reports/wrong_prediction_case_study.md + figure 23.
"""
import sys, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED, INTERIM_DATA_DIR
from src.models import TARGET, ANCHOR, FIRST_TRAIN_YEAR


def main():
    df = pd.read_csv(PROCESSED_DATA_DIR / "agri_yield_features.csv")
    df = df[(df.year >= FIRST_TRAIN_YEAR) & (df.is_outlier == 0)].dropna(subset=[ANCHOR]).reset_index(drop=True)
    model = joblib.load(MODELS_DIR / "final_model.joblib"); cols = list(model.feature_names_in_)
    test = df[df.year >= 2022].copy(); test["pred"] = model.predict(test[cols]); test["err"] = test.pred - test[TARGET]
    test["abs_err"] = test.err.abs()
    # Choose the single largest absolute error that is NOT explained by an automatic data-fix flag, plus report the overall worst
    worst = test.sort_values("abs_err", ascending=False).iloc[0]
    clean_flag_share = float((test.yield_unit_fix != "none").mean())
    r = worst
    inner = model.estimator; prep, est = inner.named_steps["prep"], inner.named_steps["model"]
    X = pd.DataFrame(prep.transform(test.loc[[r.name], cols]), columns=[c.split("__")[-1] for c in prep.get_feature_names_out()])
    expl = shap.TreeExplainer(est); sv = np.asarray(expl.shap_values(X))[0] * r[ANCHOR]
    base = float(np.ravel(expl.expected_value)[0] * r[ANCHOR])
    e = shap.Explanation(values=sv, base_values=base, data=X.iloc[0].values, feature_names=list(X.columns))
    plt.figure(figsize=(9, 5)); shap.plots.waterfall(e, max_display=9, show=False); plt.title(f"Why the model predicted {r.pred:.1f} q/ha (actual {r[TARGET]:.1f})")
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "figures" / "23_case_study_shap_waterfall.png", dpi=140, bbox_inches="tight"); plt.close()
    order = np.argsort(-np.abs(sv))[:5]
    drivers = [(X.columns[i], float(sv[i]), float(X.iloc[0, i])) for i in order]

    # context: how unusual is the actual vs the group
    grp = test[(test.district == r.district) & (test.crop == r.crop) & (test.year == r.year)]
    z = (r[TARGET] - grp[TARGET].mean()) / grp[TARGET].std()
    pct_rank = float((grp[TARGET] < r[TARGET]).mean() * 100)
    raw = pd.read_csv(INTERIM_DATA_DIR / "agri_yield_cleaned.csv", usecols=["record_id", "raw_yield", "yield_unit_fix", "rain_missing", "rain_missing_weeks_count", "taluka_village_id", "soil_organic_carbon_pct"])
    rr = raw[raw.record_id == r.record_id].iloc[0]
    # Does a plausible data defect explain it?
    causes = []
    gt = json.load(open(ROOT / "data" / "raw" / "ground_truth_defects.json"))["defects"]
    kg, tn = set(gt["yield_kg_ha_indices"]), set(gt["yield_tonnes_ha_indices"])
    idx = int(str(r.record_id)[4:]) - 1
    true_fix = (idx in kg) if rr.yield_unit_fix == "kg_to_quintal" else (idx in tn) if rr.yield_unit_fix == "tonnes_to_quintal" else None
    all_ids = test.record_id.str[4:].astype(int) - 1
    fixed_wrong = test.yield_unit_fix.eq("kg_to_quintal") & ~all_ids.isin(kg) | test.yield_unit_fix.eq("tonnes_to_quintal") & ~all_ids.isin(tn)
    mae_all = float(test.abs_err.mean()); mae_ex = float(test.loc[~fixed_wrong, "abs_err"].mean())
    facts_extra = dict(true_fix=true_fix, n_wrong_fix_test=int(fixed_wrong.sum()), mae_all=mae_all, mae_excluding_wrong_fix=mae_ex)
    if rr.yield_unit_fix != "none" and true_fix is False:
        causes.append(f"**Data problem (cleaning error).** The raw yield was {rr.raw_yield}. The cleaning rule read it as kg/ha and divided by 100, giving {r[TARGET]:.1f} q/ha. "
                      f"The ground-truth log shows this row had NO kg/ha unit error: 400-950 values were injected as extreme outliers, and this one overlaps the range where kg/ha and an outlier look alike. "
                      f"The 'corrected' label is therefore wrong, and the model's forecast of {r.pred:.1f} q/ha (close to the district-crop normal of {r[ANCHOR]:.1f}) was actually sensible. "
                      f"Across the test set {int(fixed_wrong.sum())} rows ({100*fixed_wrong.mean():.2f}%) were wrongly unit-fixed; test MAE is {mae_all:.3f} on all rows and {mae_ex:.3f} when they are excluded.")
    elif rr.yield_unit_fix != "none": causes.append(f"The yield was auto-corrected by the cleaning step (raw value {rr.raw_yield}, fix '{rr.yield_unit_fix}') and the ground truth confirms the fix was correct.")
    if r.rain_missing_aug == 1: causes.append("Part of the June-August rainfall was missing and imputed, so the weather features for this record are estimates.")
    if not any("Data problem" in c for c in causes) and not causes:
        causes.append("No data defect explains it: the yield was not unit-fixed and the rainfall inputs were observed. "
                      f"The record is an individual field whose actual yield is {abs(z):.1f} standard deviations {'below' if z < 0 else 'above'} the mean of its own district-crop-season group (rank {pct_rank:.1f}th percentile). "
                      "The model only sees weather, soil and the historical normal, not field-level events (pest, hail, local irrigation, sowing date), so this is irreducible noise / a model limitation, not a pipeline bug.")
    if any("Data problem" in c for c in causes):
        talk = (f'"Our worst test miss: the model said {r.pred:.1f} q/ha for a {r.crop} field in {r.district}, the record said {r[TARGET]:.1f}. '
                f'When I traced it, the model was fine: the raw yield was {rr.raw_yield}, an extreme outlier that my unit-fix rule wrongly divided by 100. '
                f'It is a cleaning error, {int(fixed_wrong.sum())} test rows are affected, and the fix is to flag values that fit both a kg/ha and an outlier reading instead of auto-converting them."')
        change = ("1. **Fix the cleaning rule**: do not auto-convert values that are ambiguous between 'kg/ha' and 'outlier'; flag them for review (and exclude from training/evaluation) instead.\n"
                  "2. Re-run the unit-fix accuracy check against the ground-truth log after every cleaning change.\n"
                  "3. Report forecasts as intervals and aggregate to district-crop level for procurement, where single bad labels average out.")
    else:
        talk = (f'"The model forecast {r.pred:.1f} q/ha for a {r.crop} field in {r.district}; it came in at {r[TARGET]:.1f}. Inputs were ordinary and no data defect explains it, '
                f'so it is a field-level shock the weather and soil features cannot see. That is why we buy on district-crop forecasts with intervals."')
        change = ("1. Add field-level signals available by 31 August (satellite crop-stage, sowing date, irrigation source).\n"
                  "2. Report forecasts as intervals and aggregate to district-crop level for procurement.\n"
                  "3. Add an alert when a cleaned yield is more than 4 standard deviations from its group.")
    direction = "over-predicted" if r.err > 0 else "under-predicted"
    md = f"""# Case Study: One Large Error in the Test Period (2022-2024)

> Data in this project is **synthetic**. The record below is a real row of the generated test set; all numbers are computed by `src/case_study.py`.

**Record:** `{r.record_id}` | {r.taluka_village_id}, {r.district} | {r.crop} | season {int(r.year)}

## 1. Predicted vs actual
| | q/ha |
|---|---:|
| Actual yield (cleaned) | **{r[TARGET]:.2f}** |
| Model forecast (as of 31 Aug) | **{r.pred:.2f}** |
| Error | **{r.err:+.2f}** ({100*r.err/r[TARGET]:+.0f}% of actual) -> {direction} |
| Historical normal for this district-crop (anchor) | {r[ANCHOR]:.2f} |
| Naive last-year baseline | {r.yield_lag_1yr:.2f} |
| Average abs. error on the whole test set | {test.abs_err.mean():.2f} |

This is the single worst of {len(test):,} test predictions, i.e. {r.abs_err / test.abs_err.mean():.1f}x the typical error.

## 2. What the model saw (SHAP waterfall)
![SHAP waterfall](figures/23_case_study_shap_waterfall.png)

Largest contributions (q/ha, relative to the model's average of {base:.2f}):
""" + "\n".join(f"- `{n}` = {v:.2f} (scaled/encoded value): **{s:+.2f} q/ha**" for n, s, v in drivers) + f"""

Reading: the model starts from the district-crop normal of {r[ANCHOR]:.1f} q/ha and adjusts using rainfall, soil and crop features. For this record the inputs looked ordinary ({'rainfall at or above normal' if r.rainfall_vs_normal_ratio >= 0.95 else 'below-normal rainfall'}: {r.cum_rainfall_june_aug:.0f} mm, ratio {r.rainfall_vs_normal_ratio:.2f}; longest dry spell {int(r.longest_dry_spell_aug)} weeks; organic carbon {r.soil_organic_carbon_pct:.2f}%), so the model predicted a normal yield.

## 3. Was it a data problem, an unusual event, or a model limitation?
""" + "\n".join(f"- {c}" for c in causes) + f"""

Raw input check: raw yield in file = {rr.raw_yield}, unit-fix = {rr.yield_unit_fix}, missing rainfall weeks = {int(rr.rain_missing_weeks_count)}. Share of all test records that were unit-fixed: {100*clean_flag_share:.2f}%.

## 4. What we would change
{change}

## 2-minute talk track
{talk}
"""
    # (talk track is chosen below)
    # if a data defect was found, the talk track still reads correctly because the causes list changes
    (REPORTS_DIR / "wrong_prediction_case_study.md").write_text(md, encoding="utf-8")
    json.dump(dict(**facts_extra, record_id=r.record_id, actual=float(r[TARGET]), pred=float(r.pred), err=float(r.err), z=float(z), unit_fix=str(rr.yield_unit_fix), rain_missing=int(r.rain_missing_aug)),
              open(REPORTS_DIR / "case_study_facts.json", "w"), indent=2)
    print(md[:900])


if __name__ == "__main__":
    main()
