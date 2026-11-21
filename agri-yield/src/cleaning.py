"""
Data Cleaning and Quality Restoration Pipeline for Agri Predictive Analytics Model.

Performs:
1) Deduplication: Removes redundant records.
2) Spelling Standardization: Harmonizes district and crop naming conventions.
3) Date Harmonization: Standardizes mixed date representations into ISO 8601 (YYYY-MM-DD).
4) Price Unit Normalization: Detects Rs/kg vs Rs/quintal via commodity baseline distributions,
   converts to Rs/quintal, and benchmarks detection accuracy against ground truth.
5) Yield Unit Normalization: Utilizes crop-specific biological yield ranges to detect
   kg/ha and tonnes/ha, converting all to quintals/ha.
6) Rainfall Imputation: Adds 'rain_missing' flag, preserves raw values, and imputes
   missing weekly values via localized district-year-season spatial/temporal groupings.
7) Outlier Flagging: Tags physical and measurement outliers without destructive deletion.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, INTERIM_DATA_DIR, RANDOM_SEED
from src.utils import set_seed


# -------------------------------------------------------------
# MAPPING DICTIONARIES & DOMAIN CONSTANTS
# -------------------------------------------------------------
DISTRICT_CORRECTION_MAP = {
    # Nagpur
    "nagpr": "Nagpur", "nagpurr": "Nagpur", "nagpoor": "Nagpur", "nagpur": "Nagpur",
    # Wardha
    "wardhaa": "Wardha", "vardha": "Wardha", "wardha": "Wardha",
    # Amravati
    "amrawati": "Amravati", "amaravati": "Amravati", "amravati": "Amravati",
    # Yavatmal
    "yeotmal": "Yavatmal", "yavatmaal": "Yavatmal", "yawatmal": "Yavatmal", "yavatmal": "Yavatmal",
    # Chandrapur
    "chanda": "Chandrapur", "chandrapurr": "Chandrapur", "chandrapur": "Chandrapur",
    # Bhandara
    "bhandaraa": "Bhandara", "bhandra": "Bhandara", "bhandara": "Bhandara",
}

CROP_CORRECTION_MAP = {
    # Soybean
    "soyabean": "soybean", "soya_bean": "soybean", "soyabeen": "soybean", "soybean": "soybean",
    # Cotton
    "coton": "cotton", "kapas": "cotton", "cotton": "cotton",
    # Tur / Pigeon pea
    "arhar": "tur", "tuar": "tur", "toor": "tur", "tur": "tur",
    # Rice / Paddy
    "paddy": "rice", "dhan": "rice", "rice": "rice",
    # Wheat
    "wheatt": "wheat", "gehun": "wheat", "wheat": "wheat",
    # Gram / Chickpea
    "chana": "gram", "channa": "gram", "gram": "gram",
    # Jowar / Sorghum
    "jowarr": "jowar", "sorghum": "jowar", "jowar": "jowar",
}

# Crop-specific biological yield thresholds (quintals/ha)
# Values below ton_max are deemed recorded in tonnes/ha (multiply by 10)
# Values above kg_min are deemed recorded in kg/ha (divide by 100)
CROP_YIELD_THRESHOLDS = {
    "cotton":  {"ton_max": 2.45, "kg_min": 100.0, "normal_min": 2.5, "normal_max": 50.0},
    "gram":    {"ton_max": 2.05, "kg_min": 100.0, "normal_min": 2.0, "normal_max": 35.0},
    "jowar":   {"ton_max": 2.35, "kg_min": 100.0, "normal_min": 2.5, "normal_max": 45.0},
    "rice":    {"ton_max": 4.35, "kg_min": 100.0, "normal_min": 4.5, "normal_max": 75.0},
    "soybean": {"ton_max": 2.30, "kg_min": 100.0, "normal_min": 2.5, "normal_max": 45.0},
    "tur":     {"ton_max": 1.90, "kg_min": 100.0, "normal_min": 2.0, "normal_max": 30.0},
    "wheat":   {"ton_max": 3.80, "kg_min": 100.0, "normal_min": 4.0, "normal_max": 65.0},
}


def clean_agri_dataset(
    raw_csv_path: Path = RAW_DATA_DIR / "agri_yield_raw.csv",
    gt_json_path: Path = RAW_DATA_DIR / "ground_truth_defects.json",
    output_dir: Path = INTERIM_DATA_DIR,
) -> tuple[pd.DataFrame, dict]:
    """
    Cleans raw agricultural data and outputs cleaned data and validation metrics.
    """
    start_time = time.time()
    set_seed(RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AGRI-YIELD DATA CLEANING & QUALITY RESTORATION PIPELINE")
    print("=" * 65)

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_csv_path}")

    # Load Ground Truth if available for benchmarking
    gt_data = None
    if gt_json_path.exists():
        with open(gt_json_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        print(f"[✓] Loaded ground-truth audit log for precision/recall validation.")

    # 1. LOAD DATA
    print(f"[*] Loading raw dataset: {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    initial_rows = len(df)
    initial_cols = len(df.columns)
    print(f"    - Initial rows: {initial_rows:,}")
    print(f"    - Initial columns: {initial_cols}")

    audit_report = {
        "initial_rows": initial_rows,
        "initial_columns": initial_cols,
        "steps": {},
    }

    # -------------------------------------------------------------
    # STEP 1: DEDUPLICATION
    # -------------------------------------------------------------
    print("\n[Step 1/7] Removing Duplicate Records...")
    # Duplicates were exact duplicate rows with identical record_id or attributes
    dup_mask = df.duplicated(subset=["record_id"], keep="first")
    num_duplicates = int(dup_mask.sum())
    df = df[~dup_mask].copy().reset_index(drop=True)
    post_dedup_rows = len(df)
    print(f"    - Duplicates identified and dropped: {num_duplicates:,}")
    print(f"    - Rows remaining: {post_dedup_rows:,}")

    audit_report["steps"]["deduplication"] = {
        "duplicates_removed": num_duplicates,
        "rows_after_dedup": post_dedup_rows,
    }

    # -------------------------------------------------------------
    # STEP 2: FIX SPELLINGS (DISTRICTS & CROPS)
    # -------------------------------------------------------------
    print("\n[Step 2/7] Standardizing Categorical Spellings...")
    # Normalize districts
    orig_districts = df["district"].copy()
    cleaned_district_series = (
        df["district"].astype(str).str.strip().str.lower().map(DISTRICT_CORRECTION_MAP)
    )
    # Fallback for any unmapped
    cleaned_district_series = cleaned_district_series.fillna(df["district"].str.title())
    dist_fixed_count = int((orig_districts != cleaned_district_series).sum())
    df["district"] = cleaned_district_series

    # Normalize crops
    orig_crops = df["crop"].copy()
    cleaned_crop_series = (
        df["crop"].astype(str).str.strip().str.lower().map(CROP_CORRECTION_MAP)
    )
    cleaned_crop_series = cleaned_crop_series.fillna(df["crop"].str.lower())
    crop_fixed_count = int((orig_crops != cleaned_crop_series).sum())
    df["crop"] = cleaned_crop_series

    print(f"    - District spelling corrections applied: {dist_fixed_count:,}")
    print(f"      Unique valid districts now: {sorted(df['district'].unique().tolist())}")
    print(f"    - Crop spelling corrections applied: {crop_fixed_count:,}")
    print(f"      Unique valid crops now: {sorted(df['crop'].unique().tolist())}")

    audit_report["steps"]["spelling_corrections"] = {
        "district_corrections": dist_fixed_count,
        "crop_corrections": crop_fixed_count,
        "valid_districts": sorted(df["district"].unique().tolist()),
        "valid_crops": sorted(df["crop"].unique().tolist()),
    }

    # -------------------------------------------------------------
    # STEP 3: DATE PARSING & HARMONIZATION
    # -------------------------------------------------------------
    print("\n[Step 3/7] Harmonizing Mixed Date Formats to ISO-8601 (YYYY-MM-DD)...")
    parsed_dates = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    nat_count = int(parsed_dates.isna().sum())
    df["date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    print(f"    - Dates successfully standardized: {len(df):,} (Failed/NaT: {nat_count})")

    audit_report["steps"]["date_harmonization"] = {
        "standardized_count": len(df),
        "nat_count": nat_count,
        "date_range": [df["date"].min(), df["date"].max()],
    }

    # -------------------------------------------------------------
    # STEP 4: MANDI PRICE UNIT STANDARDIZATION (Rs/kg vs Rs/quintal)
    # -------------------------------------------------------------
    print("\n[Step 4/7] Detecting & Normalizing Mandi Prices (Rs/kg -> Rs/quintal)...")
    # Decision rationale:
    # 1 quintal = 100 kg. Standard mandi prices in quintals range from ₹1,200 to ₹12,000+.
    # Prices entered in Rs/kg range from ₹15 to ₹120.
    # Therefore, prices where 0 < price <= 250 are mathematically Rs/kg and need multiplying by 100.
    # Prices <= 0 or > 25,000 are extreme outliers (handled in Step 7).
    price_col = "mandi_price_rs_q"
    df["raw_mandi_price"] = df[price_col].copy()

    # Detect Rs/kg condition: positive price below Rs 250 (roughly 100x below commodity median)
    detected_rs_kg_mask = (df[price_col] > 0) & (df[price_col] <= 250.0)
    detected_rs_kg_count = int(detected_rs_kg_mask.sum())

    # Standardize to Rs/quintal
    df["price_unit_converted"] = detected_rs_kg_mask.astype(int)
    df.loc[detected_rs_kg_mask, price_col] = (df.loc[detected_rs_kg_mask, price_col] * 100.0).round(1)

    print(f"    - Detected prices recorded in Rs/kg: {detected_rs_kg_count:,}")
    print(f"    - Successfully converted {detected_rs_kg_count:,} prices to Rs/quintal (multiplied by 100).")

    # Benchmarking detection against ground-truth log
    price_eval = {}
    if gt_data and "price_unit_error_indices" in gt_data.get("defects", {}):
        gt_price_indices = set(gt_data["defects"]["price_unit_error_indices"])
        detected_indices = set(df[detected_rs_kg_mask].index)

        tp = len(detected_indices & gt_price_indices)
        fp = len(detected_indices - gt_price_indices)
        fn = len(gt_price_indices - detected_indices)
        precision = round(tp / (tp + fp), 5) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 5) if (tp + fn) > 0 else 0.0
        f1 = round(2 * (precision * recall) / (precision + recall), 5) if (precision + recall) > 0 else 0.0

        print(f"    - Ground-Truth Benchmark for Price Unit Detection:")
        print(f"      True Positives: {tp:,} | False Positives: {fp:,} | False Negatives: {fn:,}")
        print(f"      Precision: {precision * 100:.2f}% | Recall: {recall * 100:.2f}% | F1-Score: {f1:.4f}")
        price_eval = {
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1_score": f1,
        }

    audit_report["steps"]["price_unit_standardization"] = {
        "rs_kg_detected_and_converted": detected_rs_kg_count,
        "evaluation": price_eval,
    }

    # -------------------------------------------------------------
    # STEP 5: YIELD UNIT STANDARDIZATION (kg/ha and tonnes/ha -> quintals/ha)
    # -------------------------------------------------------------
    print("\n[Step 5/7] Detecting & Normalizing Crop Yield Units...")
    # Decision rationale:
    # 1 quintal = 100 kg = 0.1 metric tonne.
    # Biological plausible yield ranges per crop:
    # - Tonnes/ha error: Yield < crop's ton_max threshold (e.g. 0.25 to 2.2 tonnes/ha) -> multiply by 10.
    # - kg/ha error: Yield > 100 q/ha (typically 250 to 3,800 kg/ha) -> divide by 100.
    yield_col = "yield_quintal_per_ha"
    df["raw_yield"] = df[yield_col].copy()
    df["yield_unit_fix"] = "none"

    tonnes_fixed_total = 0
    kg_fixed_total = 0

    # A) Tonnes/ha conversion (multiply by 10)
    for crop_name, thresholds in CROP_YIELD_THRESHOLDS.items():
        ton_mask = (df["crop"] == crop_name) & (df[yield_col] > 0) & (df[yield_col] < thresholds["ton_max"])
        ton_count = int(ton_mask.sum())
        if ton_count > 0:
            df.loc[ton_mask, yield_col] = (df.loc[ton_mask, yield_col] * 10.0).round(2)
            df.loc[ton_mask, "yield_unit_fix"] = "tonnes_to_quintal"
            tonnes_fixed_total += ton_count

    # B) kg/ha conversion (divide by 100)
    kg_mask = (df[yield_col] > 100.0)
    kg_fixed_total = int(kg_mask.sum())
    if kg_fixed_total > 0:
        df.loc[kg_mask, yield_col] = (df.loc[kg_mask, yield_col] / 100.0).round(2)
        df.loc[kg_mask, "yield_unit_fix"] = "kg_to_quintal"

    total_yield_fixed = tonnes_fixed_total + kg_fixed_total
    print(f"    - Yield in tonnes/ha detected & converted (x10): {tonnes_fixed_total:,}")
    print(f"    - Yield in kg/ha detected & converted (/100): {kg_fixed_total:,}")
    print(f"    - Total yield unit anomalies resolved: {total_yield_fixed:,}")

    # Benchmarking against ground truth
    yield_eval = {}
    if gt_data:
        gt_kg = set(gt_data.get("defects", {}).get("yield_kg_ha_indices", []))
        gt_ton = set(gt_data.get("defects", {}).get("yield_tonnes_ha_indices", []))

        # Precision/Recall for tonnes/ha
        detected_ton = set(df[df["yield_unit_fix"] == "tonnes_to_quintal"].index)
        tp_ton = len(detected_ton & gt_ton)
        prec_ton = tp_ton / len(detected_ton) if detected_ton else 1.0
        rec_ton = tp_ton / len(gt_ton) if gt_ton else 1.0

        # Precision/Recall for kg/ha
        detected_kg = set(df[df["yield_unit_fix"] == "kg_to_quintal"].index)
        tp_kg = len(detected_kg & gt_kg)
        prec_kg = tp_kg / len(detected_kg) if detected_kg else 1.0
        rec_kg = tp_kg / len(gt_kg) if gt_kg else 1.0

        print(f"    - Yield Ground-Truth Benchmark:")
        print(f"      Tonnes/ha: Precision {prec_ton * 100:.2f}% | Recall {rec_ton * 100:.2f}%")
        print(f"      kg/ha:     Precision {prec_kg * 100:.2f}% | Recall {rec_kg * 100:.2f}%")
        yield_eval = {
            "tonnes_precision": round(prec_ton, 4), "tonnes_recall": round(rec_ton, 4),
            "kg_precision": round(prec_kg, 4), "kg_recall": round(rec_kg, 4),
        }

    audit_report["steps"]["yield_unit_standardization"] = {
        "tonnes_to_quintal_count": tonnes_fixed_total,
        "kg_to_quintal_count": kg_fixed_total,
        "total_yield_unit_fixes": total_yield_fixed,
        "evaluation": yield_eval,
    }

    # -------------------------------------------------------------
    # STEP 6: MISSING RAINFALL IMPUTATION & COMPARISON SETUP
    # -------------------------------------------------------------
    print("\n[Step 6/7] Handling Missing Rainfall Data (Localized Imputation)...")
    # Decision rationale:
    # A simple global mean ignores seasonal weather cycles and regional microclimates.
    # 1. We preserve the raw un-imputed weekly rainfall columns (rainfall_w01_raw .. rainfall_w16_raw)
    #    so downstream experiments can easily benchmark models with and without imputation.
    # 2. We add a binary 'rain_missing' flag (1 = any missing weekly rainfall, 0 = complete record).
    # 3. We impute using localized group medians by [district, year, season] for each specific week.
    weekly_cols = [f"rainfall_w{i:02d}" for i in range(1, 17)]

    # Flag whether any weekly rainfall was missing
    df["rain_missing"] = df[weekly_cols].isna().any(axis=1).astype(int)
    df["rain_missing_weeks_count"] = df[weekly_cols].isna().sum(axis=1)

    initial_missing_cells = int(df[weekly_cols].isna().sum().sum())
    initial_missing_rows = int(df["rain_missing"].sum())
    print(f"    - Total missing rainfall cells across June-Sept: {initial_missing_cells:,}")
    print(f"    - Rows flagged with 'rain_missing=1': {initial_missing_rows:,} ({initial_missing_rows/len(df)*100:.1f}%)")

    # Preserve raw weekly columns
    for col in weekly_cols:
        df[f"{col}_raw"] = df[col].copy()

    # Localized Imputation: Group by [district, year, season] for each weekly rainfall
    for col in weekly_cols:
        local_avg = df.groupby(["district", "year", "season"])[col].transform("median")
        df[col] = df[col].fillna(local_avg)
        # Fallback to regional year-season average if any remaining
        if df[col].isna().any():
            regional_avg = df.groupby(["year", "season"])[col].transform("median")
            df[col] = df[col].fillna(regional_avg)
        df[col] = df[col].round(1)

    remaining_missing = int(df[weekly_cols].isna().sum().sum())
    print(f"    - Localized spatial-temporal imputation complete. Remaining missing cells: {remaining_missing}")

    # Derived summary features
    df["total_rainfall_imputed_mm"] = df[weekly_cols].sum(axis=1).round(1)

    audit_report["steps"]["rainfall_imputation"] = {
        "missing_cells_imputed": initial_missing_cells,
        "rows_with_rain_missing": initial_missing_rows,
        "imputation_grouping": ["district", "year", "season"],
        "preserved_raw_columns": [f"{c}_raw" for c in weekly_cols],
    }

    # -------------------------------------------------------------
    # STEP 7: OUTLIER DETECTION & NON-DESTRUCTIVE FLAGGING
    # -------------------------------------------------------------
    print("\n[Step 7/7] Detecting & Flagging Physical/Sensor Outliers...")
    # Decision rationale:
    # Dropping outliers deletes valuable information. In production, flagging allows:
    # - Training models with and without outliers.
    # - Robust loss functions (Huber / MAE) to down-weight them.
    # - Field teams to audit sensor malfunction.

    # 1. Area outliers: impossible negative values or placeholder 999 ha
    df["is_outlier_area"] = ((df["area_ha"] <= 0) | (df["area_ha"] > 100.0)).astype(int)

    # 2. Mandi price outliers: negative prices or placeholder 99999
    df["is_outlier_price"] = ((df["mandi_price_rs_q"] <= 0) | (df["mandi_price_rs_q"] > 25000.0)).astype(int)

    # 3. Soil pH outliers: physically impossible acidity/alkalinity for arable soil (<4.0 or >10.0)
    df["is_outlier_soil_ph"] = ((df["soil_ph"] < 4.0) | (df["soil_ph"] > 10.0)).astype(int)

    # 4. Yield outliers: impossible yields remaining above crop biological limit (e.g. > 80 q/ha)
    df["is_outlier_yield"] = (df[yield_col] > 80.0).astype(int)

    # Combined master flag
    df["is_outlier"] = (
        df["is_outlier_area"]
        | df["is_outlier_price"]
        | df["is_outlier_soil_ph"]
        | df["is_outlier_yield"]
    ).astype(int)

    outlier_counts = {
        "area_outliers": int(df["is_outlier_area"].sum()),
        "price_outliers": int(df["is_outlier_price"].sum()),
        "soil_ph_outliers": int(df["is_outlier_soil_ph"].sum()),
        "yield_outliers": int(df["is_outlier_yield"].sum()),
        "total_unique_outlier_rows": int(df["is_outlier"].sum()),
    }

    print(f"    - Flagged {outlier_counts['area_outliers']:,} area outliers (negative or 999 ha).")
    print(f"    - Flagged {outlier_counts['price_outliers']:,} price outliers (negative or 99999).")
    print(f"    - Flagged {outlier_counts['soil_ph_outliers']:,} soil pH outliers (<4.0 or >10.0).")
    print(f"    - Flagged {outlier_counts['yield_outliers']:,} yield outliers (>80 quintals/ha).")
    print(f"    - Total outlier rows flagged (preserved without deletion): {outlier_counts['total_unique_outlier_rows']:,}")

    audit_report["steps"]["outlier_flagging"] = outlier_counts

    # -------------------------------------------------------------
    # SAVE CLEAN DATASET & AUDIT REPORT
    # -------------------------------------------------------------
    clean_csv_path = output_dir / "agri_yield_cleaned.csv"
    audit_json_path = output_dir / "cleaning_audit_report.json"

    print(f"\n[*] Saving cleaned dataset to: {clean_csv_path}...")
    df.to_csv(clean_csv_path, index=False)

    print(f"[*] Saving cleaning audit report to: {audit_json_path}...")
    elapsed = round(time.time() - start_time, 2)
    audit_report["final_rows"] = len(df)
    audit_report["final_columns"] = len(df.columns)
    audit_report["elapsed_seconds"] = elapsed

    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print(f"[✓] Data cleaning pipeline completed in {elapsed} seconds!")
    print(f"    - File saved: {clean_csv_path} ({os.path.getsize(clean_csv_path) / (1024*1024):.2f} MB)")
    print(f"    - Rows: {initial_rows:,} -> {len(df):,} (Duplicates removed: {num_duplicates:,})")
    print(f"    - Columns: {initial_cols} -> {len(df.columns)}")

    return df, audit_report


if __name__ == "__main__":
    df_clean, audit = clean_agri_dataset()
    print("\n" + "=" * 65)
    print("BEFORE / AFTER DATA SUMMARY")
    print("=" * 65)
    print(f"Rows:        Before: {audit['initial_rows']:,}  |  After: {audit['final_rows']:,}")
    print(f"Columns:     Before: {audit['initial_columns']}   |  After: {audit['final_columns']}")
    print("-" * 65)
    print("Corrections & Fixes Applied:")
    print(f" • Duplicates Removed:                {audit['steps']['deduplication']['duplicates_removed']:,}")
    print(f" • District Spelling Typos Corrected: {audit['steps']['spelling_corrections']['district_corrections']:,}")
    print(f" • Crop Spelling Typos Corrected:     {audit['steps']['spelling_corrections']['crop_corrections']:,}")
    print(f" • Dates Harmonized:                  {audit['steps']['date_harmonization']['standardized_count']:,}")
    print(f" • Mandi Prices Converted (Rs/kg->q): {audit['steps']['price_unit_standardization']['rs_kg_detected_and_converted']:,}")
    if "evaluation" in audit["steps"]["price_unit_standardization"]:
        pe = audit["steps"]["price_unit_standardization"]["evaluation"]
        print(f"   └─ Precision: {pe.get('precision', 0)*100:.2f}% | Recall: {pe.get('recall', 0)*100:.2f}%")
    print(f" • Yield Units Converted (to q/ha):   {audit['steps']['yield_unit_standardization']['total_yield_unit_fixes']:,}")
    print(f"   ├─ From tonnes/ha (x10):           {audit['steps']['yield_unit_standardization']['tonnes_to_quintal_count']:,}")
    print(f"   └─ From kg/ha (/100):              {audit['steps']['yield_unit_standardization']['kg_to_quintal_count']:,}")
    print(f" • Rainfall Missing Cells Imputed:    {audit['steps']['rainfall_imputation']['missing_cells_imputed']:,}")
    print(f" • Total Outlier Records Flagged:     {audit['steps']['outlier_flagging']['total_unique_outlier_rows']:,}")
    print("=" * 65)
