"""
Synthetic Data Generator for Agricultural Predictive Analytics Model (agri-yield).

Generates ~350,000 rows across 6 districts of Vidarbha, Maharashtra (2006-2024),
modeling realistic agronomic relationships between rainfall, dry spells, soil chemistry,
crop varieties, drought years, and crop yield.

Deliberately injects controlled data quality defects (price units, missing rainfall blocks,
yield unit errors, spelling errors, mixed date formats, outliers, and duplicates)
and records a complete ground-truth log for validation.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, RAW_DATA_DIR
from src.utils import set_seed


def generate_agri_dataset(
    target_rows: int = 350000,
    seed: int = RANDOM_SEED,
    output_dir: Path = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, dict]:
    """
    Generates synthetic agricultural dataset and deliberately injects defects.

    Parameters
    ----------
    target_rows : int
        Approximate number of rows to generate (default: 350,000).
    seed : int
        Random seed for full reproducibility (default: 42).
    output_dir : Path
        Directory to save raw dataset and ground-truth logs.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Generated DataFrame and complete ground-truth defect log.
    """
    start_time = time.time()
    set_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"[*] Starting dataset generation: {target_rows:,} target rows...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Geographic & High-Cardinality Setup (6 Districts, 210 Taluka/Village IDs)
    districts_config = {
        "Nagpur": [f"NGP_{t}_{i:03d}" for t in ["KATOL", "SAONER", "UMRED", "NARKHED", "HINGNA"] for i in range(1, 8)],
        "Wardha": [f"WRD_{t}_{i:03d}" for t in ["ARVI", "DEOLI", "HINGANGHAT", "SAMUDRAPUR", "SELOO"] for i in range(1, 8)],
        "Amravati": [f"AMR_{t}_{i:03d}" for t in ["ACHALPUR", "MORSHI", "WARUD", "CHANDUR", "DARYAPUR"] for i in range(1, 8)],
        "Yavatmal": [f"YTL_{t}_{i:03d}" for t in ["PUSAD", "UMARKHED", "DIGRAS", "WANI", "DARWHA"] for i in range(1, 8)],
        "Chandrapur": [f"CHD_{t}_{i:03d}" for t in ["WARORA", "BHADRAVATI", "RAJURA", "BRAMHAPURI", "NAGBHID"] for i in range(1, 8)],
        "Bhandara": [f"BHD_{t}_{i:03d}" for t in ["TUMSAR", "MOHADI", "SAKOLI", "LAKHANI", "PAUNI"] for i in range(1, 8)],
    }
    districts = list(districts_config.keys())
    all_villages = [v for v_list in districts_config.values() for v in v_list]
    print(f"    - Setup 6 districts with {len(all_villages)} unique village/taluka IDs (high cardinality).")

    # 2. Crops & Seasons Setup
    kharif_crops = ["soybean", "cotton", "tur", "rice"]
    rabi_crops = ["wheat", "gram", "jowar"]
    years = list(range(2006, 2025))  # 2006 to 2024 (19 years)
    drought_years = {2009, 2014, 2015, 2018}

    # Base baseline yields (quintal/hectare) & Mandi price baselines (Rs/quintal)
    crop_specs = {
        "soybean": {"season": "Kharif", "base_yield": 14.5, "base_price": 3200, "price_trend": 130, "optimal_rain": 880},
        "cotton":  {"season": "Kharif", "base_yield": 15.5, "base_price": 4200, "price_trend": 180, "optimal_rain": 920},
        "tur":     {"season": "Kharif", "base_yield": 9.5,  "base_price": 3900, "price_trend": 220, "optimal_rain": 820},
        "rice":    {"season": "Kharif", "base_yield": 28.0, "base_price": 1850, "price_trend": 95,  "optimal_rain": 1150},
        "wheat":   {"season": "Rabi",   "base_yield": 25.5, "base_price": 1650, "price_trend": 85,  "optimal_rain": 700},
        "gram":    {"season": "Rabi",   "base_yield": 11.5, "base_price": 2850, "price_trend": 140, "optimal_rain": 680},
        "jowar":   {"season": "Rabi",   "base_yield": 14.0, "base_price": 1550, "price_trend": 80,  "optimal_rain": 650},
    }

    # Calculate rows per year/district/crop
    # 350,000 target rows: ~346,500 base rows + ~3,500 duplicates = 350,000 total rows
    n_duplicates = 3500
    n_base = target_rows - n_duplicates

    # Sample attributes with realistic distribution weights
    district_probs = [0.20, 0.16, 0.18, 0.20, 0.14, 0.12]
    sampled_districts = rng.choice(districts, size=n_base, p=district_probs)
    sampled_years = rng.choice(years, size=n_base)

    # Village assignment conditional on district
    sampled_villages = np.empty(n_base, dtype=object)
    for dist in districts:
        mask = (sampled_districts == dist)
        v_list = districts_config[dist]
        sampled_villages[mask] = rng.choice(v_list, size=np.sum(mask))

    # Sample crops conditional on district (e.g., Bhandara grows more rice; Yavatmal grows more cotton/soybean)
    sampled_crops = np.empty(n_base, dtype=object)
    for dist in districts:
        mask = (sampled_districts == dist)
        count = np.sum(mask)
        if dist == "Bhandara":
            c_probs = [0.15, 0.05, 0.10, 0.45, 0.10, 0.10, 0.05]
        elif dist in ["Yavatmal", "Wardha"]:
            c_probs = [0.32, 0.35, 0.12, 0.01, 0.08, 0.08, 0.04]
        elif dist == "Amravati":
            c_probs = [0.35, 0.25, 0.18, 0.02, 0.08, 0.08, 0.04]
        elif dist == "Chandrapur":
            c_probs = [0.22, 0.22, 0.12, 0.20, 0.10, 0.10, 0.04]
        else:  # Nagpur
            c_probs = [0.28, 0.20, 0.14, 0.10, 0.12, 0.12, 0.04]
        all_crops_list = list(crop_specs.keys())
        sampled_crops[mask] = rng.choice(all_crops_list, size=count, p=c_probs)

    sampled_seasons = np.array([crop_specs[c]["season"] for c in sampled_crops], dtype=object)

    # 3. Farm Area (hectares): Log-normal distribution typical of Indian smallholder/medium farms (0.4 to 12 ha)
    area = np.clip(np.exp(rng.normal(0.65, 0.55, size=n_base)), 0.4, 15.0).round(2)

    # 4. Generate Weekly Rainfall for June-September (Weeks 1 to 16)
    # Drought years have lower total rainfall and longer dry spells
    is_drought = np.isin(sampled_years, list(drought_years))
    rain_weeks = []
    weekly_cols = [f"rainfall_w{i:02d}" for i in range(1, 17)]

    # Matrix of 16 weeks of rain
    rainfall_matrix = np.zeros((n_base, 16), dtype=np.float64)
    for w in range(16):
        # Monsoon peak is weeks 5-11 (July/August)
        peak_factor = np.sin((w + 1) / 16.0 * np.pi)
        mean_rain = np.where(is_drought, 32.0 * peak_factor + 12.0, 58.0 * peak_factor + 22.0)
        # Gamma distribution for weekly rainfall with intermittent dry weeks
        shape = 1.8
        scale = mean_rain / shape
        rain_val = rng.gamma(shape=shape, scale=scale, size=n_base)
        # Prob of zero/near-zero rain higher in drought years
        zero_prob = np.where(is_drought, 0.25, 0.10)
        rain_val = np.where(rng.random(size=n_base) < zero_prob, 0.0, rain_val)
        rainfall_matrix[:, w] = np.round(rain_val, 1)

    total_rainfall = np.sum(rainfall_matrix, axis=1)

    # Calculate maximum consecutive dry spell (weeks with rain < 10 mm)
    dry_weeks_bool = (rainfall_matrix < 10.0)
    max_dry_spells = np.zeros(n_base, dtype=int)
    for i in range(n_base):
        curr_streak = 0
        max_streak = 0
        for is_dry in dry_weeks_bool[i]:
            if is_dry:
                curr_streak += 1
                if curr_streak > max_streak:
                    max_streak = curr_streak
            else:
                curr_streak = 0
        max_dry_spells[i] = max_streak

    # 5. Soil Properties (N, P, K, pH, Organic Carbon)
    # Typical Vidarbha Vertisols / Black soils
    soil_n = np.clip(rng.normal(230, 35, size=n_base), 120, 360).round(1)
    soil_p = np.clip(rng.normal(18.5, 5.0, size=n_base), 6.0, 42.0).round(1)
    soil_k = np.clip(rng.normal(260, 45, size=n_base), 140, 440).round(1)
    soil_ph = np.clip(rng.normal(7.35, 0.42, size=n_base), 5.8, 8.6).round(2)
    soil_oc = np.clip(rng.normal(0.56, 0.12, size=n_base), 0.25, 1.05).round(2)

    # 6. Realistic Yield Generation Formula
    # Agronomic mechanics:
    # - Base crop yield
    # - Technology trend (+1.1% per year from 2006)
    # - Drought year severe drop (2009, 2014, 2015, 2018)
    # - Rainfall non-linear response: optimal plateau, waterlogging penalty if excessive, drought penalty if low
    # - Dry spell penalty: >= 3 weeks drops yield
    # - Soil organic carbon boost: +12% per 0.2% increase in OC
    # - Balanced pH and NPK boost
    print("    - Computing agronomic yield equations...")
    base_yield_arr = np.array([crop_specs[c]["base_yield"] for c in sampled_crops])
    optimal_rain_arr = np.array([crop_specs[c]["optimal_rain"] for c in sampled_crops])

    # Upward technology / seed genetics trend
    trend_factor = 1.0 + 0.011 * (sampled_years - 2006)

    # Drought year shock
    drought_penalty = np.ones(n_base)
    drought_penalty[sampled_years == 2009] = 0.74
    drought_penalty[sampled_years == 2014] = 0.72
    drought_penalty[sampled_years == 2015] = 0.62  # Historic Maharashtra drought
    drought_penalty[sampled_years == 2018] = 0.76

    # Rainfall curvature
    rain_ratio = total_rainfall / optimal_rain_arr
    # Quadratic response: peaks near 1.0, falls if deficit or excess (waterlogging)
    rain_response = np.where(
        rain_ratio < 1.0,
        np.clip(rain_ratio ** 0.85, 0.45, 1.0),
        np.clip(1.0 - 0.35 * np.maximum(0.0, rain_ratio - 1.0) ** 1.5, 0.55, 1.0),
    )

    # Dry spell penalty
    dry_spell_penalty = np.where(
        max_dry_spells >= 3,
        np.clip(1.0 - 0.055 * (max_dry_spells - 2), 0.55, 1.0),
        1.0,
    )

    # Soil organic carbon effect
    oc_boost = np.clip(1.0 + 0.22 * (soil_oc - 0.55), 0.80, 1.25)

    # Soil chemical index (pH balance + NPK)
    ph_penalty = 1.0 - 0.06 * np.abs(soil_ph - 7.3)
    npk_boost = 1.0 + 0.02 * ((soil_n - 230) / 100.0 + (soil_p - 18.5) / 20.0 + (soil_k - 260) / 150.0)

    # Residual noise
    residual_noise = rng.normal(1.0, 0.08, size=n_base)

    clean_yield = (
        base_yield_arr
        * trend_factor
        * drought_penalty
        * rain_response
        * dry_spell_penalty
        * oc_boost
        * ph_penalty
        * npk_boost
        * residual_noise
    )
    clean_yield = np.clip(clean_yield, 2.5, 65.0).round(2)

    # 7. Mandi Prices (Rs/quintal)
    base_price_arr = np.array([crop_specs[c]["base_price"] for c in sampled_crops])
    price_trend_arr = np.array([crop_specs[c]["price_trend"] for c in sampled_crops])
    # Supply shock during drought years (+12-25% price surge)
    price_drought_shock = np.where(is_drought, rng.uniform(1.10, 1.25, size=n_base), 1.0)
    price_noise = rng.normal(0, 140, size=n_base)
    clean_mandi_price = (
        (base_price_arr + price_trend_arr * (sampled_years - 2006)) * price_drought_shock + price_noise
    )
    clean_mandi_price = np.clip(clean_mandi_price, 1200, 12500).round(1)

    # 8. Date generation with realistic seasonality
    base_dates = []
    for yr, ssn in zip(sampled_years, sampled_seasons):
        if ssn == "Kharif":
            # Harvest between Oct 1 and Dec 15
            day_offset = rng.integers(0, 75)
            dt = datetime(yr, 10, 1) + timedelta(days=int(day_offset))
        else:
            # Rabi harvest between Feb 15 and Apr 20 of year+1 (or year)
            day_offset = rng.integers(0, 65)
            dt = datetime(yr, 2, 15) + timedelta(days=int(day_offset))
        base_dates.append(dt)

    # Build primary clean DataFrame
    df = pd.DataFrame({
        "record_id": [f"REC_{i:07d}" for i in range(1, n_base + 1)],
        "date": [d.strftime("%Y-%m-%d") for d in base_dates],
        "year": sampled_years,
        "season": sampled_seasons,
        "district": sampled_districts,
        "taluka_village_id": sampled_villages,
        "crop": sampled_crops,
        "area_ha": area,
        "soil_n_kg_ha": soil_n,
        "soil_p_kg_ha": soil_p,
        "soil_k_kg_ha": soil_k,
        "soil_ph": soil_ph,
        "soil_organic_carbon_pct": soil_oc,
        "mandi_price_rs_q": clean_mandi_price.copy(),
        "yield_quintal_per_ha": clean_yield.copy(),
    })

    # Add weekly rainfall columns
    for idx, col in enumerate(weekly_cols):
        df[col] = rainfall_matrix[:, idx]

    print(f"    - Generated baseline clean matrix ({len(df):,} rows).")

    # -------------------------------------------------------------
    # 9. DELIBERATE DEFECT INJECTION & GROUND TRUTH LOGGING
    # -------------------------------------------------------------
    print("[*] Injecting deliberate anomalies and building ground-truth audit log...")
    ground_truth = {
        "metadata": {
            "target_rows": target_rows,
            "random_seed": seed,
            "created_at": datetime.now().isoformat(),
            "districts": districts,
            "crops": list(crop_specs.keys()),
            "years_covered": [int(min(years)), int(max(years))],
            "drought_years": sorted(list(drought_years)),
        },
        "defect_summary": {},
        "defects": {},
    }

    # DEFECT 1: 15-20% of Mandi Prices in Rs/kg instead of Rs/quintal
    # 1 quintal = 100 kg -> price_in_kg = price / 100
    price_defect_pct = 0.175  # 17.5%
    n_price_defects = int(n_base * price_defect_pct)
    price_defect_idx = rng.choice(n_base, size=n_price_defects, replace=False)
    orig_prices = df.loc[price_defect_idx, "mandi_price_rs_q"].values
    injected_prices = np.round(orig_prices / 100.0, 2)
    df.loc[price_defect_idx, "mandi_price_rs_q"] = injected_prices

    ground_truth["defect_summary"]["mandi_price_in_rs_per_kg"] = {
        "count": int(n_price_defects),
        "percentage": round(price_defect_pct * 100, 2),
        "conversion": "divided_by_100_to_represent_Rs_per_kg",
    }
    ground_truth["defects"]["price_unit_error_indices"] = price_defect_idx.tolist()

    # DEFECT 2: Missing rainfall in June-September (Random gaps + Multi-week blocks)
    # A) Random Gaps: ~6% of cells set to NaN across rainfall columns
    rain_cells_total = n_base * 16
    n_random_rain_nan = int(rain_cells_total * 0.055)
    random_row_coords = rng.choice(n_base, size=n_random_rain_nan)
    random_col_coords = rng.choice(16, size=n_random_rain_nan)
    for r_idx, c_idx in zip(random_row_coords, random_col_coords):
        df.iat[r_idx, df.columns.get_loc(weekly_cols[c_idx])] = np.nan

    # B) Multi-week sensor failure blocks (2 to 4 consecutive weeks) for ~6% of rows
    n_multi_week_rows = int(n_base * 0.06)
    multi_week_row_indices = rng.choice(n_base, size=n_multi_week_rows, replace=False)
    multi_week_records = []
    for r_idx in multi_week_row_indices:
        start_w = rng.integers(0, 13)
        block_len = rng.integers(2, 5)  # 2, 3, or 4 weeks
        affected_cols = weekly_cols[start_w : start_w + block_len]
        for col_name in affected_cols:
            df.iat[r_idx, df.columns.get_loc(col_name)] = np.nan
        multi_week_records.append({
            "row_idx": int(r_idx),
            "start_week": int(start_w + 1),
            "block_length": int(block_len),
            "columns": affected_cols,
        })

    total_missing_rain_cells = int(df[weekly_cols].isna().sum().sum())
    ground_truth["defect_summary"]["missing_rainfall"] = {
        "total_missing_cells": total_missing_rain_cells,
        "total_rainfall_cells": rain_cells_total,
        "missing_percentage": round(total_missing_rain_cells / rain_cells_total * 100, 2),
        "multi_week_block_rows_count": n_multi_week_rows,
    }
    ground_truth["defects"]["multi_week_block_rows"] = multi_week_row_indices.tolist()

    # DEFECT 3: Yield values in wrong units (kg/ha and tonnes/ha)
    # 2.5% in kg/ha (multiplied by 100), 2.5% in tonnes/ha (divided by 10)
    avail_yield_idx = np.setdiff1d(np.arange(n_base), price_defect_idx[:5000])  # keep distinct
    n_kg_ha = int(n_base * 0.025)
    n_tonnes_ha = int(n_base * 0.025)

    idx_kg_ha = rng.choice(avail_yield_idx, size=n_kg_ha, replace=False)
    remaining_yield_idx = np.setdiff1d(avail_yield_idx, idx_kg_ha)
    idx_tonnes_ha = rng.choice(remaining_yield_idx, size=n_tonnes_ha, replace=False)

    # In kg/ha: 1 quintal = 100 kg -> yield * 100
    orig_yield_kg = df.loc[idx_kg_ha, "yield_quintal_per_ha"].values
    df.loc[idx_kg_ha, "yield_quintal_per_ha"] = np.round(orig_yield_kg * 100.0, 1)

    # In tonnes/ha: 1 quintal = 0.1 tonne -> yield / 10
    orig_yield_tonnes = df.loc[idx_tonnes_ha, "yield_quintal_per_ha"].values
    df.loc[idx_tonnes_ha, "yield_quintal_per_ha"] = np.round(orig_yield_tonnes / 10.0, 3)

    ground_truth["defect_summary"]["yield_unit_errors"] = {
        "kg_per_ha_count": n_kg_ha,
        "tonnes_per_ha_count": n_tonnes_ha,
        "total_yield_unit_defects": n_kg_ha + n_tonnes_ha,
        "percentage": round((n_kg_ha + n_tonnes_ha) / n_base * 100, 2),
    }
    ground_truth["defects"]["yield_kg_ha_indices"] = idx_kg_ha.tolist()
    ground_truth["defects"]["yield_tonnes_ha_indices"] = idx_tonnes_ha.tolist()

    # DEFECT 4: Spelling mistakes in district and crop names (~3% of rows)
    district_typos = {
        "Nagpur": ["Nagpr", "Nagpurr", "Nagpoor"],
        "Wardha": ["Wardhaa", "Vardha"],
        "Amravati": ["Amrawati", "Amaravati"],
        "Yavatmal": ["Yeotmal", "Yavatmaal", "Yawatmal"],
        "Chandrapur": ["Chanda", "Chandrapurr"],
        "Bhandara": ["Bhandaraa", "Bhandra"],
    }
    crop_typos = {
        "soybean": ["soyabean", "soya_bean", "soyabeen"],
        "cotton": ["coton", "kapas"],
        "tur": ["arhar", "tuar", "toor"],
        "rice": ["paddy", "dhan"],
        "wheat": ["wheatt", "gehun"],
        "gram": ["chana", "channa"],
        "jowar": ["jowarr", "sorghum"],
    }

    n_dist_typos = int(n_base * 0.02)
    dist_typo_idx = rng.choice(n_base, size=n_dist_typos, replace=False)
    for idx in dist_typo_idx:
        cur_d = df.iat[idx, df.columns.get_loc("district")]
        df.iat[idx, df.columns.get_loc("district")] = rng.choice(district_typos[cur_d])

    n_crop_typos = int(n_base * 0.025)
    crop_typo_idx = rng.choice(n_base, size=n_crop_typos, replace=False)
    for idx in crop_typo_idx:
        cur_c = df.iat[idx, df.columns.get_loc("crop")]
        df.iat[idx, df.columns.get_loc("crop")] = rng.choice(crop_typos[cur_c])

    ground_truth["defect_summary"]["spelling_mistakes"] = {
        "district_typo_count": n_dist_typos,
        "crop_typo_count": n_crop_typos,
        "total_typos": n_dist_typos + n_crop_typos,
    }
    ground_truth["defects"]["district_typo_indices"] = dist_typo_idx.tolist()
    ground_truth["defects"]["crop_typo_indices"] = crop_typo_idx.tolist()

    # DEFECT 5: Outliers (impossible yields, negative area/prices, extreme soil pH)
    outlier_records = []
    # Extreme yield outliers (e.g., 450 to 900 q/ha)
    n_yield_outliers = 350
    yield_outlier_idx = rng.choice(n_base, size=n_yield_outliers, replace=False)
    df.loc[yield_outlier_idx, "yield_quintal_per_ha"] = rng.uniform(400.0, 950.0, size=n_yield_outliers).round(1)
    for idx in yield_outlier_idx:
        outlier_records.append({"row_idx": int(idx), "column": "yield_quintal_per_ha", "type": "extreme_high_yield"})

    # Area outliers (negative area or 999.0 ha)
    n_area_outliers = 200
    area_outlier_idx = rng.choice(n_base, size=n_area_outliers, replace=False)
    area_vals = rng.choice([-2.5, -1.0, 0.0, 999.0, 888.8], size=n_area_outliers)
    df.loc[area_outlier_idx, "area_ha"] = area_vals
    for idx in area_outlier_idx:
        outlier_records.append({"row_idx": int(idx), "column": "area_ha", "type": "impossible_area"})

    # Mandi price outliers (extreme price like 99999 or -500)
    n_price_outliers = 150
    price_outlier_idx = rng.choice(n_base, size=n_price_outliers, replace=False)
    df.loc[price_outlier_idx, "mandi_price_rs_q"] = rng.choice([99999.0, 88888.0, -100.0, 0.0], size=n_price_outliers)
    for idx in price_outlier_idx:
        outlier_records.append({"row_idx": int(idx), "column": "mandi_price_rs_q", "type": "impossible_price"})

    # Soil pH outliers (pH = 1.2 or 14.0)
    n_ph_outliers = 150
    ph_outlier_idx = rng.choice(n_base, size=n_ph_outliers, replace=False)
    df.loc[ph_outlier_idx, "soil_ph"] = rng.choice([1.2, 1.8, 13.5, 14.0], size=n_ph_outliers)
    for idx in ph_outlier_idx:
        outlier_records.append({"row_idx": int(idx), "column": "soil_ph", "type": "impossible_soil_ph"})

    ground_truth["defect_summary"]["outliers"] = {
        "yield_outliers": n_yield_outliers,
        "area_outliers": n_area_outliers,
        "price_outliers": n_price_outliers,
        "soil_ph_outliers": n_ph_outliers,
        "total_outliers": len(outlier_records),
    }
    ground_truth["defects"]["outlier_samples"] = outlier_records[:100]  # sample in summary

    # DEFECT 6: Mixed Date Formats
    # 60% YYYY-MM-DD (already set)
    # 15% DD/MM/YYYY
    # 10% MM-DD-YYYY
    # 10% DD-b-YYYY (e.g. 15-Oct-2019)
    # 5% YYYY/MM/DD
    fmt_indices = np.arange(n_base)
    rng.shuffle(fmt_indices)
    s1 = int(n_base * 0.60)
    s2 = s1 + int(n_base * 0.15)
    s3 = s2 + int(n_base * 0.10)
    s4 = s3 + int(n_base * 0.10)
    # indices s4: end get 5%

    date_objs = base_dates
    for idx in fmt_indices[s1:s2]:
        df.iat[idx, df.columns.get_loc("date")] = date_objs[idx].strftime("%d/%m/%Y")
    for idx in fmt_indices[s2:s3]:
        df.iat[idx, df.columns.get_loc("date")] = date_objs[idx].strftime("%m-%d-%Y")
    for idx in fmt_indices[s3:s4]:
        df.iat[idx, df.columns.get_loc("date")] = date_objs[idx].strftime("%d-%b-%Y")
    for idx in fmt_indices[s4:]:
        df.iat[idx, df.columns.get_loc("date")] = date_objs[idx].strftime("%Y/%m/%d")

    ground_truth["defect_summary"]["mixed_date_formats"] = {
        "YYYY-MM-DD": s1,
        "DD/MM/YYYY": s2 - s1,
        "MM-DD-YYYY": s3 - s2,
        "DD-b-YYYY": s4 - s3,
        "YYYY/MM/DD": n_base - s4,
    }

    # DEFECT 7: Exact Duplicate Rows (~3,500 rows)
    dup_sample_indices = rng.choice(n_base, size=n_duplicates, replace=False)
    duplicate_rows = df.iloc[dup_sample_indices].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    ground_truth["defect_summary"]["duplicates"] = {
        "duplicate_rows_added": n_duplicates,
        "total_rows_final": len(df),
        "percentage": round(n_duplicates / len(df) * 100, 2),
    }
    ground_truth["defects"]["duplicate_source_indices"] = dup_sample_indices.tolist()
    ground_truth["defects"]["duplicate_target_indices"] = list(range(n_base, n_base + n_duplicates))

    # -------------------------------------------------------------
    # 10. SAVE ARTIFACTS
    # -------------------------------------------------------------
    csv_path = output_dir / "agri_yield_raw.csv"
    gt_path = output_dir / "ground_truth_defects.json"
    summary_path = output_dir / "ground_truth_summary.json"

    print(f"[*] Writing dataset to CSV: {csv_path}...")
    df.to_csv(csv_path, index=False)

    print(f"[*] Saving ground-truth logs...")
    # Summary JSON (lightweight, human readable)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": ground_truth["metadata"],
            "defect_summary": ground_truth["defect_summary"],
        }, f, indent=2)

    # Detailed JSON with indices for evaluation
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f)

    elapsed = round(time.time() - start_time, 2)
    print(f"[✓] Generation and defect injection completed in {elapsed} seconds!")
    print(f"    - File saved: {csv_path} ({os.path.getsize(csv_path) / (1024*1024):.2f} MB)")
    print(f"    - Total rows: {len(df):,}")
    print(f"    - Total columns: {len(df.columns)}")

    return df, ground_truth


if __name__ == "__main__":
    df, gt = generate_agri_dataset()
    print("\n" + "=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nDistricts Distribution:")
    print(df["district"].value_counts().head(10))
    print("\nCrops Distribution:")
    print(df["crop"].value_counts().head(12))
    print("\nInjected Defects Overview:")
    for defect_name, details in gt["defect_summary"].items():
        print(f" - {defect_name}: {details}")
    print("=" * 60)
