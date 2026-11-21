# Data Dictionary: Agri Predictive Analytics Model (`agri_yield_raw.csv`)

## 1. Overview & Dataset Context

- **Dataset File:** `data/raw/agri_yield_raw.csv`
- **Total Records:** ~350,000 rows
- **Geographic Scope:** 6 Districts across Vidarbha, Maharashtra, India:
  - Nagpur, Wardha, Amravati, Yavatmal, Chandrapur, Bhandara
  - High-cardinality village/taluka representations: 210 unique identifiers
- **Temporal Scope:** 2006 to 2024 (19 agricultural years)
- **Crops Represented:**
  - **Kharif Season (Monsoon):** Soybean, Cotton, Tur (Pigeon pea), Rice
  - **Rabi Season (Winter):** Wheat, Gram (Chickpea), Jowar (Sorghum)
- **Synthetic Disclaimer:** All records are synthetically generated based on real-world agronomic and agro-meteorological research principles of central India's Vertisol (Black Cotton Soil) agricultural belt.

---

## 2. Schema Specification

| Column Name | Data Type | Physical Unit | Description & Agronomic Role | Expected / Valid Range | Injected Defects Present |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `record_id` | String | Identifier | Unique primary key for each observation | `REC_0000001` - `REC_00350000` | Duplicates present in ~1% of dataset |
| `date` | String | Date | Harvest or recording date of the crop season | 2006-01-01 to 2024-12-31 | **Mixed Date Formats** (`YYYY-MM-DD`, `DD/MM/YYYY`, `MM-DD-YYYY`, `DD-b-YYYY`, `YYYY/MM/DD`) |
| `year` | Integer | Year | Crop year | `2006` - `2024` | None |
| `season` | String | Categorical | Agricultural season (`Kharif` or `Rabi`) | `Kharif`, `Rabi` | Clean |
| `district` | String | Categorical | Administrative district in Vidarbha | 6 valid names | **Spelling errors & variants** (e.g. `Nagpr`, `Yeotmal`, `Amrawati`, `Wardhaa`, `Chanda`) |
| `taluka_village_id` | String | Categorical | High-cardinality village/taluka locator ID | 210 unique IDs (e.g., `NGP_KATOL_001`, `YTL_PUSAD_005`) | High-cardinality distribution |
| `crop` | String | Categorical | Crop cultivated | 7 standard crops | **Spelling & vernacular variants** (e.g. `soyabean`, `kapas`, `arhar`, `paddy`, `chana`, `gehun`) |
| `area_ha` | Float | Hectares | Cultivated farm plot area | Typically `0.4` - `15.0` ha | **Outliers** (negative values like `-2.5` ha, extreme values like `999.0` ha) |
| `soil_n_kg_ha` | Float | kg/ha | Available Soil Nitrogen | `120.0` - `360.0` kg/ha | None |
| `soil_p_kg_ha` | Float | kg/ha | Available Soil Phosphorus | `6.0` - `42.0` kg/ha | None |
| `soil_k_kg_ha` | Float | kg/ha | Available Soil Potassium | `140.0` - `440.0` kg/ha | None |
| `soil_ph` | Float | pH scale | Soil acidity / alkalinity | Typically `5.8` - `8.6` | **Outliers** (physically impossible values like `1.2` or `14.0`) |
| `soil_organic_carbon_pct` | Float | Percentage (%) | Soil Organic Carbon content | `0.25%` - `1.05%` | Clean |
| `mandi_price_rs_q` | Float | Rs / quintal | Market price at local APMC / Mandi | Typically `₹1,200` - `₹12,500` / quintal | **15-20% recorded in Rs/kg** (divided by 100, e.g. `₹48.5` instead of `₹4,850`), plus negative/outliers |
| `yield_quintal_per_ha` | Float | Quintals / ha | **Target Variable**: Crop harvest yield per unit area | Typically `2.5` - `65.0` quintals/ha (1 q = 100 kg) | **Unit confusion errors** (~2.5% in `kg/ha` [x100], ~2.5% in `tonnes/ha` [/10]), and extreme outliers (>400 q/ha) |
| `rainfall_w01` to `rainfall_w16` | Float | mm | Weekly rainfall amounts during monsoon season (June W1 to Sept W4) | `0.0` - `250.0` mm per week | **Missing Values**: random cell gaps (~5.5%) + **multi-week sensor blackout blocks** (2-4 consecutive weeks) |

---

## 3. Deliberately Injected Data Defects & Ground-Truth Reference

To simulate real-world messy enterprise agricultural data pipelines, deliberate quality defects have been introduced. Ground-truth logs are stored at:
- Summary: `data/raw/ground_truth_summary.json`
- Full Details & Indices: `data/raw/ground_truth_defects.json`

### 1) Mandi Price Unit Ambiguity (Rs/kg vs. Rs/quintal)
- **Defect Description:** 15% to 20% of rows contain mandi prices recorded in **₹/kg** instead of standard **₹/quintal** within the same column.
- **Conversion Factor:** \( \text{Price}_{\text{kg}} = \frac{\text{Price}_{\text{quintal}}}{100} \)
- **Impact:** Causes a distinct bimodal distribution (e.g. ₹35-₹80 / kg mixed with ₹3,500-₹8,000 / quintal).

### 2) Missing Rainfall Data (June - September)
- **Defect Description:**
  - **Random Gaps:** Approximately 5-6% of individual weekly rainfall cells across all 16 weeks are set to `NaN`.
  - **Multi-Week Block Outages:** Approximately 6% of rows have multi-week sensor failures (2 to 4 consecutive weeks of continuous `NaN` values).
- **Impact:** Tests time-series interpolation, KNN imputation, or forward-fill strategies versus naive mean replacement.

### 3) Yield Unit Inconsistencies (kg/ha and tonnes/ha)
- **Defect Description:**
  - **kg/ha Unit Error:** ~2.5% of rows were mistakenly logged in kg/ha (\( \text{Yield} \times 100 \)), creating false spikes up to 1,500-4,000.
  - **tonnes/ha Unit Error:** ~2.5% of rows were logged in metric tonnes/ha (\( \frac{\text{Yield}}{10} \)), creating false depressions around 0.5-4.0.
- **Impact:** Severe distortion of regression target without anomaly detection and unit standardization.

### 4) Categorical Spelling & Vernacular Typos
- **Districts:** Typographical errors and historic spelling variants (~2% of records):
  - `Nagpur` ➔ `Nagpr`, `Nagpurr`, `Nagpoor`
  - `Wardha` ➔ `Wardhaa`, `Vardha`
  - `Amravati` ➔ `Amrawati`, `Amaravati`
  - `Yavatmal` ➔ `Yeotmal`, `Yavatmaal`, `Yawatmal`
  - `Chandrapur` ➔ `Chanda`, `Chandrapurr`
  - `Bhandara` ➔ `Bhandaraa`, `Bhandra`
- **Crops:** Hindi/Marathi vernacular names and typos (~2.5% of records):
  - `soybean` ➔ `soyabean`, `soya_bean`, `soyabeen`
  - `cotton` ➔ `coton`, `kapas`
  - `tur` ➔ `arhar`, `tuar`, `toor`
  - `rice` ➔ `paddy`, `dhan`
  - `wheat` ➔ `wheatt`, `gehun`
  - `gram` ➔ `chana`, `channa`
  - `jowar` ➔ `jowarr`, `sorghum`

### 5) Physical Outliers & Invalid Values
- **Yield Outliers:** ~350 rows with yields between 400.0 and 950.0 q/ha.
- **Area Outliers:** Negative values (`-2.5`, `-1.0`) and placeholder values (`999.0`).
- **Price Outliers:** Erroneous values (`99999.0`, `-100.0`, `0.0`).
- **Soil pH Outliers:** Chemically impossible readings (`1.2`, `14.0`).

### 6) Mixed Date Formats
- Multiple date formatting strings within the `date` column:
  - 60% `YYYY-MM-DD` (ISO 8601)
  - 15% `DD/MM/YYYY` (Standard Indian format)
  - 10% `MM-DD-YYYY` (US format)
  - 10% `DD-b-YYYY` (e.g. `15-Oct-2018`)
  - 5% `YYYY/MM/DD`

### 7) Exact Duplicate Rows
- ~3,500 exact duplicate records (~1% of dataset) randomly cloned and appended to the raw data file.

---

## 4. Agronomic Modeling Logic (Realistic Yield Dynamics)

The synthetic yield is generated via realistic agronomic response functions:
1. **Base Yields:** Calibrated to ICAR / Maharashtra Agriculture Department district averages.
2. **Technological Trend:** Slow upward gain of ~1.1% per year from 2006 to 2024 (seed genetics, mechanization).
3. **Severe Drought Years:** Explicit downward yield shocks for documented historical drought years in Vidarbha:
   - **2009:** Monsoon deficit (-26% yield penalty)
   - **2014:** Severe dry spell (-28% yield penalty)
   - **2015:** Acute Maharashtra drought (-38% yield penalty)
   - **2018:** Late monsoon withdrawal & deficit (-24% yield penalty)
4. **Rainfall Curvature:** Optimal rainfall response curves per crop; yields suffer if total rain falls below optimal, or if excessive rain causes waterlogging and fungal disease.
5. **Dry Spell Penalties:** Continuous dry spells exceeding 3 weeks impose an exponential yield penalty.
6. **Soil Carbon Boost:** Soil Organic Carbon acts as a multiplier; higher organic matter retains moisture and enhances nutrient uptake.
