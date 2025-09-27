# Student Risk Modeling (Grades 1–6)

This script trains **per-sheet risk models** that estimate each student’s probability of being **below proficiency (<50%)**.  
It reads grade workbooks (e.g., `Grade_1.xlsx` … `Grade_6.xlsx`), builds features from item/domain columns, fits simple models, and writes CSV/JSON outputs for analysis.

---

## What this script does (in plain terms)

1. **Reads Excel sheets** from `Grade_1.xlsx` … `Grade_6.xlsx` (only the files that exist).
2. **Builds features** from the sheet’s item columns using a built-in `DOMAIN_CONFIG`:
   - Binary items (0/1) are used as-is.
   - Count/points items are converted to **percent of max**.
   - Region and gender are added as categorical features (one-hot).
3. **Sets the target**: whether a student is **below 50% proficiency** (based on available items in that sheet).
4. **Trains a simple model** per cohort (file + sheet):
   - **Logistic Regression** with class weighting and **5-fold stratified CV**.
   - Tries to **calibrate** probabilities (Platt sigmoid). If calibration fails or data are insufficient, it falls back to a **transparent heuristic** (1 − mean of item scores).
5. **Writes results** to `data_proc/` (UTF-8 with BOM so Arabic/French work in Excel).

---

## Inputs

- Excel files: `Grade_1.xlsx` … `Grade_6.xlsx`
- The script will process all sheets it finds.  
- Column names for domains/items are defined in the internal `DOMAIN_CONFIG`.  
- Expected ID/demo columns (if present):  
  `ScRN` (school id), `StRN` (student id), `Governate` (region), `Gender`.

> **Note:** Region names in Arabic are normalized to English (e.g., `بيروت → Beirut`).

---

## Outputs (in `data_proc/`)

- **`student_risk_scores.csv`** — one row per student per sheet:
  - `student_id, school_id, region, grade, subject, modality, sheet, file`
  - `risk_prob_below` (0–1), `is_below` (0/1)

- **`risk_by_region_grade_subject.csv`** — cohort-level summary:
  - `region, grade, subject, avg_risk (0–100%), pct_students_above80 (0–100%), n_students`

- **`risk_model_metrics.json`** — modeling quality per cohort (list of dicts):
  - `n, pos_rate` (share below 50%), cross-val means of `auc`, `avg_precision`, `brier`, and per-fold values

- **`risk_model_coefficients.csv`** — average logistic coefficients:
  - `file, sheet, grade, subject, modality, feature, coef`
  - Sorted so the most positive coefficients appear first within each cohort

---

## How the model works (brief)

- **Features**  
  - From `DOMAIN_CONFIG`:
    - `domains` (binary items) → used directly.
    - `domains_counts` (points) → converted to **percentage of max** per column.
  - **Auto-detect**: if some item-like columns aren’t in the config but look binary, they may be included automatically.
  - **Categoricals**: region and gender (one-hot encoded).

- **Target (y)**  
  - For each sheet, compute each student’s average across item-like columns → `%`.
  - `is_below = 1` if `% < 50`, else `0`.

- **Training**  
  - Logistic Regression, `class_weight="balanced"`, 5-fold **StratifiedKFold**.
  - If possible, wrap with a **CalibratedClassifierCV (sigmoid)** for better probabilities.
  - If a cohort is too small (<300 rows) **or** has all-0/all-1 target, skip modeling and use a **heuristic risk**:
    - `risk_prob_below = 1 − mean(item features)` (clipped to [0,1]).

- **Metrics**  
  - AUC, Average Precision, Brier score, per-fold and averages.

---

## Requirements

Add this to `requirements.txt` (or install directly):
pandas
numpy
scipy
scikit-learn
openpyxl

How to run

- Place the script and any of Grade_1.xlsx … Grade_6.xlsx in the same folder.

- Run: python train_risk_models_all.py
- Check the data_proc/ folder for outputs.
