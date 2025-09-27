# Student Learning Data Processing

This script processes student assessment Excel files (`Grade_1.xlsx` … `Grade_6.xlsx`) to calculate learning outcomes at the **overall subject level** and the **domain level**.  
It produces clean CSV outputs that can be used for analysis or visualization.

---

## What the script does

1. **Loads Excel files**  
   Looks for `Grade_1.xlsx` … `Grade_6.xlsx` in the working directory.

2. **Cleans and normalizes data**  
   - Detects where data starts (by `ScRN` and `StRN`).  
   - Standardizes region names from Arabic to English.  
   - Strips Arabic diacritics and normalizes text.  
   - Converts item responses into binary (0/1) or numeric scores.

3. **Calculates per-student scores**  
   - Overall score (% correct across all items in a sheet).  
   - Domain scores (% correct in specific skill areas like Reading, Writing, etc.).  
   - Flags whether a student is **below proficiency (<50%)**.

4. **Aggregates results**  
   - By **region × grade × subject**: average score, % below proficiency, student counts.  
   - By **region × grade × subject × domain**: same measures at the domain level.

5. **Analyzes gender gaps**  
   - Compares Male vs Female results (only if both groups ≥50 students).  
   - Uses Welch’s t-test for significance.  
   - Reports % below proficiency for each gender and the percentage-point gap.

6. **Flags “overlap” cases**  
   A region/grade/subject (or domain) is flagged if:  
   - % below proficiency is **10 points higher** than the national average, **and**  
   - There is a **significant gender gap** (p<0.05 and gap ≥5 points).

7. **Saves outputs** (CSV with UTF-8 BOM for Excel compatibility):
   - `agg_region_grade_subject.csv`  
   - `agg_gender.csv`  
   - `overlap.csv`  
   - `agg_region_grade_subject_domain.csv`  
   - `agg_gender_domain.csv`  
   - `overlap_domain.csv`

---

## Input expectations

- Excel files named `Grade_1.xlsx` … `Grade_6.xlsx`.  
- Each file has sheets such as `En - Written`, `Fr - oral`, `Ar - Oral`, etc.  
- Student info columns: `ScRN`, `StRN`, `Class Name`, `Course Name`, `Exam Type`, `Gender`, `Governate`.  
- Item score columns vary by sheet and are mapped in the script’s **DOMAIN_CONFIG**.

---

## How to run

1. Install dependencies:
   ```bash
   pip install pandas numpy scipy openpyxl


Notes

- Proficiency threshold = 50%.

- Region names are normalized (e.g., بيروت → Beirut).

- Gender is mapped from Arabic (ذكر → Male, انثى → Female).

- Console output shows diagnostic info about matched columns and any missing config.
