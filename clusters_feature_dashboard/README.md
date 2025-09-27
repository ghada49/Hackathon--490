#  Region × Grade Hybrid Clustering Pipeline

This repository contains a **data processing pipeline** that classifies each *region × grade* into one of three performance categories:

- **High performers**  
- **Medium performers**  
- **Low performers**  

The output (`region_grade_clusters_hybrid.csv`) is standardized and ready for analysis or visualization in dashboards (e.g., Streamlit).

---

# Purpose
Educational performance varies significantly across **regions** and **grades**.  
This project provides a reproducible way to:

- Engineer standardized features from raw performance data  
- Apply **unsupervised clustering (Gaussian Mixture Models)**  
- Ensure stability with a **percentile fallback** when clusters are unbalanced  
- Export a clean CSV with interpretable labels (High / Medium / Low)

---

# Input Files

# Required
- **`agg_region_grade_subject_domain.csv`**  
  Aggregated by *region × grade × subject*.  
  Must include:  
  - `region`  
  - `subject`  
  - `avg_score`  
  - `pct_below`  
  Optional: `grade`, `n_students`

# Optional
- **`agg_gender_domain.csv`**  
  Adds gender-based gap features (female vs. male).  
  Script runs fine without it.

---

## ⚙️ Methodology

### 1. Feature Engineering
- **Z-scores per grade & subject**  
  - `z_avg` (average scores)  
  - `z_ok` (success rates)  
- **Composite subject strength**  
z_subject_strength = 0.5 × z_avg + 0.5 × z_ok
- **Size feature** (log-transformed student counts → `z_size`)  
- **Subject-specific strengths** (Arabic, English, French)  
- **Percentiles** (global + within-grade performance ranks)

### 2. Clustering
- **Primary**: Gaussian Mixture Model (3 clusters)  
- **Fallback**: Percentile binning (≤ 33% → Low, ≥ 67% → High, else Medium)

### 3. Output
- Cluster labels mapped consistently (`High → Low → Medium`)  
- Sorted by grade → cluster → region

---

## 📑 Output File

The pipeline produces **`region_grade_clusters_hybrid.csv`** with:

| Column | Description |
|--------|-------------|
| `region_canonical` | Standardized region name |
| `grade` | Grade level |
| `cluster_label` | High / Medium / Low |
| `performance_percentile_global` | Rank across all grades |
| `performance_percentile_within_grade` | Rank within same grade |
| `composite_overall` | Composite performance score |
| `mean_z_avg` | Mean z-score of avg scores |
| `mean_z_ok` | Mean z-score of success rates |
| `mean_z_size` | Mean z-score of student size |
| `subjects_covered` | Number of subjects included |
| `entries` | Number of subject rows aggregated |
| `strength_Arabic` | Subject strength in Arabic |
| `strength_English` | Subject strength in English |
| `strength_French` | Subject strength in French |


