  Project Title: Equity & Learning Gaps Dashboard

👥 Team Members:
Roaa Hajj Chehade
Ghada Al Danab
Line Faour
# Equity & Learning Gaps Dashboard

This project was built during the **EECE490/690 Hackathon – "From Data to Impact"**.  
Our mission was to transform raw **National Student Assessment data (Grades 1–6)** into actionable insights that highlight **learning gaps, equity issues, and at-risk students**.

---

## 🎯 Problem Statement
Lebanon’s National Student Assessment dataset contains thousands of records across grades, subjects (Arabic, English, French, Math), and regions.  
However, raw Excel sheets are fragmented and difficult to analyze.  
Our project aims to answer key questions:

- Which regions and subjects show the highest learning gaps?
- Where do equity concerns (e.g., gender differences) appear?
- Which students are most at risk of falling below proficiency?
- How can policymakers triage and prioritize interventions?

---
##DataSet Used:
[https://meheadmin-my.sharepoint.com/personal/meheminister_mehe_gov_lb/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmeheminister%5Fmehe%5Fgov%5Flb%2FDocuments%2FNational%20Assessment%20Results&ga=1](https://bit.ly/45rjFRD)

## 🛠️ Approach & Architecture

The project has four main parts:

1. **Data Processing**  
   - Cleaned and normalized raw Excel files (Grades 1–6).  
   - Handled noisy column names in Arabic, English, and French.  
   - Produced **tidy CSVs** with:
     - Region × Grade × Subject aggregates  
     - Gender gap analysis  
     - Domain-level breakdowns  

2. **Risk Modeling**  
   - Trained **logistic regression models** to estimate student risk of scoring below proficiency.  
   - Calibrated predictions with cross-validation.  
   - Exported:
     - `student_risk_scores.csv` → per-student probabilities  
     - `risk_model_metrics.json` → model metrics (AUC, Brier score, etc.)  
     - `risk_model_coefficients.csv` → feature importance  

3. **Clustering & Feature Exploration**  
   - Grouped students by performance patterns.  
   - Identified clusters of common weaknesses (e.g., reading comprehension vs writing).  

4. **Frontend Dashboard (Streamlit)**  
   - Interactive UI with pages:  
     - Overview  
     - Learning Gaps  
     - Equity  
     - Overlap  
     - Triage  
     - Clusters  
     - Students  
   - Visualizes results in an **easy-to-use equity dashboard**.  
   - Built with **Streamlit**, runs fully inside **Docker**.

---

## 📂 Repository Structure
├── Data Processing/ # Scripts for cleaning & aggregating raw data
├── Train Risk Model/ # Risk prediction models
├── clusters_feature_dashboard/ # Clustering and feature exploration
├── frontend/ # Streamlit dashboard (main entry point)
│ ├── app.py
│ ├── pages/
│ ├── utils/
│ ├── data_proc.zip
│ └── requirements.txt
└── README.md # This file

##To Run the dashboard:
### 1. With Docker (recommended)
From inside the `frontend/` folder:
```bash
docker build -t equity-dashboard .
docker run --rm -p 8501:8501 equity-dashboard

## 2. Without Docker
cd frontend
pip install -r requirements.txt
streamlit run app.py

📊 Outputs

Aggregated Data: data_proc/agg_region_grade_subject.csv

Risk Scores: data_proc/student_risk_scores.csv

Equity Metrics: data_proc/agg_gender.csv, overlap.csv

Model Coefficients: risk_model_coefficients.csv

