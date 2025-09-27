# frontend/pages/5_Triage.py
import pandas as pd
import streamlit as st

from utils.data import (
    load_risk_cohort, load_student_risks, load_risk_coeffs, load_risk_metrics
)
from utils.charts import bar_horizontal
from utils.stats import kpi_row, aggregate_domain_importance

st.set_page_config(page_title="Triage", page_icon="🧭", layout="wide")
st.title("Student Triage")

# ---------- Load model outputs ----------
cohort  = load_risk_cohort()        # data_proc/risk_by_region_grade_subject.csv
risks   = load_student_risks()      # data_proc/student_risk_scores.csv
coefs   = load_risk_coeffs()        # data_proc/risk_model_coefficients.csv
metrics = load_risk_metrics()       # data_proc/risk_model_metrics.json (not used directly here)

# ---------- Filters (only values that exist in risk files) ----------
grades   = sorted(cohort["grade"].dropna().unique().tolist())
subjects = sorted(cohort["subject"].dropna().unique().tolist())
regions  = ["All"] + sorted(cohort["region"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filters")
    g = st.selectbox("Grade", grades, index=0)
    s = st.selectbox("Subject", subjects, index=0)
    r = st.selectbox("Region", regions, index=0)
    thr = st.slider("High-risk threshold", 0.50, 0.95, 0.85, 0.01)

# ---------- Slice data ----------
cohort_gs = cohort.query("grade == @g and subject == @s").copy()
if r != "All":
    cohort_gs = cohort_gs.query("region == @r")

# KPIs
n_regions    = cohort_gs["region"].nunique()
avg_risk_val = cohort_gs["avg_risk"].mean() if len(cohort_gs) else 0.0
p_thr        = cohort_gs["pct_students_above80"].mean() if "pct_students_above80" in cohort_gs else float("nan")
n_students   = int(cohort_gs["n_students"].sum()) if "n_students" in cohort_gs else 0

kpi_row([
    ("Regions shown", str(n_regions), None),
    ("Average risk", f"{avg_risk_val:.1f}%", "Mean probability of being below proficiency"),
    (f"% students ≥{int(thr*100)}% risk", f"{p_thr:.1f}%", "Share of high-risk students"),
    ("Students (total)", f"{n_students:,}", None),
])

# ---------- Regions by risk table ----------
st.subheader(f"Regions by risk — Grade {g} {s}")
tbl = cohort_gs.sort_values("avg_risk", ascending=False).reset_index(drop=True)
styled = (
    tbl.rename(columns={
        "avg_risk": "Average risk (%)",
        "pct_students_above80": f"% students ≥{int(thr*100)}%",
        "n_students": "N students"
    })
    .style
    .background_gradient(subset=["Average risk (%)"], cmap="Reds")
)
st.dataframe(styled, use_container_width=True)

# Download
csv = tbl.to_csv(index=False).encode("utf-8-sig")
st.download_button("Download table (CSV)", csv, file_name=f"triage_grade{g}_{s}.csv", mime="text/csv")

# ---------- Drilldown: top-risk students (if region chosen) ----------
if r != "All":
    st.subheader(f"Top-risk students — {r} · Grade {g} · {s}")
    top = (
        risks.query("region == @r and grade == @g and subject == @s")
             .sort_values("risk_prob_below", ascending=False)
             .assign(risk_pct=lambda d: (100 * d["risk_prob_below"]).round(1))
             [["student_id","school_id","risk_pct"]]
             .head(150)
    )
    st.dataframe(top, use_container_width=True)

# ---------- Drivers: DOMAIN-LEVEL coefficients ----------
st.subheader("Strongest drivers (model coefficients, domain-level)")
coef_slice = coefs.query("grade == @g and subject == @s")[["feature","coef"]].copy()

if coef_slice.empty:
    st.info("No model coefficients available for this cohort (a heuristic risk was used).")
else:
    dom_imp = aggregate_domain_importance(coef_slice)  # domain, weight, mean_sign

    # You can show just one list (top domains) …
    top_domains = dom_imp.head(10).rename(columns={"domain":"Domain","weight":"Importance"})
    st.altair_chart(
        bar_horizontal(top_domains, x="Importance", y="Domain", title="Domain importance (sum |coef|)"),
        use_container_width=True
    )

    # …or split by sign (optional)
    # pos = dom_imp.sort_values(["weight","mean_sign"], ascending=[False, False]).head(8)
    # neg = dom_imp.sort_values(["weight","mean_sign"], ascending=[False, True]).head(8)
    # c1, c2 = st.columns(2)
    # with c1:
    #     st.caption("Risk ↑ (domains)")
    #     st.altair_chart(bar_horizontal(pos.rename(columns={"domain":"Domain","weight":"Importance"}),
    #                                    x="Importance", y="Domain", title="Top domains increasing risk"),
    #                     use_container_width=True)
    # with c2:
    #     st.caption("Risk ↓ (domains)")
    #     st.altair_chart(bar_horizontal(neg.rename(columns={"domain":"Domain","weight":"Importance"}),
    #                                    x="Importance", y="Domain", title="Top domains decreasing risk"),
    #                     use_container_width=True)

# ---------- Suggested actions ----------
st.subheader("Suggested actions")
if len(tbl):
    top_regions = tbl.head(2)["region"].tolist()
    bullet = f"""
- Prioritise **{', '.join(top_regions)}** for small-group tutoring and weekly progress checks.  
- Allocate extra teacher support to classes with the highest **% students ≥{int(thr*100)}% risk**.  
- Use the **Learning Gaps** tab to target weak domains (e.g., *Reading Comprehension*, *Writing*).  
"""
    st.markdown(bullet)

st.caption("Model quality — see `data_proc/risk_model_metrics.json` for AUC/AP and calibration details.")
