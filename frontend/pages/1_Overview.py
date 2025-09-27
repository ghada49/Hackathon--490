import streamlit as st
import pandas as pd
import altair as alt

from utils.data import load_region_subject, available_subjects, available_grades
# Mapping Arabic → English for regions
region_map = {
    "بيروت": "Beirut",
    "جبل لبنان": "Mount Lebanon",
    "الشمال": "North",
    "الجنوب": "South",
    "البقاع": "Bekaa",
    "عكار": "Akkar",
    "النبطية": "Nabatieh",
    "بعلبك والهرمل": "Baalbek-Hermel",
}

st.set_page_config(layout="wide")

# Load subject-level data
df = load_region_subject()

# Filters
subject = st.selectbox("Subject", available_subjects(df))
grade = st.selectbox("Grade", available_grades(df))

# Load domain-level data if subject supports it
if subject in ["Arabic", "English", "French"]:
    domain_data = pd.read_csv("data_proc/agg_region_grade_subject_domain.csv")
    domain_df = domain_data[(domain_data.subject == subject) & (domain_data.grade == grade)]
    domains = sorted(domain_df["domain"].unique())
    selected_domain = st.selectbox("Domain", domains)
    cur = domain_df[domain_df.domain == selected_domain]
else:
    cur = df[(df.subject == subject) & (df.grade == grade)]

# Snapshot metrics
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Students (rows)", f"{cur.n_students.sum():,}")
with c2:
    st.metric("Avg Score", f"{cur.avg_score.mean():.1f}")
with c3:
    st.metric("% Below", f"{cur.pct_below.mean():.1f}%")

st.subheader("Learning Gaps by Region")

# ✅ Worst 5 regions bar chart
worst = cur.sort_values("pct_below", ascending=False).head(5)
bar = (
    alt.Chart(worst)
    .mark_bar(color="#4BA3F2")
    .encode(
        x=alt.X("pct_below:Q", title="% Below Proficiency"),
        y=alt.Y("region:N", sort='-x', title=None),
        tooltip=["region", "pct_below"]
    )
    .properties(height=300)
)

st.write(f"### Worst 5 Regions — {subject}, Grade {grade}" + 
         (f" — {selected_domain}" if subject in ["Arabic", "English", "French"] else ""))

st.altair_chart(bar, use_container_width=True)

st.dataframe(
    worst[["region","pct_below"]].rename(
        columns={"region":"Region","pct_below":"% Below Proficiency"}
    )
)

# ✅ Download snapshot
st.download_button(
    "Download snapshot (CSV)",
    data=cur.to_csv(index=False).encode('utf-8'),
    file_name=f"snapshot_{subject}_G{grade}.csv",
    mime="text/csv"
)
