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
domain_df = None
if subject in ["Arabic", "English", "French"]:
    domain_data = pd.read_csv("data_proc/agg_region_grade_subject_domain.csv")
    domain_df = domain_data[(domain_data.subject == subject) & (domain_data.grade == grade)]
    domains = sorted(domain_df["domain"].unique())
    selected_domain = st.selectbox("Domain", domains)
    cur = domain_df[domain_df.domain == selected_domain]
else:
    cur = df[(df.subject == subject) & (df.grade == grade)]

st.subheader("Learning Gaps — % Below Proficiency")

# Heatmap
heatmap = (
    alt.Chart(cur)
    .mark_rect()
    .encode(
        y=alt.Y("region:N", sort="-x", title=None),
        x=alt.X("grade:O", title="Grade"),
        color=alt.Color("pct_below:Q", scale=alt.Scale(scheme="reds")),
        tooltip=["region", "grade", "pct_below"],
    )
    .properties(height=350)
)

# Top 5 worst regions
worst = cur.sort_values("pct_below", ascending=False).head(5)
bar = (
    alt.Chart(worst)
    .mark_bar(color="#E45756")
    .encode(
        x=alt.X("pct_below:Q", title="% Below Proficiency"),
        y=alt.Y("region:N", sort="-x", title=None),
        tooltip=["region", "pct_below"],
    )
    .properties(height=200)
)

# Layout
left, right = st.columns([2, 1])
with left:
    st.altair_chart(heatmap, use_container_width=True)
with right:
    st.write(f"### Top 5 Worst Regions — {subject}, Grade {grade}")
    st.altair_chart(bar, use_container_width=True)
    st.dataframe(
        worst[["region", "pct_below"]].rename(
            columns={"region": "Region", "pct_below": "% Below Proficiency"}
        )
    )

# Download
st.download_button(
    "Download learning gaps (CSV)",
    data=cur.to_csv(index=False).encode("utf-8"),
    file_name=f"learning_gaps_{subject}_G{grade}.csv",
    mime="text/csv",
)
