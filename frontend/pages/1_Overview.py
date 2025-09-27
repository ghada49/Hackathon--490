import streamlit as st
import pandas as pd
import altair as alt
from utils.data import load_region_subject, available_subjects, available_grades

st.set_page_config(layout="wide")

df = load_region_subject()

# Filters
subject = st.selectbox("Subject", available_subjects(df))
grade = st.selectbox("Grade", available_grades(df))

cur = df[(df.subject==subject) & (df.grade==grade)]

# Snapshot metrics
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Students (rows)", f"{cur.n_students.sum():,}")
with c2:
    st.metric("Avg Score", f"{cur.avg_score.mean():.1f}")
with c3:
    st.metric("% Below", f"{cur.pct_below.mean():.1f}%")

# Learning gaps by region — choropleth placeholder + bar chart
st.subheader("Learning Gaps by Region")

left, right = st.columns([2,1])

# Choropleth placeholder (Lebanon map not implemented unless geojson provided)
with left:
    st.caption("Lebanon choropleth can be added here if geojson is available.")
    # Example: plotly choropleth integration if you add geojson file

# Worst 5 regions for subject+grade
worst = cur.sort_values("pct_below", ascending=False).head(5)
bar = (
    alt.Chart(worst)
    .mark_bar(color="#4BA3F2")
    .encode(
        x=alt.X("pct_below:Q", title="% Below Proficiency"),
        y=alt.Y("region:N", sort='-x', title=None),
        tooltip=["region","pct_below"]
    )
    .properties(height=200)
)

with right:
    st.write(f"### Worst 5 Regions — {subject}, Grade {grade}")
    st.altair_chart(bar, use_container_width=True)
    st.dataframe(worst[["region","pct_below"]].rename(columns={"region":"Region","pct_below":"% Below Proficiency"}))

# Download snapshot
st.download_button(
    "Download snapshot (CSV)",
    data=cur.to_csv(index=False).encode('utf-8'),
    file_name=f"snapshot_{subject}_G{grade}.csv",
    mime='text/csv'
)
