import streamlit as st
import pandas as pd
import altair as alt
from utils.data import load_region_subject, available_subjects

st.set_page_config(layout="wide")

df = load_region_subject()

# Filters
subject = st.selectbox("Subject", available_subjects(df))
grade = st.selectbox("Grade", sorted(df["grade"].unique()))

st.subheader("Learning Gaps — % Below Proficiency")

# Heatmap: region × grade, colored by % below
heatmap_data = df[df.subject == subject][["region","grade","pct_below"]]

heatmap = (
    alt.Chart(heatmap_data)
    .mark_rect()
    .encode(
        y=alt.Y("region:N", sort="-x", title=None),
        x=alt.X("grade:O", title=None),
        color=alt.Color("pct_below:Q", scale=alt.Scale(scheme="reds")),
        tooltip=["region","grade","pct_below"]
    )
    .properties(height=350)
)

# Top 5 worst regions for selected subject+grade
cur = df[(df.subject==subject) & (df.grade==grade)]
worst = cur.sort_values("pct_below", ascending=False).head(5)

bar = (
    alt.Chart(worst)
    .mark_bar(color="#E45756")
    .encode(
        x=alt.X("pct_below:Q", title="% Below Proficiency"),
        y=alt.Y("region:N", sort='-x', title=None),
        tooltip=["region","pct_below"]
    )
    .properties(height=200)
)

# Layout
left, right = st.columns([2,1])
with left:
    st.altair_chart(heatmap, use_container_width=True)
with right:
    st.write(f"### Top 5 Worst Regions ({subject})")
    st.altair_chart(bar, use_container_width=True)
    st.dataframe(worst[["region","pct_below"]].rename(columns={"region":"Region","pct_below":"% Below Proficiency"}))

# Download button
st.download_button(
    "Download learning gaps (CSV)",
    data=df[df.subject==subject].to_csv(index=False).encode('utf-8'),
    file_name=f"learning_gaps_{subject}.csv",
    mime='text/csv'
)
