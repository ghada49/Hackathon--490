import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Clustered Education Dashboard",
    page_icon="📊",
    layout="wide",
)

CSV_CANDIDATES = [
    "region_grade_clusters_hybrid.csv",
    os.path.join("frontend", "data_proc", "region_grade_clusters_hybrid.csv"),
    os.path.join("data_proc", "region_grade_clusters_hybrid.csv"),
]

REQUIRED_COLS = [
    "region_canonical","grade","cluster_label",
    "performance_percentile_global","performance_percentile_within_grade",
    "composite_overall","mean_z_avg","mean_z_ok","mean_z_size",
    "subjects_covered","entries",
    "strength_Arabic","strength_English","strength_French",
]

@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
        if path is None:
            st.warning("Upload a CSV in the sidebar or place "
                       "region_grade_clusters_hybrid.csv in the app folder.")
            st.stop()
        df = pd.read_csv(path)

    # Normalize expected columns/casing
    df.columns = [c.strip() for c in df.columns]

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
        st.stop()

    # Types / cleanup
    df["grade"] = df["grade"].astype(str)
    df["cluster_label"] = df["cluster_label"].astype(str).str.title()

    # Convenience
    df["region_grade"] = df["region_canonical"].astype(str) + " — G" + df["grade"].astype(str)
    return df

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙ Controls")

uploaded = st.sidebar.file_uploader(
    "Upload region_grade_clusters_hybrid.csv (optional)", type=["csv"]
)
df = load_data(uploaded)

grades = st.sidebar.multiselect(
    "Grade(s)", sorted(df["grade"].unique()), default=sorted(df["grade"].unique())
)
clusters = st.sidebar.multiselect(
    "Cluster label(s)", ["High","Medium","Low"], default=["High","Medium","Low"]
)
regions = st.sidebar.multiselect(
    "Region(s)", sorted(df["region_canonical"].unique()),
    default=sorted(df["region_canonical"].unique())
)
search = st.sidebar.text_input("Search region contains…", value="").strip()

mask = (
    df["grade"].isin(grades) &
    df["cluster_label"].isin(clusters) &
    df["region_canonical"].isin(regions)
)
if search:
    mask &= df["region_canonical"].str.contains(search, case=False, na=False)

f = df[mask].copy()

# -----------------------------
# Header
# -----------------------------
st.title("📊 Region × Grade Clusters (High / Medium / Low)")
st.caption("Source: region_grade_clusters_hybrid.csv")

# -----------------------------
# KPI Row
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows (region×grade)", f"{len(f):,}")
share_high = (f["cluster_label"].eq("High").mean()*100) if len(f) else 0
k2.metric("Share High (%)", f"{share_high:0.1f}")
k3.metric("Avg Composite", None if f.empty else round(float(f["composite_overall"].mean()), 3))
k4.metric("Avg Global Percentile", None if f.empty else round(float(f["performance_percentile_global"].mean()), 1))

# -----------------------------
# Chart 1: Count by Grade & Cluster (stacked)
# -----------------------------
st.subheader("Counts by Grade & Cluster")
count_df = (f.groupby(["grade","cluster_label"], as_index=False)
              .agg(n=("region_canonical","count")))
chart1 = (
    alt.Chart(count_df)
      .mark_bar()
      .encode(
          x=alt.X("grade:N", title="Grade", sort=sorted(df["grade"].unique())),
          y=alt.Y("n:Q", title="Count"),
          color=alt.Color("cluster_label:N", sort=["High","Medium","Low"],
                          scale=alt.Scale(scheme="tableau20")),
          tooltip=["grade","cluster_label","n"]
      )
      .properties(height=320)
)
st.altair_chart(chart1, use_container_width=True)

# -----------------------------
# Chart 2: Subject strengths by Cluster
# -----------------------------
st.subheader("Average Subject Strengths by Cluster")
long = f.melt(
    id_vars=["cluster_label"],
    value_vars=["strength_Arabic","strength_English","strength_French"],
    var_name="subject", value_name="strength"
)
sub_strength = long.groupby(["cluster_label","subject"], as_index=False)["strength"].mean()

chart2 = (
    alt.Chart(sub_strength)
      .mark_bar()
      .encode(
          x=alt.X("subject:N", title="Subject"),
          y=alt.Y("strength:Q", title="Mean z-strength"),
          column=alt.Column("cluster_label:N", sort=["High","Medium","Low"], title="Cluster"),
          tooltip=["cluster_label","subject","strength"]
      )
      .resolve_scale(y='shared')
      .properties(height=300)
)
st.altair_chart(chart2, use_container_width=True)

# -----------------------------
# Chart 3: Composite vs Percentile (scatter)
# -----------------------------
st.subheader("Composite vs Global Percentile")
scatter = px.scatter(
    f, x="composite_overall", y="performance_percentile_global",
    color="cluster_label",
    hover_data=["region_canonical","grade","strength_Arabic","strength_English","strength_French"],
    category_orders={"cluster_label":["High","Medium","Low"]},
    height=400
)
st.plotly_chart(scatter, use_container_width=True)

# -----------------------------
# Chart 4: Heatmap of Regions × Grades
# -----------------------------
st.subheader("Heatmap: Cluster by Region & Grade")
heat_df = f.copy()
heat_df["cluster_code"] = heat_df["cluster_label"].map({"High":2, "Medium":1, "Low":0})

heat = (
    alt.Chart(heat_df)
      .mark_rect()
      .encode(
          x=alt.X("grade:N", sort=sorted(df["grade"].unique())),
          y=alt.Y("region_canonical:N", sort=sorted(df["region_canonical"].unique())),
          color=alt.Color(
              "cluster_code:Q",
              scale=alt.Scale(domain=[0,1,2], range=["#d62728","#ffbf00","#2ca02c"]),
              legend=alt.Legend(title="Cluster", labelExpr="{'0':'Low','1':'Medium','2':'High'}[datum.label]")
          ),
          tooltip=["region_canonical","grade","cluster_label","composite_overall"]
      )
      .properties(height=28*max(6, heat_df['region_canonical'].nunique()))
)
st.altair_chart(heat, use_container_width=True)

# -----------------------------
# Detail Table + Download
# -----------------------------
st.subheader("Filtered Rows")
show_cols = [
    "region_canonical","grade","cluster_label",
    "performance_percentile_global","performance_percentile_within_grade",
    "composite_overall","mean_z_avg","mean_z_ok","mean_z_size",
    "subjects_covered","entries","strength_Arabic","strength_English","strength_French"
]
st.dataframe(
    f[show_cols].sort_values(["grade","cluster_label","region_canonical"]).reset_index(drop=True),
    use_container_width=True
)

csv_buf = io.StringIO()
f[show_cols].to_csv(csv_buf, index=False)
st.download_button("Download filtered CSV", data=csv_buf.getvalue(),
                   file_name="filtered_clusters.csv", mime="text/csv")

# -----------------------------
# Notes / Help
# -----------------------------
with st.expander("ℹ What am I looking at?"):
    st.markdown("""
- *Cluster label* comes from your hybrid pipeline (GMM, with percentile fallback).
- *Composite overall* is the mean of per-subject z-scores of average score and success rate.
- *Subject strengths* are per-subject z-scores (Arabic/English/French).
- *Percentiles: global is across all region×grade; *within-grade is relative only to the same grade.
""")