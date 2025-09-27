import streamlit as st
import pandas as pd
from utils.data import load_overlap, available_subjects

st.set_page_config(layout="wide")
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

# Load overlap dataset
df = load_overlap()

# Subject dropdown (from available subjects in data)
subject = st.selectbox("Subject", available_subjects(df))

# Grade dropdown (force always 1–6)
grade = st.selectbox("Grade", [1, 2, 3, 4, 5, 6])

# If subject has domains, check if domain-level overlap exists
domain_df = None
selected_domain = None
if subject in ["Arabic", "English", "French"]:
    try:
        domain_data = pd.read_csv("data_proc/overlap_domain.csv")
        domain_df = domain_data[
            (domain_data.subject == subject) & (domain_data.grade == grade)
        ]
        if not domain_df.empty:
            domains = sorted(domain_df["domain"].unique())
            selected_domain = st.selectbox("Domain", domains)
            cur = domain_df[domain_df.domain == selected_domain]
        else:
            st.warning(f"No domain-level data found for {subject}, Grade {grade}.")
            cur = pd.DataFrame()
    except FileNotFoundError:
        st.warning("⚠️ overlap_domain.csv not found.")
        cur = pd.DataFrame()
else:
    # Fallback to subject-level overlap
    cur = df[(df.subject == subject) & (df.grade == grade)]

# Main section
st.subheader(f"Overlap — Where learning gaps meet equity gaps (Priority cohorts)")
st.caption(f"{subject}, Grade {grade}" + (f" (Domain: {selected_domain})" if selected_domain else ""))

if cur.empty:
    st.warning("No overlap records found for this selection.")
else:
    # Rename columns for display
    table = cur.rename(
        columns={
            "region": "Region",
            "grade": "Grade",
            "subject": "Subject",
            "domain": "Domain" if "domain" in cur.columns else None,
            "focus_group": "Focus Group",
            "avg_score": "Avg Score",
            "pct_below": "% Below",
            "gap_pp": "Gap (pp)",
            "p_value": "p-value",
            "n_group": "N (group)",
        }
    ).dropna(axis=1, how="all")  # drop None columns

    # Show table
    st.dataframe(table)

    # Show narrative cards for flagged overlaps
    for _, row in cur.iterrows():
        if row.get("flag_overlap"):
            msg = (
                f"**{row['region']} — Grade {grade} {subject}"
                + (f" (Domain: {row['domain']})" if 'domain' in row else "")
                + "**\n\n"
                f"Learning gap: {row['pct_below']}% below proficiency (avg score {row['avg_score']}).\n\n"
                f"Gender gap: {row['gap_pp']} pp, p={row['p_value']}.\n\n"
                f"Focus: {row['focus_group']}. Suggested targeted support."
            )
            st.info(msg)

    # Download button
    st.download_button(
        "Download overlaps (CSV)",
        cur.to_csv(index=False).encode("utf-8"),
        file_name=f"overlaps_{subject}_G{grade}.csv",
    )
