import streamlit as st
import pandas as pd
import altair as alt
from utils.data import load_gender, available_subjects, available_grades

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

# Load subject-level gender equity data
df = load_gender()

# Filters
subject = st.selectbox("Subject", available_subjects(df))
grade = st.selectbox("Grade", available_grades(df))
dim = st.selectbox("Equity dimension", ["Gender"], index=0)

# If subject has domains, use domain-level file
if subject in ["Arabic", "English", "French"]:
    domain_data = pd.read_csv("data_proc/agg_gender_domain.csv")
    domain_df = domain_data[
        (domain_data.subject == subject) & (domain_data.grade == grade)
    ]
    domains = sorted(domain_df["domain"].unique())
    selected_domain = st.selectbox("Domain", domains)
    cur = domain_df[domain_df.domain == selected_domain]
else:
    cur = df[(df.subject == subject) & (df.grade == grade)]

st.subheader("Equity Analysis")

if not cur.empty:
    cur.columns = [c.strip().lower() for c in cur.columns]  # normalize

    if "pct_a" in cur.columns and "pct_b" in cur.columns:
        # ✅ Build Male vs Female bar chart
        rows = []
        for _, row in cur.iterrows():
            rows.append({"Group": "Male", "% Below": row["pct_a"]})
            rows.append({"Group": "Female", "% Below": row["pct_b"]})

        bar_df = pd.DataFrame(rows)

        bar = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("Group:N", title=None),
                y=alt.Y("% Below:Q", title="% Below Proficiency"),
                color=alt.Color(
                    "Group:N",
                    scale=alt.Scale(domain=["Male", "Female"], range=["#4BA3F2", "#F28E2B"]),
                ),
                tooltip=["Group", "% Below"],
            )
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        # Fallback to gap only
        st.info("Showing gap only (pct_a / pct_b not found in dataset)")
        gap_chart = (
            alt.Chart(cur)
            .mark_bar(color="#4BA3F2")
            .encode(
                x=alt.X("region:N", title="Region"),
                y=alt.Y("gap_pp:Q", title="Gap (Male − Female, % below)"),
                tooltip=["region", "gap_pp", "p_value"],
            )
        )
        st.altair_chart(gap_chart, use_container_width=True)

    st.caption("p < 0.05 indicates statistically significant gap")

# ✅ Significant gaps table with both Male & Female values
st.write("### Significant gaps")
st.caption("Δ = percentage-point difference")

cur_table = cur.rename(
    columns={
        "region": "Region",
        "grade": "Grade",
        "subject": "Subject",
        "pct_a": "% Below (Male)",
        "pct_b": "% Below (Female)",
        "gap_pp": "Gap (pp)",
        "p_value": "p-value",
        "n_a": "nA",
        "n_b": "nB",
    }
)

# Determine focus group
cur_table["Focus Group"] = cur_table.apply(
    lambda r: "Boys" if r["Gap (pp)"] > 0 else "Girls",
    axis=1,
)

st.dataframe(
    cur_table[
        ["Region", "Grade", "Subject", "% Below (Male)", "% Below (Female)", 
         "Gap (pp)", "p-value", "nA", "nB", "Focus Group"]
    ]
)

# ✅ Download button
st.download_button(
    "Download CSV",
    cur_table.to_csv(index=False).encode("utf-8"),
    file_name=f"equity_{subject}_G{grade}.csv",
)
