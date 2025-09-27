import streamlit as st
import pandas as pd
import altair as alt
from utils.data import load_gender, available_subjects, available_grades

st.set_page_config(layout="wide")

# Load gender aggregated data
df = load_gender()

# Filters
subject = st.selectbox("Subject", available_subjects(df))
grade = st.selectbox("Grade", available_grades(df))
dim = st.selectbox("Equity dimension", ["Gender"], index=0, disabled=True)

# Filter dataset
cur = df[(df.subject == subject) & (df.grade == grade)]

st.subheader("Equity Analysis")

if not cur.empty:
    # Build dataframe for Male vs Female bars
    rows = []
    for _, row in cur.iterrows():
        rows.append({"Group": row["subgroup_a"], "% Below": row["pct_below_a"]})
        rows.append({"Group": row["subgroup_b"], "% Below": row["pct_below_b"]})

    bar_df = pd.DataFrame(rows)

    # Bar chart: Male vs Female
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
    st.caption("p < 0.05 indicates statistically significant gap")

# Significant gaps table
st.write("### Significant gaps")
st.caption("Δ = percentage-point difference")

cur_table = cur.rename(
    columns={
        "region": "Region",
        "grade": "Grade",
        "subject": "Subject",
        "subgroup_a": "Group A",
        "gap_pp": "Gap (pp)",
        "p_value": "p-value",
        "n_a": "nA",
        "n_b": "nB",
    }
)

st.dataframe(
    cur_table[
        ["Region", "Grade", "Subject", "Group A", "Gap (pp)", "p-value", "nA", "nB"]
    ]
)

st.download_button(
    "Download CSV",
    cur_table.to_csv(index=False).encode("utf-8"),
    file_name=f"equity_{subject}_G{grade}.csv",
)
