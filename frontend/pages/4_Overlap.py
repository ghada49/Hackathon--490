import streamlit as st
import pandas as pd
from utils.data import load_overlap, available_subjects, available_grades


st.set_page_config(layout="wide")


df = load_overlap()


# User chooses subject & grade explicitly
subject = st.selectbox("Subject", available_subjects(df))
grade = st.selectbox("Grade", available_grades(df))


st.subheader(f"Overlap — Where learning gaps meet equity gaps (Priority cohorts)\n{subject}, Grade {grade}")


cur = df[(df.subject==subject) & (df.grade==grade)]


st.dataframe(cur)


for _, row in cur.iterrows():
    if row.get("flag_overlap"):
        st.info(
        f"**{row['region']} — Grade {grade} {subject}**\n\n"
        f"Learning gap: {row['pct_below']}% below proficiency (avg score {row['avg_score']}).\n\n"
        f"Gender gap: {row['gap_pp']} pp, p={row['p_value']}.\n\n"
        f"Focus: {row['focus_group']}. Suggested targeted support."
        )


st.download_button("Download overlaps (CSV)", cur.to_csv(index=False).encode('utf-8'), file_name=f"overlaps_{subject}_G{grade}.csv")