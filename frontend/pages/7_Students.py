import math
import pandas as pd
import streamlit as st
from utils.data import load_student_risks

# Optional: increase Styler limit (can be omitted if using the gate below)
pd.set_option("styler.render.max_elements", 1_000_000)

st.set_page_config(page_title="Students", page_icon="👩‍🎓", layout="wide")
st.title("Per-student Risk")

# ---------- Load ----------
risks = load_student_risks()

# ---------- Clean types ----------
risks["grade"] = pd.to_numeric(risks["grade"], errors="coerce")
risks = risks.dropna(subset=["grade"]).copy()
risks["grade"] = risks["grade"].astype(int)

if "risk_prob_below" in risks.columns:
    # keep more precision so 0.9996 shows as 99.96, not 100.0
    risks["risk_pct"] = (risks["risk_prob_below"] * 100).round(4)


# ---------- Filters ----------
cols = st.columns(5)
with cols[0]:
    g = st.selectbox("Grade", sorted(risks["grade"].unique().tolist()))
with cols[1]:
    s = st.selectbox("Subject", sorted(risks.query("grade == @g")["subject"].dropna().unique().tolist()))
with cols[2]:
    r = st.selectbox("Region", ["All"] + sorted(risks.query("grade == @g and subject == @s")["region"].dropna().unique().tolist()))
with cols[3]:
    thr = st.slider("Highlight ≥", 0.50, 0.99, 0.85, 0.01)
with cols[4]:
    q = st.text_input("Search student_id…", "")

# ---------- Apply filters ----------
cur = risks.query("grade == @g and subject == @s").copy()
if r != "All":
    cur = cur.query("region == @r")
if q:
    cur = cur[cur["student_id"].astype(str).str.contains(q, case=False, na=False)]

# Optional: School/Class filters
c1, c2 = st.columns(2)
with c1:
    schools = ["All"] + sorted(cur["school_id"].dropna().astype(str).unique().tolist())
    school = st.selectbox("School", schools, index=0)
    if school != "All":
        cur = cur[cur["school_id"].astype(str) == school]
with c2:
    if "class_name" in cur.columns:
        classes = ["All"] + sorted(cur["class_name"].dropna().astype(str).unique().tolist())
        cls = st.selectbox("Class", classes, index=0)
        if cls != "All":
            cur = cur[cur["class_name"].astype(str) == cls]

# ---------- Prepare table ----------
cols_to_show = [c for c in ["student_id","school_id","class_name","region","grade","subject","modality","risk_prob_below","risk_pct"] if c in cur.columns]
cur = (cur.sort_values("risk_prob_below", ascending=False)[cols_to_show]
          .reset_index(drop=True))

st.caption(f"{len(cur):,} students")

# ---------- Pagination ----------
rows_per_page = st.sidebar.slider("Rows per page", 50, 1000, 200, 50)
total_rows = len(cur)
total_pages = max(1, math.ceil(total_rows / rows_per_page))
page = st.sidebar.number_input("Page", 1, total_pages, 1)
start = (page - 1) * rows_per_page
end = start + rows_per_page
view = cur.iloc[start:end].copy()

st.caption(f"Showing rows {start+1:,}–{min(end,total_rows):,} of {total_rows:,}")

# ---------- Safe styling gate ----------
MAX_STYLE_CELLS = 250_000
num_cells = int(view.shape[0] * view.shape[1])

if num_cells <= MAX_STYLE_CELLS and "risk_pct" in view.columns:
    st.dataframe(
        view.style.background_gradient(
            subset=["risk_pct"],
            cmap="Reds"
        ),
        use_container_width=True
    )
else:
    if num_cells > MAX_STYLE_CELLS:
        st.info("Large result set—showing without styling for performance.")
    st.dataframe(view, use_container_width=True)

# ---------- Downloads ----------
st.download_button(
    "Download current page (CSV)",
    view.to_csv(index=False, encoding="utf-8-sig"),
    file_name="students_page.csv",
    mime="text/csv",
)
st.download_button(
    "Download current view (filtered, all rows) (CSV)",
    cur.to_csv(index=False, encoding="utf-8-sig"),
    file_name=f"students_grade{g}_{s}_{r}.csv",
    mime="text/csv"
)
