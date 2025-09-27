import streamlit as st
from utils.data import load_region_subject


st.set_page_config(
page_title="Equity & Learning Gaps Dashboard",
page_icon="📊",
layout="wide",
)


st.title("Equity & Learning Gaps Dashboard")


# Load data once and put in session
if "agg_df" not in st.session_state:
    st.session_state["agg_df"] = load_region_subject()


st.write("Navigate pages: Overview, Learning Gaps, Equity, Overlap.")
