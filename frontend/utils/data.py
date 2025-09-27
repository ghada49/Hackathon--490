import os
import pandas as pd
import streamlit as st


DATA_DIR = "data_proc"


@st.cache_data
def load_region_subject() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "agg_region_grade_subject.csv")
    return pd.read_csv(path)


@st.cache_data
def load_gender() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "agg_gender.csv")
    return pd.read_csv(path)


@st.cache_data
def load_overlap() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "overlap.csv")
    return pd.read_csv(path)


def available_subjects(df: pd.DataFrame):
    return sorted(df["subject"].dropna().unique().tolist())


def available_grades(df: pd.DataFrame):
    return sorted(df["grade"].dropna().unique().tolist())