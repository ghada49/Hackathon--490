import os
import pandas as pd
import streamlit as st
from pathlib import Path
import json

FRONTEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = FRONTEND_DIR / "data_proc"

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

@st.cache_data
def load_student_risks() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "student_risk_scores.csv")
    return pd.read_csv(path, encoding="utf-8-sig")

@st.cache_data
def load_risk_cohort() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "risk_by_region_grade_subject.csv")
    return pd.read_csv(path, encoding="utf-8-sig")

@st.cache_data
def load_risk_coeffs() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "risk_model_coefficients.csv")
    return pd.read_csv(path, encoding="utf-8-sig")

@st.cache_data
def load_risk_metrics() -> list:
    path = os.path.join(DATA_DIR, "risk_model_metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _find_data(path_str: str):
    """Try common locations for data_proc/* regardless of how Streamlit was launched."""
    here = Path(__file__).resolve()
    candidates = [
        Path(path_str),                         # as-is
        Path.cwd() / path_str,                  # from current working dir
        here.parents[2] / path_str,             # repo root (…/frontend/utils -> up 2)
        here.parents[1] / path_str,             # …/frontend
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def load_risk_metrics():
    p = _find_data("data_proc/risk_model_metrics.json")
    if p is None:
        return []
    import json
    return json.loads(Path(p).read_text(encoding="utf-8"))
