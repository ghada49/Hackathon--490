# frontend/utils/stats.py
import re
import pandas as pd

def kpi_row(items):
    """Small helper to render 3–4 KPIs as columns.
    items: list of tuples -> [(title, value, help), ...]
    """
    import streamlit as st
    cols = st.columns(len(items))
    for col, (title, value, helptext) in zip(cols, items):
        with col:
            st.metric(label=title, value=value, help=helptext)

# ---------- Domain-level explainability helpers ----------

def feature_to_domain(raw: str) -> str:
    """
    Map raw feature names to human domain labels.

    Examples
    --------
    'Reading Comprehension__Unnamed: 11' -> 'Reading Comprehension'
    'Writing__Section 2: Writing'        -> 'Writing'
    'region_Bekaa'                       -> 'Region'
    'gender_Male'                        -> 'Gender'
    'AUTO__Unnamed: 8'                   -> 'Reading Comprehension' (if upstream created '<domain>__<col>')
    """
    s = str(raw)
    if s.startswith("region_"):
        return "Region"
    if s.startswith("gender_"):
        return "Gender"
    if "__" in s:
        # canonical format created by your model scripts: "<domain>__<column>"
        return s.split("__", 1)[0]
    # Fallback: clean generic labels; treat any 'Unnamed:' as 'Item'
    s = re.sub(r"Unnamed:\s*\d+", "Item", s)
    # If still nothing domain-like, return original so we don't drop info
    return s

def aggregate_domain_importance(coef_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate feature-level coefficients into domain-level importances.

    Strategy:
      - Importance = sum of absolute weights per domain (stable, order-invariant).
      - mean_sign  = average coefficient sign (lets you split pos/neg lists if you want).

    Returns columns: ['domain', 'weight', 'mean_sign']
    """
    if coef_df.empty:
        return coef_df

    tmp = coef_df.copy()
    if "feature" not in tmp.columns or "coef" not in tmp.columns:
        raise ValueError("coef_df must have columns ['feature','coef']")

    tmp["domain"] = tmp["feature"].apply(feature_to_domain)
    out = (
        tmp.assign(abs_w=tmp["coef"].abs())
           .groupby("domain", as_index=False)
           .agg(weight=("abs_w", "sum"), mean_sign=("coef", "mean"))
           .sort_values("weight", ascending=False, ignore_index=True)
    )
    return out

def pretty_label(s: str) -> str:
    """Optional: cleaner labels for any item strings you DO decide to show."""
    s = str(s)
    s = s.replace("Unnamed:", "Item ").replace("__", " → ")
    s = s.replace("Section 1:", "").replace("Section 2:", "").strip()
    return s
