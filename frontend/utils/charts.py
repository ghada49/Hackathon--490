import altair as alt
import pandas as pd
import streamlit as st

# ---------- Small helpers ----------
def _fmt_pct():
    return alt.Axis(format=".0f")

def kpi_row(items):
    """items: list of (label, value, helptext_or_None)"""
    cols = st.columns(len(items))
    for c, (label, value, helptext) in zip(cols, items):
        with c:
            st.metric(label, value)
            if helptext:
                st.caption(helptext)

def bar_horizontal(df: pd.DataFrame, x: str, y: str, title: str = ""):
    """
    Generic horizontal bar (Feature on Y, numeric on X).
    """
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:Q", title=title),
            y=alt.Y(f"{y}:N", sort="-x"),
            tooltip=[y, alt.Tooltip(f"{x}:Q", format=".3f")],
        )
        .properties(height=max(28 * len(df), 140))
    )
    return chart

# ---------- Other charts you already use ----------
def heatmap(df: pd.DataFrame, value="pct_below", title="% below proficiency",
            scheme="reds", zero=False, domain=None):
    base = alt.Chart(df).transform_filter(alt.datum[value] != None)
    color_scale = alt.Scale(scheme=scheme, domain=domain, nice=not domain, zero=zero)
    return (
        base.mark_rect()
        .encode(
            y=alt.Y("region:N", title="Region", sort=alt.SortField("region", order="ascending")),
            x=alt.X("grade:O", title="Grade"),
            color=alt.Color(f"{value}:Q", scale=color_scale, title=title),
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("grade:O",  title="Grade"),
                alt.Tooltip("subject:N", title="Subject"),
                alt.Tooltip(f"{value}:Q", title=title, format=".1f"),
            ],
        )
        .properties(height=360)
    )

def bar_top_regions(df: pd.DataFrame, value="pct_below", topn=5, title="% below proficiency"):
    top = (df[["region", value]].dropna().sort_values(value, ascending=False).head(topn))
    return (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title=title, axis=_fmt_pct()),
            y=alt.Y("region:N", sort="-x", title="Region"),
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip(f"{value}:Q", title=title, format=".1f"),
            ],
        )
        .properties(height=36 * len(top))
    )

def bar_coefficients(df: pd.DataFrame, feature_col="feature", coef_col="coef", topn=8):
    d = (
        df[[feature_col, coef_col]]
        .dropna()
        .sort_values(coef_col, ascending=False)
        .head(topn)
        .rename(columns={feature_col: "Feature", coef_col: "Weight"})
    )
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X("Weight:Q", title="Coefficient (risk ↑)"),
            y=alt.Y("Feature:N", sort="-x"),
            tooltip=["Feature", alt.Tooltip("Weight:Q", format=".3f")],
        )
        .properties(height=28 * len(d))
    )

def bar_risk_regions(df: pd.DataFrame, risk_col="avg_risk", p80_col="pct_students_above80"):
    d = df[["region", risk_col, p80_col]].dropna()
    return (
        alt.Chart(d)
        .transform_fold([risk_col, p80_col], as_=["metric", "value"])
        .mark_bar()
        .encode(
            y=alt.Y("region:N", sort="-x", title="Region"),
            x=alt.X("value:Q", title="%", axis=_fmt_pct()),
            color=alt.Color("metric:N", title="Metric", scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip(f"{risk_col}:Q", title="Avg risk", format=".1f"),
                alt.Tooltip(f"{p80_col}:Q", title="% ≥80% risk", format=".1f"),
            ],
        )
    )

# frontend/utils/charts.py
import altair as alt
import pandas as pd

def _fmt_pct():
    return alt.Axis(format=".0f")

def bar_horizontal(df: pd.DataFrame, x: str, y: str, title: str = ""):
    """
    Generic horizontal bar chart.
    """
    d = df[[y, x]].dropna()
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:Q", title=title),
            y=alt.Y(f"{y}:N", sort="-x", title=""),
            tooltip=[alt.Tooltip(f"{y}:N", title=""), alt.Tooltip(f"{x}:Q", title=title, format=".3f")],
        )
        .properties(height=max(28 * len(d), 120))
    )
