import altair as alt
import pandas as pd


def heatmap(df: pd.DataFrame):
    return (
    alt.Chart(df)
    .mark_rect()
    .encode(
        y=alt.Y("region:N", title="Region"),
        x=alt.X("grade:O", title="Grade"),
        color=alt.Color("pct_below:Q", scale=alt.Scale(scheme="reds")),
        tooltip=["region","grade","subject","pct_below"]
    )
    .properties(height=350)
    )


def bar_top_regions(df: pd.DataFrame, topn: int = 5):
    top = df.nlargest(topn, "pct_below")
    return (
        alt.Chart(top)
        .mark_bar()
        .encode(
        x=alt.X("pct_below:Q", title="% Below Proficiency"),
        y=alt.Y("region:N", sort='-x'),
        tooltip=["region","pct_below"]
    )
    )