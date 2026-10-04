"""Altair specifications for the movement and scoring charts.

Pure builders returning ``alt.Chart``, so a test can read ``to_dict()`` and
check the encodings and the colour range without a browser. Altair is already
installed as a Streamlit dependency; Plotly would have been a new pin for no
gain here.

Colours come from :mod:`ucl.webapp.palette`, the same list that
``.streamlit/config.toml`` sets as ``chartCategoricalColors``, so a chart drawn
here and one Streamlit draws itself agree.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from ucl.webapp import clubs
from ucl.webapp.palette import (
    AWAY, BORDER, CATEGORICAL, DRAW, FAINT, HOME, MUTED, TEXT,
)

__all__ = [
    "MAX_SERIES", "LABELS", "trend", "small_multiples", "running_score",
    "position_bars", "position_heatmap", "BAND_EDGES",
]

#: The last position in each qualifying band: 1-8 straight through, 9-24 to the
#: play-off, 25-36 out. Drawn as a rule so the heatmap reads against the format.
BAND_EDGES = (8, 24)

#: Beyond eight lines a categorical palette stops distinguishing them, which is
#: why the club picker caps rather than colouring all 36.
MAX_SERIES = 8

LABELS = {
    "p_champion": "P(champion)",
    "p_top8": "P(top 8)",
    "p_eliminated": "P(eliminated)",
    "mean_position": "Mean position",
}


def _base(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["code"] = out.team.map(clubs.code)
    return out


def _axes(column: str) -> tuple[alt.X, alt.Y]:
    percent = column != "mean_position"
    return (
        alt.X("matchday:O", title="matchday",
              axis=alt.Axis(labelColor=MUTED, titleColor=FAINT, grid=False)),
        alt.Y(f"{column}:Q", title=LABELS.get(column, column),
              axis=alt.Axis(format=".0%" if percent else ".1f",
                            labelColor=MUTED, titleColor=FAINT,
                            gridColor=BORDER),
              scale=alt.Scale(zero=column != "mean_position",
                              reverse=column == "mean_position")),
    )


def trend(series: pd.DataFrame, column: str, selected) -> alt.LayerChart:
    """Selected clubs in colour over every other club in faint grey.

    The context lines matter: a club rising is only interesting against what
    the rest of the field did.
    """
    frame = _base(series)
    selected = list(selected)[:MAX_SERIES]
    x, y = _axes(column)

    context = (
        alt.Chart(frame[~frame.team.isin(selected)])
        .mark_line(color=BORDER, strokeWidth=1, opacity=0.55)
        .encode(x=x, y=y, detail="team:N")
    )
    chosen = frame[frame.team.isin(selected)]
    highlight = (
        alt.Chart(chosen)
        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=38))
        .encode(
            x=x, y=y,
            color=alt.Color("code:N", title="club",
                            scale=alt.Scale(domain=sorted(chosen.code.unique()),
                                            range=CATEGORICAL),
                            legend=alt.Legend(labelColor=TEXT, titleColor=FAINT,
                                              orient="bottom", columns=4)),
            tooltip=["team:N", "matchday:O", alt.Tooltip(f"{column}:Q", format=".3f")],
        )
    )
    return (context + highlight).properties(height=330).configure_view(
        strokeWidth=0).configure(background="transparent")


def small_multiples(series: pd.DataFrame, column: str, selected,
                    *, columns: int = 2) -> alt.FacetChart:
    """One panel per club. Easier to read than eight lines in one frame."""
    frame = _base(series)
    selected = list(selected)[:MAX_SERIES]
    frame = frame[frame.team.isin(selected)]
    x, y = _axes(column)
    return (
        alt.Chart(frame)
        .mark_area(line={"color": HOME, "strokeWidth": 2}, opacity=0.25,
                   color=HOME)
        .encode(x=x, y=y,
                tooltip=["team:N", "matchday:O",
                         alt.Tooltip(f"{column}:Q", format=".3f")])
        .properties(width=150, height=110)
        .facet(facet=alt.Facet("code:N", title=None,
                               header=alt.Header(labelColor=TEXT,
                                                 labelFontWeight="bold")),
               columns=columns)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
    )


def running_score(frame: pd.DataFrame, metric: str) -> alt.Chart:
    """Cumulative log-loss or RPS by matchday, model series against uniform."""
    labels = sorted(frame.series.unique())
    palette = {"uniform": DRAW, "frozen at cutoff": HOME, "pre-matchday": AWAY}
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=45))
        .encode(
            x=alt.X("matchday:O", title="matchday",
                    axis=alt.Axis(labelColor=MUTED, titleColor=FAINT, grid=False)),
            y=alt.Y(f"{metric}:Q", title=metric.replace("_", "-"),
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelColor=MUTED, titleColor=FAINT,
                                  gridColor=BORDER)),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(
                                domain=labels,
                                range=[palette.get(name, CATEGORICAL[0])
                                       for name in labels]),
                            legend=alt.Legend(labelColor=TEXT, orient="bottom")),
            strokeDash=alt.StrokeDash(
                "series:N", legend=None,
                scale=alt.Scale(domain=labels,
                                range=[[4, 3] if name == "uniform" else [1, 0]
                                       for name in labels])),
            tooltip=["series:N", "matchday:O", alt.Tooltip(f"{metric}:Q", format=".4f")],
        )
        .properties(height=280)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
    )


def position_bars(frame: pd.DataFrame, team: str) -> alt.LayerChart:
    """One club's finishing-position distribution, with the band edges marked.

    The per-club slice of the position matrix. It reads at phone width where
    the full 36x36 grid is tight, and it is the question a person actually has
    about their own club.
    """
    mine = frame[frame.team == team]
    bars = (
        alt.Chart(mine)
        .mark_bar(color=HOME, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("position:O", title="finishing position",
                    axis=alt.Axis(values=[1, 4, 8, 12, 16, 20, 24, 28, 32, 36],
                                  labelColor=MUTED, titleColor=FAINT,
                                  labelAngle=0, grid=False)),
            y=alt.Y("probability:Q", title=None,
                    axis=alt.Axis(format=".0%", labelColor=MUTED,
                                  gridColor=BORDER)),
            tooltip=["position:O", alt.Tooltip("probability:Q", format=".2%")],
        )
    )
    edges = (
        alt.Chart(pd.DataFrame({"position": list(BAND_EDGES)}))
        .mark_rule(color=FAINT, strokeDash=[3, 3], xOffset=6)
        .encode(x=alt.X("position:O"))
    )
    return (bars + edges).properties(height=150)


def position_heatmap(frame: pd.DataFrame, order) -> alt.LayerChart:
    """P(finishing position) for every club, clubs down and positions across.

    Three choices make this legible on a phone rather than a wall of one
    colour. The scale is square-root: the mass sits on each club's own few
    positions, and a linear ramp leaves nine cells in ten indistinguishable.
    Cells under half a percent are dropped, so the band each club occupies is
    the only thing drawn. And rows are ordered by mean position, which turns
    the whole grid into a single descending diagonal - the point of the chart.
    """
    drawn = frame[frame.probability >= 0.005].copy()
    drawn["code"] = drawn.team.map(clubs.code)
    codes = [clubs.code(team) for team in order]
    cells = (
        alt.Chart(drawn)
        .mark_rect()
        .encode(
            x=alt.X("position:O", title=None,
                    axis=alt.Axis(values=[1, 8, 16, 24, 36], labelAngle=0,
                                  labelColor=MUTED, domain=False, ticks=False)),
            y=alt.Y("code:N", title=None, sort=codes,
                    axis=alt.Axis(labelColor=MUTED, labelFontSize=9,
                                  labelPadding=1, domain=False, ticks=False)),
            color=alt.Color(
                "probability:Q", legend=None,
                scale=alt.Scale(type="sqrt", domain=[0, 0.12], clamp=True,
                                range=["#16343a", HOME])),
            tooltip=["team:N", "position:O",
                     alt.Tooltip("probability:Q", format=".1%")],
        )
    )
    # A rule cannot sit between two ordinal categories, so the band edge is
    # drawn as the outline of its own column.
    edges = (
        alt.Chart(drawn[drawn.position.isin(BAND_EDGES)])
        .mark_rect(fill=None, stroke=TEXT, strokeWidth=0.6, strokeOpacity=0.5)
        .encode(x=alt.X("position:O"), y=alt.Y("code:N", sort=codes))
    )
    return (cells + edges).properties(height=36 * 12)
