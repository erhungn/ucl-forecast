"""Where every club finishes: the distribution behind the mean position.

``mean_position`` has always been published; the 36-way distribution it was
averaged from was computed 50,000 times and discarded. This view is that
distribution.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ucl.webapp import cards, charts, clubs, loaders, paths, st_helpers
from ucl.webapp.palette import AWAY, HOME, MUTED

SEASON = paths.SEASON
matchday = paths.latest_matchday(SEASON)

st.title("Projected table")

try:
    forecast = loaders.load_forecast(matchday, SEASON)
    positions = loaders.load_positions(matchday, SEASON)
except FileNotFoundError as error:
    st_helpers.missing(error)
    st.stop()

iterations = positions.attrs["iterations"]
order = forecast.sort_values("mean_position").team.tolist()
st.caption(
    f"Through matchday {matchday}, over {iterations:,} simulated seasons. "
    f"Clubs ordered by mean finishing position, so the band each one occupies "
    f"runs down the diagonal.")

# --- the bands, which are what the format actually decides ----------------
st_helpers.section("The three bands",
                   "1-8 straight to the round of 16 · 9-24 play-off · 25-36 out")
pin = st_helpers.pinned(SEASON)
bands = positions.assign(
    band=pd.cut(positions.position, bins=[0, 8, 24, 36],
                labels=["1-8", "9-24", "25-36"])
).groupby(["team", "band"], observed=True).probability.sum().unstack()
bands = bands.reindex(order)

st.html("".join(
    f'<div style="display:flex;align-items:center;gap:6px;padding:5px 0;'
    f'border-bottom:1px solid #1e293b">'
    f'{cards.badge_html(team)}'
    f'<span style="color:#94a3b8;font:500 11px/1.3 {cards.FONT};'
    f'width:34px;text-align:right">'
    f'{forecast.loc[forecast.team == team, "mean_position"].iloc[0]:.1f}</span>'
    f'{cards.stacked_bar_html(row["1-8"], row["9-24"], row["25-36"], height=18)}'
    f'</div>'
    for team, row in bands.iterrows()))
st_helpers.note(
    "The bar is P(1-8) / P(9-24) / P(25-36), left to right, and the small "
    "number is the mean finishing position. These three sum to one by "
    "construction and match the title-race page exactly - they are the same "
    "counts, grouped.")

# --- the full grid --------------------------------------------------------
st_helpers.section("Every position",
                   "darker is likelier; cells under 0.5% are left blank")
st.altair_chart(charts.position_heatmap(positions, order),
                width="stretch")
st_helpers.note(
    "Shading is on a square-root scale: a club's mass sits on a handful of "
    "positions, and a linear ramp would leave nine cells in ten looking "
    "identically empty. The outlined columns are 8th and 24th, the two places "
    "the format turns on. Tap-and-hold a cell on a desktop for the number; "
    "the cells are 9px on a phone, which is below a reliable tap target, so "
    "the per-club chart on a club's own page is the better way to read one "
    "club's distribution."
)

# --- the extremes, which the grid cannot show precisely ------------------
st_helpers.section("Sharpest and flattest", "how settled each club's finish is")
spread = positions.groupby("team").probability.max().reindex(order)
sharp, flat = spread.nlargest(5), spread.nsmallest(5)
left, right = st.columns(2, gap="medium")
with left:
    st_helpers.note("most concentrated")
    st.html(cards.compact_rows_html(
        [{"team": team, "value": float(value), "colour": HOME}
         for team, value in sharp.items()], pin=pin))
with right:
    st_helpers.note("most open")
    st.html(cards.compact_rows_html(
        [{"team": team, "value": float(value), "colour": AWAY}
         for team, value in flat.items()], pin=pin))
st_helpers.note(
    "The number is the club's single likeliest finishing position's "
    "probability. Even the most settled club clears only a few per cent: "
    "eight matches decide a 36-club table, so almost nothing about the finish "
    "is determined."
)
