"""Home: where the season stands, in one screen.

Built for a phone held in one hand: compact rows rather than metric cards,
which are about half a screen tall each and mostly empty space.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ucl.webapp import cards, loaders, movement, paths, st_helpers
from ucl.webapp.palette import AWAY, HOME

SEASON = paths.SEASON
PURPLE = "#a78bfa"

status = loaders.load_status(SEASON)
try:
    information = loaders.load_information(SEASON)
except FileNotFoundError:
    information = None

matchday = paths.latest_matchday(SEASON)

# --- header ---------------------------------------------------------------
left, right = st.columns([3, 2], vertical_alignment="center")
left.title("UCL 2026-27")
with right:
    st_helpers.freshness_badge(status, information)

headline = f"**Matchday {matchday}** played" if matchday else "Pre-tournament"
next_matchday = status.get("next_matchday")
if next_matchday:
    try:
        fixtures = loaders.load_next_matchday(next_matchday, SEASON)
        days = sorted({d.date() for d in fixtures.date})
        when = (days[0].strftime("%d %b") if len(days) == 1
                else f"{days[0].strftime('%d')}-{days[-1].strftime('%d %b')}")
        headline += f" · **matchday {next_matchday}** on {when}"
    except FileNotFoundError:
        headline += f" · matchday {next_matchday} next"
st.markdown(headline)

try:
    forecast = loaders.load_forecast(matchday, SEASON)
except FileNotFoundError as error:
    st_helpers.missing(error)
    st.stop()

# --- favourites and movers, side by side on a desktop, stacked on a phone -
favourites, movers = st.columns(2, gap="medium")

with favourites:
    st_helpers.section("Title favourites", "probability of winning the final")
    top = forecast.nlargest(5, "p_champion")
    st.html(cards.compact_rows_html([
        {"team": row.team, "value": row.p_champion, "colour": PURPLE,
         "delta": getattr(row, "d_p_champion", None)}
        for row in top.itertuples()
    ]))

with movers:
    moved = movement.movement(SEASON)
    previous = moved.attrs.get("previous_matchday") if len(moved) else None
    st_helpers.section(
        "Biggest movers",
        f"change in P(top 8), matchday {previous} to {matchday}"
        if len(moved) else "needs a second matchday")
    if len(moved):
        ranked = movement.biggest_movers(moved, "d_p_top8", count=5)
        st.html(cards.mover_rows_html([
            {"team": row.team, "label": "top 8",
             "before": row.p_top8_prev, "after": row.p_top8,
             "delta": row.d_p_top8}
            for row in ranked.itertuples()
        ]))
    else:
        st.html(cards.mover_rows_html([]))

# --- the band the table is in --------------------------------------------
st_helpers.section("The three bands",
                   "1-8 straight to the round of 16 · 9-24 play-off · 25-36 out")
eight = forecast.nlargest(3, "p_top8")
out = forecast.nlargest(3, "p_eliminated")
band_one, band_two = st.columns(2, gap="medium")
with band_one:
    st.html(cards.compact_rows_html([
        {"team": row.team, "value": row.p_top8, "colour": HOME,
         "delta": getattr(row, "d_p_top8", None)} for row in eight.itertuples()]))
with band_two:
    st.html(cards.compact_rows_html([
        {"team": row.team, "value": row.p_eliminated, "colour": AWAY,
         "delta": getattr(row, "d_p_eliminated", None)} for row in out.itertuples()]))

# --- next fixtures --------------------------------------------------------
if next_matchday:
    try:
        upcoming = loaders.load_next_matchday(next_matchday, SEASON)
    except FileNotFoundError:
        upcoming = None
    if upcoming is not None and len(upcoming):
        st_helpers.section(f"Next up, matchday {next_matchday}",
                           "the three closest fixtures")
        closest = upcoming.assign(
            margin=(upcoming[["p_home", "p_draw", "p_away"]].max(axis=1)
                    - upcoming[["p_home", "p_draw", "p_away"]].min(axis=1))
        ).nsmallest(3, "margin")
        rows = closest.assign(date=closest.date.dt.strftime("%d %b")).to_dict("records")
        st.html(cards.fixture_cards_html(rows))

st_helpers.note(
    "The market comparison, the per-fixture probabilities and the scoring all "
    "live on their own pages; this screen is the state of the season."
)
