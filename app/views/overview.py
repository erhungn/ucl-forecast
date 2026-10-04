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

# The run banner, the freshness badge and the staleness warnings are written
# by st_helpers.banner() in Home.py, which runs on every page.
status = loaders.load_status(SEASON)

matchday = paths.latest_matchday(SEASON)

# --- header ---------------------------------------------------------------
# No freshness badge here: st_helpers.banner() writes it on every page now.
st.title("UCL 2026-27")

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

# --- the pinned club, if there is one -------------------------------------
# The pin lives in the URL, so it survives the page loads that tapping a club
# card causes, and saving the page to a phone's home screen is what makes it
# outlive the session. Nothing is written to disk and there are no accounts.
with st.expander("Pin a club", icon=":material/push_pin:"):
    pin = st_helpers.pin_control(sorted(forecast.team))
    st_helpers.note(
        "Shown at the top of this screen. It travels in the address, so add "
        "this page to your home screen to keep it.")

if pin and (forecast.team == pin).any():
    ranked = forecast.sort_values("p_champion", ascending=False).reset_index(drop=True)
    ranked["position"] = ranked.index + 1
    mine = ranked.loc[ranked.team == pin]
    st_helpers.section(pin, "your pinned club")
    st.html(cards.club_cards_html(mine.to_dict("records"), pin=pin))

# --- favourites and movers, side by side on a desktop, stacked on a phone -
favourites, movers = st.columns(2, gap="medium")

with favourites:
    st_helpers.section("Title favourites", "probability of winning the final")
    top = forecast.nlargest(5, "p_champion")
    st.html(cards.compact_rows_html([
        {"team": row.team, "value": row.p_champion, "colour": PURPLE,
         "delta": getattr(row, "d_p_champion", None)}
        for row in top.itertuples()
    ], pin=pin))

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
        ], pin=pin))
    else:
        st.html(cards.mover_rows_html([], pin=pin))

# --- the band the table is in --------------------------------------------
st_helpers.section("The three bands",
                   "1-8 straight to the round of 16 · 9-24 play-off · 25-36 out")
eight = forecast.nlargest(3, "p_top8")
out = forecast.nlargest(3, "p_eliminated")
band_one, band_two = st.columns(2, gap="medium")
with band_one:
    st.html(cards.compact_rows_html([
        {"team": row.team, "value": row.p_top8, "colour": HOME,
         "delta": getattr(row, "d_p_top8", None)}
        for row in eight.itertuples()], pin=pin))
with band_two:
    st.html(cards.compact_rows_html([
        {"team": row.team, "value": row.p_eliminated, "colour": AWAY,
         "delta": getattr(row, "d_p_eliminated", None)}
        for row in out.itertuples()], pin=pin))

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
