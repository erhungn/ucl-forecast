"""The next round's fixtures, one card each."""

from __future__ import annotations

import streamlit as st

from ucl.webapp import cards, loaders, paths, st_helpers

SEASON = paths.SEASON

st.title("Next matchday")

status = loaders.load_status(SEASON)
matchday = paths.latest_matchday(SEASON) + 1

if matchday > 8:
    st.success("The league phase is complete; there is no next matchday.")
    st.stop()

try:
    frame = loaders.load_next_matchday(matchday, SEASON)
except FileNotFoundError as error:
    st_helpers.missing(error)
    st.stop()

days = sorted({d.date() for d in frame.date})
when = (days[0].strftime("%d %B") if len(days) == 1
        else f"{days[0].strftime('%d')}-{days[-1].strftime('%d %B')}")
st.caption(
    f"Matchday {matchday}, {len(frame)} fixtures, {when}. Probabilities from "
    f"the model as of {status.get('as_of', 'the last update')} - no simulation "
    f"is involved, so they do not depend on the iteration count."
)

order = st.segmented_control(
    "Order", ["by date", "closest first", "most one-sided"], default="by date",
    label_visibility="collapsed")
spread = (frame[["p_home", "p_draw", "p_away"]].max(axis=1)
          - frame[["p_home", "p_draw", "p_away"]].min(axis=1))
if order == "closest first":
    frame = frame.assign(spread=spread).sort_values("spread")
elif order == "most one-sided":
    frame = frame.assign(spread=spread).sort_values("spread", ascending=False)
else:
    frame = frame.sort_values(["date", "home"])

rows = frame.assign(date=frame.date.dt.strftime("%d %b")).to_dict("records")
st.html(cards.fixture_cards_html(rows))

st_helpers.note(
    "The bar is home / draw / away, left to right. <b>pick</b> is simply the "
    "most likely of the three, which for most fixtures is a long way from a "
    "confident call, and the modal scoreline carries only ~8-12% probability - "
    "it is not a prediction."
)
