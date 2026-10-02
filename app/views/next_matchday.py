"""The next round's 18 fixtures, from the model frozen at the as-of date."""

from __future__ import annotations

import streamlit as st

from ucl.webapp import loaders, paths, st_helpers

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

st.caption(
    f"Matchday {matchday}, {len(frame)} fixtures, as of "
    f"{status.get('as_of', 'the last update')}. These are the fitted model's "
    f"own probabilities - no simulation is involved, so they do not depend on "
    f"the iteration count."
)

display = frame.copy()
display["date"] = display.date.dt.date
columns = ["date", "home", "away", "p_home", "p_draw", "p_away", "pick",
           "p_pick", "modal_score", "p_modal"]
st_helpers.probability_table(
    display[columns], percent=["p_home", "p_draw", "p_away", "p_pick", "p_modal"],
    height=36 * (len(display) + 1))

st.caption(
    "The modal scoreline carries only ~8-12% probability; it is not a "
    "prediction. `pick` is simply the most likely of the three outcomes, which "
    "for most fixtures is a long way from a confident call."
)
