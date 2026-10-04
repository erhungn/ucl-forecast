"""The next round's fixtures, one card each."""

from __future__ import annotations

import streamlit as st

from ucl.models import scorelines
from ucl.webapp import cards, clubs, loaders, paths, st_helpers

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

# One card per element rather than one blob, so each can carry its own detail
# panel underneath. cards.fixture_cards_html is exactly this concatenation, and
# a test pins that, so the markup cannot drift from the other views'.
grids = scorelines.folded_matrix(frame.xg_home.to_numpy(), frame.xg_away.to_numpy())
for index, row in enumerate(rows):
    st.html(cards.fixture_card_html(row))
    with st.expander(f"Scoreline grid · {clubs.code(row['home'])} v "
                     f"{clubs.code(row['away'])}"):
        st.html(cards.scoreline_grid_html(grids[index], row["home"], row["away"]))
        st_helpers.note(
            f"Every scoreline's probability, as a percentage. The last row and "
            f"column gather five goals or more, so the 36 cells are a complete "
            f"distribution and add to 100%. Independent Poisson at the "
            f"published rates ({row['xg_home']:.2f} and {row['xg_away']:.2f}), "
            f"so there is no score-correlation correction - Dixon-Coles was "
            f"tested against this model and returned a null.")

st_helpers.note(
    "The bar is home / draw / away, left to right. <b>pick</b> is simply the "
    "most likely of the three, which for most fixtures is a long way from a "
    "confident call, and the modal scoreline carries only ~8-12% probability - "
    "it is not a prediction."
)
