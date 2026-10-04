"""Every club's chance of each finishing band, as cards."""

from __future__ import annotations

import streamlit as st

from ucl.webapp import cards, loaders, movement, paths, st_helpers

SEASON = paths.SEASON
STAGES = ["p_top8", "p_play_off", "p_eliminated", "p_round_of_16",
          "p_quarter_finals", "p_semi_finals", "p_final", "p_champion"]
MARKET = ["decimal_odds", "p_market", "model_minus_market", "model_over_market"]
DEFAULT_SHOWN = 10

st.title("Title race")

matchday = paths.latest_matchday(SEASON)
try:
    frame = loaders.load_forecast(matchday, SEASON)
except FileNotFoundError as error:
    st_helpers.missing(error)
    st.stop()

# Deltas recomputed from the published snapshots rather than trusted from the
# file, for the reason the movement page explains.
moved = movement.movement(SEASON, matchday)
if len(moved):
    deltas = moved.set_index("team")[["d_p_top8", "d_p_champion", "d_p_eliminated"]]
    frame = frame.drop(columns=[c for c in deltas.columns if c in frame.columns])
    frame = frame.merge(deltas, left_on="team", right_index=True, how="left")

st.caption(f"Through matchday {matchday}. Sorted by P(champion).")

sort_by = st.segmented_control(
    "Order by", ["champion", "top 8", "eliminated"], default="champion",
    label_visibility="collapsed")
column = {"champion": "p_champion", "top 8": "p_top8",
          "eliminated": "p_eliminated"}[sort_by or "champion"]
ordered = frame.sort_values(column, ascending=False).reset_index(drop=True)
ordered["position"] = ordered.index + 1

show_all = st.toggle(f"Show all {len(ordered)} clubs", value=False)
shown = ordered if show_all else ordered.head(DEFAULT_SHOWN)

st.html(cards.club_cards_html(shown.to_dict("records"),
                             pin=st_helpers.pinned(SEASON)))

if not show_all:
    st_helpers.note(f"Showing the top {DEFAULT_SHOWN} of {len(ordered)}.")

# --- the table, for sorting 36 rows on a desktop -------------------------
with st.expander("Table view, with the market columns"):
    columns = [c for c in ["team", *STAGES, "mean_position", *MARKET]
               if c in frame.columns]
    st_helpers.probability_table(
        frame[columns].sort_values("p_champion", ascending=False),
        percent=STAGES,
        decimals=["mean_position", "decimal_odds", "model_over_market"],
        progress=["p_champion"], height=36 * 20)
    st_helpers.note(
        "The market column is one bookmaker's pre-season outright snapshot "
        "(maltafootball.com, 2026-08-18) de-vigged by proportional "
        "normalisation: 27 of the 36 clubs are priced, the overround is 30.8%, "
        "and the 9 unpriced clubs hold about 1% of model probability, so every "
        "market number reads roughly 1% high. Proportional de-vig also ignores "
        "favourite-longshot bias, which flatters longshots and understates "
        "favourites. <b>The market is a diagnostic and never entered any "
        "objective.</b>")
