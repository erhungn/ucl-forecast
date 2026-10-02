"""Every club's chance of each finishing band, beside the market."""

from __future__ import annotations

import streamlit as st

from ucl.webapp import loaders, paths, st_helpers

SEASON = paths.SEASON
STAGES = ["p_top8", "p_play_off", "p_eliminated", "p_round_of_16",
          "p_quarter_finals", "p_semi_finals", "p_final", "p_champion"]
MARKET = ["decimal_odds", "p_market", "model_minus_market", "model_over_market"]

st.title("Title race")

matchday = paths.latest_matchday(SEASON)
try:
    frame = loaders.load_forecast(matchday, SEASON)
except FileNotFoundError as error:
    st_helpers.missing(error)
    st.stop()

st.caption(
    f"Forecast through matchday {matchday}. Bands: 1-8 straight to the round of "
    f"16, 9-24 into the knockout play-off, 25-36 out of Europe altogether."
)

leader = frame.loc[frame.p_champion.idxmax()]
top8 = frame.loc[frame.p_top8.idxmax()]
priced = frame[frame.p_market.notna()] if "p_market" in frame else frame.iloc[:0]
left, middle, right = st.columns(3)
left.metric("Most likely champion", leader.team, f"{leader.p_champion:.1%}")
middle.metric("Most likely top 8", top8.team, f"{top8.p_top8:.1%}")
if len(priced):
    widest = priced.loc[priced.model_minus_market.abs().idxmax()]
    right.metric("Largest market disagreement", widest.team,
                 f"{widest.model_minus_market:+.1%}")

show_market = st.toggle("Show the market columns", value=True)
columns = ["team", *STAGES, "mean_position"] + (MARKET if show_market else [])
columns = [c for c in columns if c in frame.columns]

st_helpers.probability_table(
    frame[columns], percent=STAGES, decimals=["mean_position", "decimal_odds",
                                              "model_over_market"],
    progress=["p_champion"], height=36 * 36)

st.caption(
    "Click a column header to sort. The market column is one bookmaker's "
    "pre-season outright snapshot (maltafootball.com, 2026-08-18) de-vigged by "
    "proportional normalisation: 27 of the 36 clubs are priced, the overround "
    "is 30.8%, and the 9 unpriced clubs hold about 1% of model probability, so "
    "every market number reads roughly 1% high. Proportional de-vig also "
    "ignores favourite-longshot bias, which flatters longshots and understates "
    "favourites. **The market is a diagnostic and never entered any objective.**"
)
