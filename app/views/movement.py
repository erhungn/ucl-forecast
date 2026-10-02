"""How the forecast has moved as results came in."""

from __future__ import annotations

import streamlit as st

from ucl.webapp import loaders, movement, paths, st_helpers

SEASON = paths.SEASON
METRICS = {"P(champion)": "p_champion", "P(top 8)": "p_top8",
           "P(eliminated)": "p_eliminated", "Mean position": "mean_position"}

st.title("Movement")

matchdays = paths.available_matchdays(SEASON)
if len(matchdays) < 2:
    st.info(
        f"Only matchday {matchdays[0] if matchdays else 0} is published, so "
        f"there is nothing to compare yet. The chart starts once a second "
        f"matchday has been forecast.",
        icon=":material/hourglass_empty:")

try:
    series = movement.forecast_series(SEASON)
except FileNotFoundError as error:
    st_helpers.missing(error)
    st.stop()

moved = movement.movement(SEASON)

# --- the chart ------------------------------------------------------------
latest = series[series.matchday == series.matchday.max()]
default = latest.nlargest(8, "p_champion").team.tolist()
chosen = st.multiselect("Clubs", sorted(series.team.unique()), default=default)
label = st.radio("Metric", list(METRICS), horizontal=True)
column = METRICS[label]

if chosen:
    st.line_chart(series[series.team.isin(chosen)], x="matchday", y=column,
                  color="team", height=420)
    st.caption(f"{label} by matchday. Matchday 0 is the pre-tournament forecast.")

# --- biggest movers -------------------------------------------------------
if len(moved):
    previous = moved.attrs.get("previous_matchday")
    current = moved.attrs.get("matchday")
    st.subheader(f"Biggest movers, matchday {previous} to {current}")
    metric = st.selectbox("Rank by", ["d_p_top8", "d_p_champion",
                                      "d_p_eliminated"], index=0)
    table = movement.biggest_movers(moved, metric, count=12)
    columns = ["team", "p_top8_prev", "p_top8", "d_p_top8",
               "p_champion_prev", "p_champion", "d_p_champion",
               "p_eliminated_prev", "p_eliminated", "d_p_eliminated"]
    st_helpers.probability_table(
        table[[c for c in columns if c in table.columns]],
        percent=[c for c in columns if c != "team"])

    # Integrity: simulate.py computes its own d_* against outputs/, which is
    # scratch. Silence here means the published chain agrees with it.
    disagreement = movement.chain_disagreement(moved)
    if len(disagreement):
        st.error(
            f"{len(disagreement)} stored delta(s) disagree with the published "
            f"snapshots, so the forecast's own d_* columns were computed "
            f"against something other than live/. The table above is "
            f"recomputed from live/ and is the one to trust.")
        st.dataframe(disagreement, hide_index=True)
    else:
        st.caption(
            "Deltas recomputed from the published snapshots; they agree with "
            "the ones the simulator wrote.")
