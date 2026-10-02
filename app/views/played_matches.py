"""What was forecast, what happened, and how the two compare."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ucl.webapp import paths, scoring, st_helpers

SEASON = paths.SEASON

st.title("Played matches")

scored = paths.scored_matchdays(SEASON)
if not scored:
    st.info("No matchday has been scored yet.", icon=":material/hourglass_empty:")
    st.stop()

frozen = scoring.frozen_series(SEASON)
pre = scoring.pre_matchday_series(SEASON)

st.caption(
    "Two series, because they answer different questions. **Frozen at cutoff** "
    "is the pre-tournament forecast, fitted at 2026-09-07 and scored on "
    "everything played since - no hindsight at all. **Pre-matchday** is what "
    "the model said the week before each round, with the earlier results "
    "already in the ratings. They are identical for matchday 1, both being "
    "fitted at the cutoff, which is the check that the saved tables are what "
    "they claim to be."
)

# --- aggregates -----------------------------------------------------------
summaries = {"frozen at cutoff": scoring.summarise(frozen)}
if len(pre):
    summaries["pre-matchday"] = scoring.summarise(pre)

columns = st.columns(len(summaries) + 1)
for column, (label, summary) in zip(columns, summaries.items()):
    column.metric(f"{label}: log-loss", f"{summary['log_loss']:.4f}",
                  f"{summary['log_loss'] - summary['uniform_log_loss']:+.4f} vs uniform",
                  delta_color="inverse")
    column.metric(f"{label}: RPS", f"{summary['rps']:.4f}",
                  f"{summary['rps'] - summary['uniform_rps']:+.4f} vs uniform",
                  delta_color="inverse")
reference = next(iter(summaries.values()))
columns[-1].metric("Uniform baseline", f"{reference['uniform_log_loss']:.4f}",
                   f"RPS {reference['uniform_rps']:.4f}", delta_color="off")

st.warning(
    f"{reference['n']} matches, standard error on the log-loss "
    f"{reference['standard_error']:.3f}. That is roughly a hundred times the "
    f"differences the five-season backtest was built to resolve: this is "
    f"anecdote and distinguishes this model from no other sensible one.",
    icon=":material/warning:")

# --- cumulative chart -----------------------------------------------------
pieces = []
for label, frame in (("frozen at cutoff", frozen), ("pre-matchday", pre)):
    if not len(frame):
        continue
    running = scoring.cumulative(frame)
    pieces.append(pd.DataFrame({
        "matchday": running.matchday, "series": label,
        "log_loss": running.cumulative_log_loss, "rps": running.cumulative_rps}))
    pieces.append(pd.DataFrame({
        "matchday": running.matchday, "series": "uniform",
        "log_loss": running.uniform_log_loss, "rps": running.uniform_rps}))
if pieces:
    running = pd.concat(pieces, ignore_index=True).drop_duplicates(
        subset=["matchday", "series"])
    metric = st.radio("Running metric", ["log_loss", "rps"], horizontal=True)
    st.line_chart(running, x="matchday", y=metric, color="series", height=320)
    st.caption("Cumulative, so each point includes every matchday up to it. "
               "Lower is better.")

# --- per match ------------------------------------------------------------
st.subheader("Fixtures")
which = st.radio("Series", list(summaries), horizontal=True)
frame = frozen if which == "frozen at cutoff" else pre
chosen = st.selectbox("Matchday", ["all", *scored], index=0)
if chosen != "all":
    frame = frame[frame.matchday == chosen]

display = frame.copy()
if "date" in display.columns:
    display["date"] = pd.to_datetime(display.date).dt.date
columns = ["matchday", "date", "home", "away", "score", "p_home", "p_draw",
           "p_away", "result", "p_actual", "log_loss", "rps"]
st_helpers.probability_table(
    display[[c for c in columns if c in display.columns]],
    percent=["p_home", "p_draw", "p_away", "p_actual"],
    decimals=["log_loss", "rps"], height=36 * min(len(display) + 1, 20))
