"""What was forecast, what happened, and how the two compare."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ucl.webapp import cards, charts, loaders, paths, scoring, st_helpers

SEASON = paths.SEASON

st.title("Played matches")

scored = paths.scored_matchdays(SEASON)
if not scored:
    st.info("No matchday has been scored yet.", icon=":material/hourglass_empty:")
    st.stop()

frozen = scoring.frozen_series(SEASON)
pre = scoring.pre_matchday_series(SEASON)

summaries = {"frozen at cutoff": scoring.summarise(frozen)}
if len(pre):
    summaries["pre-matchday"] = scoring.summarise(pre)
reference = next(iter(summaries.values()))

# --- aggregates, compact ---------------------------------------------------
st_helpers.section(
    "Score against a uniform forecast",
    "lower is better; the uniform baseline is 1.0986 log-loss, 0.2593 RPS")
columns = st.columns(len(summaries), gap="small")
for column, (label, summary) in zip(columns, summaries.items()):
    with column:
        st.html(cards.card(
            f'<div style="color:#94a3b8;font:600 11px/1.3 {cards.FONT};'
            f'text-transform:uppercase;letter-spacing:0.6px">{label}</div>'
            f'<div style="display:flex;gap:16px;margin-top:6px">'
            f'<div><div style="color:#e5e7eb;font:700 20px/1.2 {cards.FONT}">'
            f'{summary["log_loss"]:.4f}</div>'
            f'<div style="color:#64748b;font:500 11px/1.3 {cards.FONT}">'
            f'log-loss</div></div>'
            f'<div><div style="color:#e5e7eb;font:700 20px/1.2 {cards.FONT}">'
            f'{summary["rps"]:.4f}</div>'
            f'<div style="color:#64748b;font:500 11px/1.3 {cards.FONT}">'
            f'RPS</div></div>'
            f'<div style="margin-left:auto;text-align:right">'
            f'{cards.delta_html(summary["log_loss"] - summary["uniform_log_loss"], points=False)}'
            f'<div style="color:#64748b;font:500 11px/1.3 {cards.FONT}">'
            f'vs uniform</div></div></div>'))

st.warning(
    f"{reference['n']} matches, standard error on the log-loss "
    f"{reference['standard_error']:.3f}. That is roughly a hundred times the "
    f"differences the five-season backtest was built to resolve: this is "
    f"anecdote and distinguishes this model from no other sensible one.",
    icon=":material/warning:")

st_helpers.note(
    "<b>Frozen at cutoff</b> is the pre-tournament forecast, fitted at "
    "2026-09-07 and scored on everything played since - no hindsight at all. "
    "<b>Pre-matchday</b> is what the model said the week before each round, "
    "with the earlier results already in the ratings. They are identical for "
    "matchday 1, both being fitted at the cutoff, which is the check that the "
    "saved tables are what they claim to be.")

# --- running chart ---------------------------------------------------------
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
    if running.matchday.nunique() > 1:
        metric = st.segmented_control("Running metric", ["log_loss", "rps"],
                                      default="log_loss",
                                      label_visibility="collapsed")
        st.altair_chart(charts.running_score(running, metric or "log_loss"),
                        use_container_width=True)
        st_helpers.note("Cumulative, so each point includes every matchday up "
                        "to it.")

# --- the candidate models against each other -------------------------------
try:
    tracked = loaders.load_tracking(SEASON)
except FileNotFoundError:
    tracked = None

if tracked is not None and len(tracked):
    from ucl import tracking

    st_helpers.section(
        "Candidate models on this season",
        "every model from the same frozen information set, the 2026-09-07 cutoff")
    board = tracking.summary(tracked)
    rows = []
    for row in board.itertuples():
        gap = getattr(row, "vs_reference", float("nan"))
        error = getattr(row, "vs_reference_standard_error", float("nan"))
        rows.append(
            f'<div style="display:flex;align-items:center;gap:8px;padding:7px 0;'
            f'border-bottom:1px solid #243041">'
            f'<span style="color:#e5e7eb;font:700 13px/1.3 {cards.FONT};'
            f'width:72px">{row.model}</span>'
            f'<span style="color:#94a3b8;font:500 12px/1.3 {cards.FONT};'
            f'flex:1 1 auto">log-loss <b style="color:#e5e7eb">'
            f'{row.log_loss:.4f}</b> · RPS <b style="color:#e5e7eb">'
            f'{row.rps:.4f}</b></span>'
            f'<span style="color:#64748b;font:500 11.5px/1.3 {cards.FONT};'
            f'text-align:right;white-space:nowrap">'
            + (f"{gap:+.4f} ± {error:.4f} vs {tracking.REFERENCE}"
               if gap == gap else "published")
            + f'</span></div>')
    st.html("".join(rows))

    if tracked.matchday.nunique() > 1:
        running = tracking.running(tracked)
        metric = st.segmented_control(
            "Tracking metric", ["log_loss", "rps"], default="log_loss",
            label_visibility="collapsed", key="tracking_metric")
        chart = running.rename(columns={
            f"cumulative_{metric or 'log_loss'}": "value"})[
            ["matchday", "model", "value"]]
        st.line_chart(chart, x="matchday", y="value", color="model", height=300)

    paired_error = board.loc[board.model == "v1", "vs_reference_standard_error"]
    error = float(paired_error.iloc[0]) if len(paired_error) else float("nan")
    st_helpers.note(
        f"<b>Monitoring, not a gate.</b> The paired difference between v1 and "
        f"v2 carries a standard error of ±{error:.4f} on "
        f"{int(board.n.iloc[0])} matches, and over all 144 league-phase matches "
        f"it will still be about <b>three times</b> the 0.0004 the five-season "
        f"backtest resolved. This season cannot choose between them. The "
        f"decision rests on the backtest, which accepted the shot ratings with "
        f"all four folds negative; this table would catch something badly "
        f"wrong, and nothing else.")

# --- per match -------------------------------------------------------------
st_helpers.section("Fixtures")
which = st.segmented_control("Series", list(summaries), default=list(summaries)[0],
                             label_visibility="collapsed")
frame = frozen if (which or list(summaries)[0]) == "frozen at cutoff" else pre
if len(scored) > 1:
    chosen = st.segmented_control("Matchday", ["all", *scored], default="all",
                                  label_visibility="collapsed")
    if chosen and chosen != "all":
        frame = frame[frame.matchday == chosen]

display = frame.copy()
if "date" in display.columns:
    display["date"] = pd.to_datetime(display.date).dt.strftime("%d %b")
st.html(cards.fixture_cards_html(display.to_dict("records"), show_result=True))
