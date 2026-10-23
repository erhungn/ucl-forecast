"""How the forecast has moved as results came in."""

from __future__ import annotations

import streamlit as st

from ucl.webapp import cards, charts, loaders, movement, paths, st_helpers

SEASON = paths.SEASON
METRICS = {"P(champion)": "p_champion", "P(top 8)": "p_top8",
           "P(out)": "p_eliminated", "Mean position": "mean_position"}

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
latest = series[series.matchday == series.matchday.max()]

label = st.segmented_control("Metric", list(METRICS), default="P(champion)",
                             label_visibility="collapsed")
column = METRICS[label or "P(champion)"]

default = latest.nlargest(charts.MAX_SERIES, "p_champion").team.tolist()
chosen = st.multiselect(
    f"Clubs (up to {charts.MAX_SERIES})", sorted(series.team.unique()),
    default=default, max_selections=charts.MAX_SERIES,
    help="Beyond eight lines a palette stops telling them apart. Every other "
         "club is still drawn, faintly, for context.")

panels = st.toggle("Small multiples", value=False,
                   help="One panel per club - easier to read than eight lines "
                        "in one frame, but cramped on a phone.")

if chosen:
    if panels:
        st.altair_chart(charts.small_multiples(series, column, chosen),
                        width="stretch")
    else:
        st.altair_chart(charts.trend(series, column, chosen),
                        width="stretch")
    st_helpers.note(
        f"{label} by matchday. Matchday 0 is the pre-tournament forecast; the "
        f"faint lines are the clubs not selected.")
    backfilled = loaders.retroactive_matchdays(SEASON)
    if backfilled:
        st.warning(
            "Not every point on this line is a forecast that was made before "
            "its matchday: "
            + "; ".join(f"matchday **{md}** was re-forecast by {model} after "
                        f"the round was played"
                        for md, model in sorted(backfilled.items()))
            + ". A retroactive snapshot fills a gap left by a model switch. It "
              "is not something anyone could have acted on, and the movement "
              "into and out of it is not news arriving.",
            icon=":material/history:")
else:
    st.info("Pick at least one club.", icon=":material/touch_app:")

# --- biggest movers -------------------------------------------------------
if len(moved):
    previous = moved.attrs.get("previous_matchday")
    current = moved.attrs.get("matchday")
    st_helpers.section(f"Biggest movers, matchday {previous} to {current}")
    metric = st.segmented_control(
        "Rank by", ["top 8", "champion", "out"], default="top 8",
        label_visibility="collapsed")
    key = {"top 8": "d_p_top8", "champion": "d_p_champion",
           "out": "d_p_eliminated"}[metric or "top 8"]
    base = key[2:]
    ranked = movement.biggest_movers(moved, key, count=12)
    st.html(cards.mover_rows_html([
        {"team": row.team, "label": metric,
         "before": getattr(row, f"{base}_prev"), "after": getattr(row, base),
         "delta": getattr(row, key)}
        for row in ranked.itertuples()
    ]))

    # Integrity: simulate.py computes its own d_* against outputs/, which is
    # scratch. Silence here means the published chain agrees with it.
    disagreement = movement.chain_disagreement(moved)
    if len(disagreement):
        st.error(
            f"{len(disagreement)} stored delta(s) disagree with the published "
            f"snapshots, so the forecast's own d_* columns were computed "
            f"against something other than live/. The rows above are "
            f"recomputed from live/ and are the ones to trust.")
        st.dataframe(disagreement, hide_index=True)
    else:
        st_helpers.note(
            "Deltas recomputed from the published snapshots; they agree with "
            "the ones the simulator wrote.")


# --- what earlier models published ----------------------------------------
# A model switch moves the previous model's snapshots into
# live/<season>/archive_<model>/ rather than deleting them. They are read back
# here, as that model's own series, and never joined to the current one: the
# step from one model to another is not movement in any club's chances.
superseded = paths.history_models(SEASON)
if superseded:
    st_helpers.section("Published by earlier models")
    st_helpers.note(movement.history_note(SEASON))
    for model in superseded:
        history = movement.history_series(model, SEASON)
        if not len(history):
            continue
        days = sorted(history.matchday.unique())
        st.caption(
            f"**{model}** - matchday(s) {', '.join(str(d) for d in days)}, as "
            f"published at the time. Kept as the record of what this model "
            f"said; not comparable point-by-point with the lines above, which "
            f"come from a different model.")
        shown = [club for club in chosen if club in set(history.team)]
        if shown and len(days) > 1:
            st.altair_chart(charts.trend(history, column, shown), width="stretch")
        else:
            latest_history = history[history.matchday == max(days)]
            st.dataframe(
                latest_history[["matchday", "team", column]]
                .sort_values(column, ascending=False).head(12),
                hide_index=True, width="stretch")
