"""What each source knew when the forecast was made - and the update button."""

from __future__ import annotations

import streamlit as st

from ucl.webapp import loaders, paths, st_helpers, staleness

SEASON = paths.SEASON

st.title("Information set")

status = loaders.load_status(SEASON)
try:
    information = loaders.load_information(SEASON)
except FileNotFoundError as error:
    st_helpers.missing(error)
    information = None

if information is not None:
    st_helpers.freshness_badge(status, information)
    st.caption(
        "`cached_to` is the last match in the cache; `used_to` is the last one "
        "the forecast actually used. They differ whenever the cache runs past "
        "the information set, which is exactly what a forecast must not "
        "misrepresent."
    )
    st.dataframe(information, hide_index=True)

    for line in staleness.warnings(information, status):
        st.warning(line, icon=":material/schedule:")
    for line in staleness.notes(information):
        st.info(line, icon=":material/lock:")

# --- provenance -----------------------------------------------------------
if status:
    st_helpers.section("This run")
    left, right = st.columns(2)
    left.markdown(
        f"- through matchday **{status.get('through_matchday')}**\n"
        f"- as of **{status.get('as_of')}**\n"
        f"- {status.get('played_matches')} played matches\n"
        f"- next matchday **{status.get('next_matchday')}** on "
        f"{status.get('next_matchday_first_date')}\n"
        f"- state **{status.get('state')}**"
    )
    right.markdown(
        f"- model **{status.get('model')}**, {status.get('iterations'):,} "
        f"iterations, seed {status.get('seed')}\n"
        f"- verification "
        f"**{'passed' if status.get('verification_passed') else 'FAILED'}**\n"
        f"- commit `{str(status.get('git_commit', ''))[:12]}`\n"
        f"- inputs `{str(status.get('inputs_hash', ''))[7:19]}`\n"
        f"- took {status.get('duration_seconds')}s"
    )
    if status.get("pruned"):
        st.error("Files deleted from the raw cache by the last run: "
                 + ", ".join(status["pruned"]))

# --- the update -----------------------------------------------------------
st_helpers.section("Weekly update")
st.caption(
    "Refreshes the league-phase results, re-forecasts through the last "
    "complete matchday, scores what was played, and publishes everything to "
    "`live/`. It refuses mid-round, because a half-played matchday would leave "
    "the next-matchday table half stale."
)
st_helpers.update_controls(SEASON)
