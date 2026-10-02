"""The only module that imports Streamlit.

Keeping it in one place means the test suite never needs Streamlit installed,
and the logic in the sibling modules stays callable from a plain interpreter.

This module is also the only writer of anything, and it writes nothing itself:
it starts ``scripts/weekly_update.py`` as a detached child and then reads the
log that script owns. A Streamlit rerun - any widget click, a phone dropping
the websocket - raises inside the script thread, so a loop that streamed the
subprocess here would lose the tail of a five-minute run. Tailing a file does
not care.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from ucl import config
from ucl.webapp import loaders, paths, staleness

__all__ = [
    "SEASON", "update_argv", "is_running", "start_update", "sidebar",
    "probability_table", "log_pane", "update_controls", "missing",
]

SEASON = paths.SEASON
MODEL_PYTHON = config.PROJECT_ROOT / ".venv" / "bin" / "python"
UPDATE_SCRIPT = config.PROJECT_ROOT / "scripts" / "weekly_update.py"

PERCENT = {"format": "%.1f%%"}


def update_argv(season: str = SEASON, *, prune_stale: bool = False,
                force: bool = False, iterations: int | None = None) -> list[str]:
    """The command the button runs."""
    python = MODEL_PYTHON if MODEL_PYTHON.exists() else Path("python3")
    argv = [str(python), str(UPDATE_SCRIPT), "--season", season]
    if prune_stale:
        argv.append("--prune-stale")
    if force:
        argv.append("--force")
    if iterations:
        argv += ["-n", str(iterations)]
    return argv


def is_running(status: dict) -> bool:
    """True while a run is in flight, checked against the recorded pid.

    ``st.session_state`` is per browser tab, so it cannot stop a phone and a
    laptop starting two simulations; the status file can.
    """
    if status.get("state") != "running":
        return False
    pid = status.get("pid")
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def start_update(season: str = SEASON, **kwargs) -> None:
    """Launch the update in its own session, so it outlives this page."""
    subprocess.Popen(
        update_argv(season, **kwargs), cwd=config.PROJECT_ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)


def missing(error: FileNotFoundError) -> None:
    """Explain an absent artefact instead of showing a traceback."""
    st.info(f"{error}", icon=":material/hourglass_empty:")


def sidebar(season: str = SEASON) -> dict:
    """The run banner and any staleness warnings, on every page."""
    status = loaders.load_status(season)
    with st.sidebar:
        st.markdown(f"### {season}")
        if not status:
            st.warning("No update has run yet; showing the frozen archive.")
            return status
        matchday = status.get("through_matchday", 0)
        st.markdown(
            f"**through matchday {matchday}**  \n"
            f"as of {status.get('as_of', '?')}  \n"
            f"{status.get('iterations', '?'):,} iterations, model "
            f"{status.get('model', '?')}  \n"
            f"run {str(status.get('finished_at_utc', '?'))[:16]}"
        )
        if is_running(status):
            st.info("An update is running.", icon=":material/sync:")
        elif status.get("state") == "refused":
            st.error(f"Last run refused: {status.get('message', '')}")
        elif status.get("state") == "failed":
            st.error(f"Last run failed: {status.get('message', '')}")

        try:
            information = loaders.load_information(season)
        except FileNotFoundError:
            return status
        for line in staleness.warnings(information, status):
            st.warning(line, icon=":material/schedule:")
    return status


def probability_table(frame: pd.DataFrame, *, percent=(), decimals=(),
                      progress=(), height: int | None = None) -> None:
    """One place where number formatting is decided.

    ``st.dataframe`` sorts on a header click, which is what makes every table
    in the app sortable without any code of its own.
    """
    configuration = {}
    for column in percent:
        if column in frame.columns:
            configuration[column] = st.column_config.NumberColumn(format="%.3f")
    for column in decimals:
        if column in frame.columns:
            configuration[column] = st.column_config.NumberColumn(format="%.2f")
    for column in progress:
        if column in frame.columns:
            configuration[column] = st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=float(
                    max(frame[column].max(), 1e-9)))
    # Streamlit rejects height=None, so only pass one when asked for.
    extra = {"height": height} if height else {}
    st.dataframe(frame, hide_index=True, column_config=configuration, **extra)


@st.fragment(run_every=2)
def log_pane(season: str = SEASON) -> None:
    """Tail the update log, refreshing only this fragment every two seconds."""
    text = loaders.tail_log(season, lines=200)
    st.code(text or "(no log yet)", language="text")


def update_controls(season: str = SEASON) -> None:
    """The one control in the app that does anything.

    Gated on a token when one is configured, because the app is served
    unauthenticated on the local network: the read-only pages give nothing
    away, but this button starts a five-minute job and, with pruning allowed,
    deletes cached raw data.
    """
    status = loaders.load_status(season)
    running = is_running(status)

    token = None
    try:
        token = st.secrets.get("update_token")
    except Exception:                               # noqa: BLE001 - no secrets file
        token = None
    if token:
        if not st.session_state.get("unlocked"):
            supplied = st.text_input("Update token", type="password")
            if supplied and supplied == token:
                st.session_state["unlocked"] = True
                st.rerun()
            elif supplied:
                st.error("Wrong token.")
            st.caption(
                "The update button is behind a token because this app is "
                "served without authentication on your network.")
            return
    else:
        st.caption(
            "No `update_token` in .streamlit/secrets.toml: anyone who can open "
            "this page can start an update.")

    prune = st.checkbox(
        "Allow pruning superseded football-data files",
        value=False,
        help="A weekly domestic refresh lands files whose names collide with "
             "the main cache, and the loader refuses until the old copies go. "
             "Ticking this lets the update delete them permanently.")
    force = st.checkbox("Re-run even if the inputs are unchanged", value=False)

    if st.button("Run the weekly update", disabled=running, type="primary"):
        start_update(season, prune_stale=prune, force=force)
        st.rerun()
    if running:
        st.info(
            f"Running since {str(status.get('started_at_utc', ''))[:19]} "
            f"(pid {status.get('pid')}). A 50,000-iteration forecast takes "
            f"about five minutes; this log refreshes by itself.")
    log_pane(season)
