"""The only module that imports Streamlit.

Keeping it in one place means the test suite never needs Streamlit installed,
and the logic in the sibling modules stays callable from a plain interpreter.

The cards are drawn with ``st.html`` and inline styles. That is not a stylistic
preference: checked on a phone, ``st.markdown(unsafe_allow_html=True)`` lost the
labels inside the bars, while ``st.html`` rendered exactly as written. A whole
list of cards goes out in one ``st.html`` call, which keeps the spacing even and
the element count down.

This module is also the only thing that starts anything, and it writes nothing
itself: it launches ``scripts/weekly_update.py`` detached and then reads the log
that script owns. A Streamlit rerun - any widget click, a phone dropping the
websocket - raises inside the script thread, so a loop streaming a subprocess
here would lose the tail of a five-minute run. Tailing a file does not care.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from ucl import config
from ucl.webapp import freshness, loaders, paths, staleness
from ucl.webapp.cards import FONT, card
from ucl.webapp.palette import FAINT, MUTED, TEXT

__all__ = [
    "SEASON", "update_argv", "is_running", "start_update", "sidebar",
    "section", "freshness_badge", "note", "missing", "probability_table",
    "log_pane", "update_controls",
]

SEASON = paths.SEASON
MODEL_PYTHON = config.PROJECT_ROOT / ".venv" / "bin" / "python"
UPDATE_SCRIPT = config.PROJECT_ROOT / "scripts" / "weekly_update.py"


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


def section(title: str, caption: str = "") -> None:
    """A compact section heading - tighter than st.subheader on a phone."""
    body = (f'<div style="color:{TEXT};font:700 14px/1.3 {FONT};'
            f'letter-spacing:0.2px;margin:2px 0 6px 0">{title}</div>')
    if caption:
        body += (f'<div style="color:{FAINT};font:500 11.5px/1.4 {FONT};'
                 f'margin:-4px 0 8px 0">{caption}</div>')
    st.html(body)


def freshness_badge(status: dict, information: pd.DataFrame | None = None) -> str:
    """Render the freshness badge and return its level."""
    name, label, _ = freshness.level(status, information)
    st.badge(label, color=freshness.LEVELS[name][1],
             icon={"fresh": ":material/check_circle:",
                   "ageing": ":material/schedule:",
                   "stale": ":material/error:"}[name])
    return name


def note(text: str) -> None:
    """A quiet caption, for the caveats that must stay attached to a number."""
    st.html(f'<div style="color:{FAINT};font:500 11.5px/1.5 {FONT};'
            f'margin:4px 0 10px 0">{text}</div>')


def sidebar(season: str = SEASON) -> dict:
    """The run banner and any staleness warnings, on every page."""
    status = loaders.load_status(season)
    try:
        information = loaders.load_information(season)
    except FileNotFoundError:
        information = None

    with st.sidebar:
        st.markdown(f"### {season}")
        if not status:
            st.warning("No update has run yet; showing the frozen archive.")
            return status
        freshness_badge(status, information)
        st.markdown(
            f"**through matchday {status.get('through_matchday', 0)}**  \n"
            f"as of {status.get('as_of', '?')}  \n"
            f"{status.get('iterations', 0):,} iterations, model "
            f"{status.get('model', '?')}"
        )
        if is_running(status):
            st.info("An update is running.", icon=":material/sync:")
        elif status.get("state") == "refused":
            st.error(f"Last run refused: {status.get('message', '')}")
        elif status.get("state") == "failed":
            st.error(f"Last run failed: {status.get('message', '')}")
        if information is not None:
            for line in staleness.warnings(information, status):
                st.warning(line, icon=":material/schedule:")
    return status


def probability_table(frame: pd.DataFrame, *, percent=(), decimals=(),
                      progress=(), height: int | None = None) -> None:
    """The table view, kept for sorting 36 rows on a desktop.

    ``st.dataframe`` sorts on a header click, which is why this is still worth
    having beside the cards - but it scrolls sideways on a phone, so it is
    never the default.
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
                format="%.3f", min_value=0.0,
                max_value=float(max(frame[column].max(), 1e-9)))
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
            note("The update button is behind a token because this app is "
                 "served without authentication on your network.")
            return
    else:
        note("No <code>update_token</code> in .streamlit/secrets.toml: anyone "
             "who can open this page can start an update.")

    prune = st.checkbox(
        "Allow pruning superseded football-data files",
        value=False,
        help="A weekly domestic refresh lands files whose names collide with "
             "the main cache, and the loader refuses until the old copies go. "
             "Ticking this lets the update delete them permanently.")
    force = st.checkbox("Re-run even if the inputs are unchanged", value=False)

    if st.button("Run the weekly update", disabled=running, type="primary",
                 width="stretch"):
        start_update(season, prune_stale=prune, force=force)
        st.rerun()
    if running:
        st.info(
            f"Running since {str(status.get('started_at_utc', ''))[:19]} "
            f"(pid {status.get('pid')}). A 50,000-iteration forecast takes "
            f"about five minutes; this log refreshes by itself.")
    log_pane(season)
