"""Readers for the published artefacts.

A missing file is a normal state - the weekly update may not have run yet - so
these raise :class:`FileNotFoundError` naming the path and the script that
would create it, and the pages turn that into one line of explanation rather
than a traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ucl.webapp import paths

__all__ = [
    "load_forecast", "load_next_matchday", "load_evaluation",
    "load_information", "load_tracking", "load_status", "tail_log",
]

_HINT = "run scripts/weekly_update.py to publish it"


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; {_HINT}")
    return pd.read_csv(path, **kwargs)


def load_forecast(matchday: int, season: str = paths.SEASON, **kwargs) -> pd.DataFrame:
    """One matchday's stage probabilities.

    Matchday 0 falls back to the tracked archive, so the app has something to
    show before the first weekly run.
    """
    path = paths.forecast_path(matchday, season, **kwargs)
    if not path.exists() and matchday == 0:
        archived = paths.ARCHIVE_DIR / f"forecast_{season}.csv"
        if archived.exists():
            return pd.read_csv(archived)
    return _read_csv(path)


def load_next_matchday(matchday: int, season: str = paths.SEASON,
                       **kwargs) -> pd.DataFrame:
    """The frozen pre-match 1X2 table for ``matchday``."""
    return _read_csv(paths.next_matchday_path(matchday, season, **kwargs),
                     parse_dates=["date"])


def load_evaluation(matchday: int, season: str = paths.SEASON,
                    **kwargs) -> pd.DataFrame:
    """One matchday scored against the forecast frozen at the cutoff."""
    return _read_csv(paths.evaluation_path(matchday, season, **kwargs),
                     parse_dates=["date"])


def load_information(season: str = paths.SEASON, **kwargs) -> pd.DataFrame:
    """The information set as it stood when the forecast was produced."""
    return _read_csv(paths.information_path(season, **kwargs))


def load_tracking(season: str = paths.SEASON, **kwargs) -> pd.DataFrame:
    """Every candidate model's probabilities on the matches played so far."""
    return _read_csv(paths.tracking_path(season, **kwargs))


def load_status(season: str = paths.SEASON, **kwargs) -> dict:
    """The last run's status, or an empty dict if the update has never run."""
    path = paths.status_path(season, **kwargs)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def tail_log(season: str = paths.SEASON, *, lines: int = 400, **kwargs) -> str:
    """The tail of the last run's log, for the app to display while it runs."""
    path = paths.log_path(season, **kwargs)
    if not path.exists():
        return ""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return "".join(handle.readlines()[-lines:])
