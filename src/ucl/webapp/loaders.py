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
    "load_positions", "load_knockout_paths", "load_ratings",
    "load_remaining_fixtures", "load_club_inputs",
    "load_history_forecast", "retroactive_matchdays",
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


def load_evaluation(matchday: int, season: str = paths.SEASON, *,
                    model: str | None = None, **kwargs) -> pd.DataFrame:
    """One matchday scored against the forecast frozen at the cutoff.

    ``model=None`` is the published record - scored by whichever model was
    published when the matchday was played. A named model is that model's score
    for the same matches, written beside the record rather than over it.
    """
    return _read_csv(paths.evaluation_path(matchday, season, model=model, **kwargs),
                     parse_dates=["date"])


def load_history_forecast(model: str, matchday: int,
                          season: str = paths.SEASON, **kwargs) -> pd.DataFrame:
    """A snapshot a superseded model published, kept as its record."""
    return _read_csv(paths.history_forecast_path(model, matchday, season, **kwargs))


def retroactive_matchdays(season: str = paths.SEASON, **kwargs) -> dict[int, str]:
    """Published matchdays whose forecast was produced after the fact.

    ``{matchday: model}``. A backfilled snapshot - a new model re-forecasting a
    round that has already been played - is not something anyone could have
    acted on, so it carries a ``retroactive`` column and every view that shows
    a series says which points are of that kind. Snapshots without the column
    are forecasts made before the round, which is the normal case.
    """
    out: dict[int, str] = {}
    for matchday in paths.available_matchdays(season, **kwargs):
        path = paths.forecast_path(matchday, season, **kwargs)
        if not path.exists():
            continue
        head = pd.read_csv(path, nrows=1)
        if "retroactive" in head.columns and bool(head.retroactive.iloc[0]):
            out[matchday] = str(head.get("retroactive_model",
                                         pd.Series(["?"])).iloc[0])
    return out


def load_information(season: str = paths.SEASON, **kwargs) -> pd.DataFrame:
    """The information set as it stood when the forecast was produced."""
    return _read_csv(paths.information_path(season, **kwargs))


def load_tracking(season: str = paths.SEASON, **kwargs) -> pd.DataFrame:
    """Every candidate model's probabilities on the matches played so far."""
    return _read_csv(paths.tracking_path(season, **kwargs))


def load_positions(matchday: int, season: str = paths.SEASON,
                   **kwargs) -> pd.DataFrame:
    """P(finishing position) per club, as ``team, position, count``.

    Counts, not probabilities, so the invariants are exact: every club's counts
    sum to the iteration count, and so does every position's. A ``probability``
    column is added here by dividing by that total, which is read off the data
    rather than taken from the status file.
    """
    frame = _read_csv(paths.positions_path(matchday, season, **kwargs))
    iterations = int(frame.groupby("team")["count"].sum().iloc[0])
    frame["probability"] = frame["count"] / iterations
    frame.attrs["iterations"] = iterations
    return frame


def load_knockout_paths(matchday: int, season: str = paths.SEASON,
                        **kwargs) -> pd.DataFrame:
    """Who met whom per knockout round, as ``round, club, opponent, count``.

    Both directions of every tie are stored, so one club's rows are its own
    opponent distribution and sum to its probability of playing in that round.
    """
    frame = _read_csv(paths.knockout_paths_path(matchday, season, **kwargs))
    return frame


def load_ratings(matchday: int, season: str = paths.SEASON,
                 **kwargs) -> pd.DataFrame:
    """Elo as the forecast started from it - after this matchday was played."""
    return _read_csv(paths.ratings_path(matchday, season, **kwargs))


def load_remaining_fixtures(matchday: int, season: str = paths.SEASON,
                            **kwargs) -> pd.DataFrame:
    """Every unplayed fixture's 1X2, at the ratings of the matchday named.

    Beyond the next matchday this is the model evaluated at today's ratings,
    not the simulator's marginal: it ignores the results in between. Any view
    showing it must say so next to the number.
    """
    return _read_csv(paths.remaining_fixtures_path(matchday, season, **kwargs),
                     parse_dates=["date"])


def load_club_inputs(season: str = paths.SEASON, **kwargs) -> pd.DataFrame:
    """The frozen pre-cutoff covariates per club, with coverage flags."""
    return _read_csv(paths.club_inputs_path(season, **kwargs))


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
