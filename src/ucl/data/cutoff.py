"""The no-look-ahead guard.

Every loader filters its rows to strictly before a cutoff and then asserts it.
The cutoff is always an explicit argument: see :class:`ucl.config.SeasonConfig`.

In-season the cutoff generalises to an as-of date, derived from the matchday
the forecast runs through; :func:`as_of_for_matchday` is the one definition of
that rule, shared by the simulator and the weekly update.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

__all__ = [
    "LookAheadError", "assert_no_lookahead", "filter_to_cutoff",
    "as_of_for_matchday",
]


class LookAheadError(AssertionError):
    """Raised when a frame used for fitting contains a row at or after the cutoff."""


def _as_dates(frame: pd.DataFrame, date_column: str, source: str) -> pd.Series:
    if date_column not in frame.columns:
        raise KeyError(
            f"{source or 'frame'}: no column {date_column!r}; have {list(frame.columns)}"
        )
    return pd.to_datetime(frame[date_column], errors="coerce")


def assert_no_lookahead(
    frame: pd.DataFrame,
    cutoff: date,
    *,
    date_column: str = "date",
    source: str = "",
) -> None:
    """Raise :class:`LookAheadError` if any row is dated on or after ``cutoff``.

    The cutoff day itself is excluded: matchday 1 is played on it, so a row
    dated exactly on the cutoff is already look-ahead. Unparseable dates are
    rejected too — a NaT cannot be shown to be in the past.
    """
    label = source or "frame"
    dates = _as_dates(frame, date_column, label)

    missing = int(dates.isna().sum())
    if missing:
        raise LookAheadError(
            f"{label}: {missing} row(s) have an unparseable {date_column!r}; "
            "cannot prove they predate the cutoff"
        )

    offending = dates[dates >= pd.Timestamp(cutoff)]
    if len(offending):
        worst = sorted(offending.dt.date.unique())
        shown = ", ".join(str(d) for d in worst[:5])
        more = f" (+{len(worst) - 5} more dates)" if len(worst) > 5 else ""
        raise LookAheadError(
            f"{label}: {len(offending)} row(s) dated on or after the cutoff "
            f"{cutoff}: {shown}{more}"
        )


def as_of_for_matchday(fixtures: pd.DataFrame, matchday: int, cutoff: date) -> date:
    """The information-set date for a forecast run through ``matchday``.

    Everything decided before the next matchday is in the information set, so
    the as-of date is the day before that matchday's first fixture. Through
    matchday 0 it is the season's own cutoff.

    For the last matchday there is no next one, and the fallback is the day
    before the final round's own last fixture - which is *inside* that round.
    That is a wart rather than a choice: by then every match is played and
    fixed, so the ratings the as-of date selects no longer change the forecast.
    """
    if matchday <= 0:
        return cutoff
    later = fixtures[fixtures.matchday == matchday + 1]
    first = later.date.min() if len(later) else fixtures.date.max()
    return (pd.Timestamp(first) - pd.Timedelta(1, "D")).date()


def filter_to_cutoff(
    frame: pd.DataFrame,
    cutoff: date,
    *,
    date_column: str = "date",
    source: str = "",
) -> pd.DataFrame:
    """Drop rows dated on or after ``cutoff``, then assert the result is clean.

    Rows with an unparseable date are dropped here (raw files carry blank
    trailing rows); what survives must then pass :func:`assert_no_lookahead`.
    """
    label = source or "frame"
    dates = _as_dates(frame, date_column, label)
    kept = frame.loc[dates.notna() & (dates < pd.Timestamp(cutoff))].copy()
    assert_no_lookahead(kept, cutoff, date_column=date_column, source=label)
    return kept
