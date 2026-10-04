"""How current the published forecast is, as one badge.

Separate from :mod:`ucl.webapp.staleness`, which is about the *sources* behind a
forecast. This is about the forecast itself: when it was last run, and whether
the league phase has moved on since.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ucl.webapp import staleness
from ucl.webapp.palette import AGEING, FRESH, STALE

__all__ = ["FRESH_DAYS", "AGEING_DAYS", "LEVELS", "age_in_days", "level"]

#: A run from within this many days is current; a matchday comes round about
#: every three weeks, so two days is comfortably fresh.
FRESH_DAYS = 2
#: Beyond this it is stale whatever else is true.
AGEING_DAYS = 8

LEVELS = {
    "fresh": (FRESH, "green"),
    "ageing": (AGEING, "orange"),
    "stale": (STALE, "red"),
}


def _utc(value) -> datetime | None:
    if not value:
        return None
    stamp = pd.Timestamp(value)
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").to_pydatetime()


def age_in_days(status: dict, *, now: datetime | None = None) -> float | None:
    """Days since the last successful run, or None if it has never run."""
    finished = _utc(status.get("finished_at_utc"))
    if finished is None:
        return None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - finished).total_seconds() / 86400


def level(status: dict, information: pd.DataFrame | None = None, *,
          now: datetime | None = None) -> tuple[str, str, str]:
    """``(level, label, colour)`` for the freshness badge.

    ``fresh`` only when the run is recent *and* no source is behind: a forecast
    computed an hour ago from a fortnight-old cache is not current, and saying
    so on the home screen is the point of the badge.
    """
    if not status:
        return "stale", "never run", STALE
    if status.get("state") == "running":
        return "ageing", "update running", AGEING
    if status.get("state") in ("refused", "failed"):
        return "stale", f"last run {status['state']}", STALE

    days = age_in_days(status, now=now)
    if days is None:
        return "stale", "never run", STALE
    if days < 1:
        label = "updated today"
    elif days < 2:
        label = "updated yesterday"
    else:
        label = f"updated {int(days)} days ago"

    warnings = staleness.warnings(information, status) if information is not None \
        and len(information) else []
    if days >= AGEING_DAYS:
        return "stale", label, STALE
    if days > FRESH_DAYS or warnings:
        return "ageing", label, AGEING
    return "fresh", label, FRESH
