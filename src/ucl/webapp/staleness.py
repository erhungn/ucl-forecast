"""What is out of date, and what is permanently frozen.

Two different things, and the app must not blur them: a domestic cache a
fortnight behind is a hole a refresh would close, while the squad-value snapshot
is frozen because the source is paused and no refresh exists.
"""

from __future__ import annotations

import pandas as pd

from ucl.data.league_results import DOMESTIC_STALE_DAYS, staleness

__all__ = ["DOMESTIC_STALE_DAYS", "warnings", "notes"]


def _as_date(value):
    if value is None or value == "" or pd.isna(value):
        return None
    return pd.Timestamp(value).date()


def warnings(information: pd.DataFrame, status: dict) -> list[str]:
    """Things that could be fixed by fetching something."""
    out: list[str] = []
    as_of = _as_date(status.get("as_of"))
    if as_of is not None and len(information):
        # One definition, shared with the simulator's printed information set.
        note = staleness(information, as_of)
        if note:
            out.append(note)

    sources = set(information.source) if len(information) else set()
    current = information[information.source == "domestic/football_data_current"] \
        if len(information) else information
    if "domestic/football_data_current" not in sources or (
            len(current) and (current.status == "absent").all()):
        out.append(
            "data/raw/football_data_current/ is absent: the weekly domestic "
            "refresh has never been run, so the ratings end wherever the "
            "original cache ends. See scripts/colab/fetch_current.py."
        )

    first = _as_date(status.get("next_matchday_first_date"))
    if first is not None and as_of is not None and first <= as_of:
        out.append(
            f"the next matchday's first fixture is {first}, on or before the "
            f"as-of date {as_of}: the forecast is behind the fixture calendar."
        )
    return out


def notes(information: pd.DataFrame) -> list[str]:
    """Standing caveats, true by design and not fixable by fetching."""
    out: list[str] = []
    if not len(information):
        return out
    frozen = information[information.source.str.startswith("squad_value")]
    for row in frozen.itertuples():
        out.append(
            f"squad value is a pre-season prior frozen at {row.used_to}: the "
            f"Transfermarkt source has been paused since then, so this term "
            f"describes end-of-season squads and no refresh can move it."
        )
    return out
