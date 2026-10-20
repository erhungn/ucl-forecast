"""Point-in-time squad market value from the Transfermarkt dataset.

Read cache-only from ``data/raw/transfermarkt/`` (fetched by hand from Kaggle,
``davidcariboo/player-scores``).

**The source is frozen.** Appearances stop on 2026-06-28 and valuations on
2026-06-12, with no 2026-27 squads, so for the current season this feature can
only describe the squad as it stood at the end of 2025-26. To keep the backtest
honest the same staleness is imposed on every season: the as-of date is **6 July
of the season's starting year**, not the September cutoff. The feature is
therefore a *pre-season squad-quality prior*, and it cannot be refreshed weekly
while the source is paused.

Squad membership is reconstructed from appearances rather than read from
``players.csv``, whose ``current_club_id`` reflects today and would leak every
transfer since.

Club names are resolved through ``data/raw/transfermarkt_club_map.csv``, built
by ``scripts/build_transfermarkt_map.py`` from deterministic routes and
reviewed decisions. The loader only reads that file, so a club is valued if and
only if the mapping says so.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ucl.config import DATA_RAW
from ucl.data.manifest import require_exact_cache_names

__all__ = ["CACHE_DIR", "MAP_PATH", "as_of_for_season", "squad_values",
           "squad_values_minutes_weighted", "club_map"]

CACHE_DIR = DATA_RAW / "transfermarkt"
#: Club-name mapping, built by scripts/build_transfermarkt_map.py and tracked.
MAP_PATH = DATA_RAW / "transfermarkt_club_map.csv"
#: The dataset's own freeze date; every season is made equally stale.
FREEZE_MONTH_DAY = (7, 6)
#: A player counts as squad if he appeared for the club in this window.
MEMBERSHIP_WINDOW_DAYS = 180
TOP_N = 20


def as_of_for_season(season_label: str) -> date:
    return date(int(season_label[:4]), *FREEZE_MONTH_DAY)


@lru_cache(maxsize=1)
def club_map() -> dict[int, str]:
    """Transfermarkt club_id -> our club id, from the tracked mapping file."""
    if not MAP_PATH.exists():
        raise FileNotFoundError(
            f"{MAP_PATH} is missing; build it with "
            "scripts/build_transfermarkt_map.py --write")
    frame = pd.read_csv(MAP_PATH)
    duplicated = frame.tm_club_id[frame.tm_club_id.duplicated()]
    if len(duplicated):
        raise ValueError(f"transfermarkt map lists {sorted(duplicated)} twice")
    return {int(r.tm_club_id): r.canonical for r in frame.itertuples()}


@lru_cache(maxsize=1)
def _tables():
    require_exact_cache_names(CACHE_DIR)
    valuations = pd.read_csv(
        CACHE_DIR / "player_valuations.csv",
        usecols=["player_id", "date", "market_value_in_eur"], parse_dates=["date"])
    appearances = pd.read_csv(
        CACHE_DIR / "appearances.csv",
        usecols=["player_id", "player_club_id", "date", "minutes_played"],
        parse_dates=["date"])
    return valuations, appearances


@lru_cache(maxsize=1)
def _club_ids() -> frozenset[int]:
    """Every club id in ``clubs.csv``; nothing else counts as a club.

    ``appearances.csv`` also carries national-team matches, whose ids are not in
    the club table. Without this filter a player whose last appearance inside
    the membership window was for his country is assigned to the country and
    drops out of his club's value - which is what happened at the 2026-27 as-of,
    six days after the 2026 World Cup, to 16.3% of players.
    """
    clubs = pd.read_csv(CACHE_DIR / "clubs.csv", usecols=["club_id"])
    return frozenset(int(c) for c in clubs.club_id)


@lru_cache(maxsize=16)
def squad_values(as_of: date) -> dict[str, float]:
    """Total squad value per club at ``as_of``, in euros.

    A player belongs to the club he actually played for in the preceding 180
    days, and is counted at his latest valuation dated strictly before ``as_of``
    - so a transfer after that date leaves him at his old club, as it should.
    Only appearances for a **club** count: a national-team match says nothing
    about club membership, and the spec has always said club.
    """
    valuations, appearances = _tables()
    limit = pd.Timestamp(as_of)
    window = limit - pd.Timedelta(MEMBERSHIP_WINDOW_DAYS, "D")

    recent = appearances[(appearances.date < limit) & (appearances.date >= window)]
    recent = recent[recent.player_club_id.isin(_club_ids())]
    # Most recent club per player inside the window.
    membership = (recent.sort_values("date")
                  .groupby("player_id").player_club_id.last())

    priced = valuations[valuations.date < limit]
    latest = priced.sort_values("date").groupby("player_id").market_value_in_eur.last()

    frame = pd.DataFrame({"club_id": membership}).join(latest, how="inner")
    frame = frame.dropna(subset=["market_value_in_eur"])
    mapping = club_map()
    frame["canonical"] = frame.club_id.map(mapping)
    frame = frame.dropna(subset=["canonical"])

    totals = frame.groupby("canonical").market_value_in_eur.sum()
    return {club: float(value) for club, value in totals.items() if value > 0}


@lru_cache(maxsize=16)
def squad_values_minutes_weighted(as_of: date) -> dict[str, float]:
    """Minutes-weighted mean player value per club at ``as_of``, in euros.

    The total counts a bench of expensive reserves the same as a first eleven.
    This weights each player's valuation by the minutes he actually played
    inside the same 180-day membership window, so the number describes the side
    that takes the field rather than the squad list::

        v_club = sum_p (minutes_p / sum_q minutes_q) * value_p

    A minutes-weighted *mean*, not a per-90 total: a total would scale with how
    many matches a club happened to play in the window, which says nothing
    about its quality. Since the covariate is a log difference, the constant
    scaling cancels either way.

    Membership and the as-of rule are unchanged, so this covers exactly the
    same clubs as :func:`squad_values` - every player with a club assignment
    has minutes, because the assignment comes from an appearance.
    """
    valuations, appearances = _tables()
    limit = pd.Timestamp(as_of)
    window = limit - pd.Timedelta(MEMBERSHIP_WINDOW_DAYS, "D")

    recent = appearances[(appearances.date < limit) & (appearances.date >= window)]
    recent = recent[recent.player_club_id.isin(_club_ids())]
    membership = (recent.sort_values("date")
                  .groupby("player_id").player_club_id.last())
    # Minutes at whichever club the player ended the window with, so the
    # weights and the membership describe the same spell.
    minutes = (recent.merge(membership.rename("club_id"), left_on="player_id",
                            right_index=True)
               .query("player_club_id == club_id")
               .groupby("player_id").minutes_played.sum())

    priced = valuations[valuations.date < limit]
    latest = priced.sort_values("date").groupby("player_id").market_value_in_eur.last()

    frame = pd.DataFrame({"club_id": membership}).join(latest, how="inner")
    frame = frame.join(minutes.rename("minutes"), how="left")
    frame = frame.dropna(subset=["market_value_in_eur"])
    frame["minutes"] = frame.minutes.fillna(0.0)
    mapping = club_map()
    frame["canonical"] = frame.club_id.map(mapping)
    frame = frame.dropna(subset=["canonical"])

    out = {}
    for club, group in frame.groupby("canonical"):
        total_minutes = float(group.minutes.sum())
        if total_minutes <= 0:
            continue
        weighted = float((group.minutes * group.market_value_in_eur).sum()
                         / total_minutes)
        if weighted > 0:
            out[club] = weighted
    return out


def value_difference(values: dict[str, float], home, away) -> np.ndarray:
    """log(value_home) - log(value_away); 0 when either club is unknown."""
    home_v = np.array([values.get(c, np.nan) for c in np.atleast_1d(home)])
    away_v = np.array([values.get(c, np.nan) for c in np.atleast_1d(away)])
    known = ~np.isnan(home_v) & ~np.isnan(away_v)
    difference = np.zeros(len(home_v))
    difference[known] = np.log(home_v[known]) - np.log(away_v[known])
    return difference


def coverage(values: dict[str, float], home, away) -> float:
    known = np.array([c in values for c in np.atleast_1d(home)])
    known_away = np.array([c in values for c in np.atleast_1d(away)])
    return float(np.mean(known & known_away))
