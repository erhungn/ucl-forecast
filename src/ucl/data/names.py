"""The single team-name mapping across sources.

Every source spells clubs differently, and fuzzy matching is unsafe here: in
openfootball alone, substring matching confuses Roma with Lazio Roma, Sporting
CP with Sporting Braga, and Atletico with Real Madrid. So the mapping is an
explicit table, ``data/raw/team_names.csv``, in long form::

    canonical,country,source,source_name,provenance

Long form matters because openfootball carries two spellings per club — the
short form used by the ``world.more`` files and the official long form used by
the ``champions-league`` files — so one column per source cannot hold them.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from ucl.config import DATA_RAW

__all__ = [
    "MAPPING_PATH", "TEAMS_PATH", "AmbiguousAliasError",
    "load_mapping", "load_teams", "alias_to_canonical", "canonical_aliases",
    "add_canonical", "unmapped_names", "normalize_club_name", "intersection_clubs",
    "unmapped_intersection", "TYPE_TOKENS", "ASSOCIATION_OVERRIDES", "association",
]

#: Clubs whose UEFA association is not their own country, because they play in a
#: neighbour's league. Verified against the data: AS Monaco (openfootball MCO)
#: appears in football-data's Ligue 1 as "Monaco" and is never drawn against a
#: French club; FC Vaduz (LIE) appears in the Swiss file as "Vaduz". Without
#: these, such a club is one team in Europe and a different one domestically.
ASSOCIATION_OVERRIDES: dict[str, str] = {"MCO": "FRA", "LIE": "SUI"}


def association(country_code: str) -> str:
    """The UEFA association a country code's clubs are registered with."""
    return ASSOCIATION_OVERRIDES.get(country_code, country_code)

MAPPING_PATH = DATA_RAW / "team_names.csv"
TEAMS_PATH = DATA_RAW / "teams_2026_27.csv"


class AmbiguousAliasError(ValueError):
    """Raised when one source name maps to more than one canonical team."""


def load_mapping(path: Path | None = None) -> pd.DataFrame:
    """Load the mapping table and reject ambiguity up front."""
    frame = pd.read_csv(path or MAPPING_PATH)
    clashes = (
        frame.groupby(["source", "source_name"])["canonical"].nunique().loc[lambda s: s > 1]
    )
    if len(clashes):
        raise AmbiguousAliasError(
            "one source name maps to several canonical teams: "
            + ", ".join(f"{src}:{name}" for src, name in clashes.index)
        )
    return frame


def load_teams(path: Path | None = None) -> pd.DataFrame:
    """Load the confirmed league-phase team list."""
    return pd.read_csv(path or TEAMS_PATH, keep_default_na=False)


def alias_to_canonical(source: str, *, path: Path | None = None) -> dict[str, str]:
    frame = load_mapping(path)
    rows = frame[frame["source"] == source]
    return dict(zip(rows["source_name"], rows["canonical"]))


def canonical_aliases(source: str, *, path: Path | None = None) -> dict[str, list[str]]:
    frame = load_mapping(path)
    rows = frame[frame["source"] == source]
    return rows.groupby("canonical")["source_name"].apply(list).to_dict()


def add_canonical(
    frame: pd.DataFrame,
    source: str,
    *,
    columns: tuple[str, ...] = ("home", "away"),
    path: Path | None = None,
) -> pd.DataFrame:
    """Add ``<column>_canonical`` for each named column; unmapped names become NA."""
    lookup = alias_to_canonical(source, path=path)
    out = frame.copy()
    for column in columns:
        out[f"{column}_canonical"] = out[column].map(lookup)
    return out


def unmapped_names(
    frame: pd.DataFrame,
    source: str,
    *,
    columns: tuple[str, ...] = ("home", "away"),
    path: Path | None = None,
) -> pd.Series:
    """Names present in the data but absent from the mapping, with their frequency.

    The mapping currently covers the 36 league-phase clubs, so this is expected
    to be non-empty: it is the worklist for linking every other club across
    sources, which the goal model will need.
    """
    lookup = alias_to_canonical(source, path=path)
    seen = pd.concat([frame[column] for column in columns], ignore_index=True)
    missing = seen[~seen.isin(lookup)]
    return missing.value_counts()


def dead_aliases(
    frame: pd.DataFrame,
    source: str,
    *,
    columns: tuple[str, ...] = ("home", "away"),
    path: Path | None = None,
) -> list[str]:
    """Aliases in the mapping that never occur in the data — typos or stale names."""
    lookup = alias_to_canonical(source, path=path)
    seen = set(pd.concat([frame[column] for column in columns], ignore_index=True))
    return sorted(name for name in lookup if name not in seen)


# --- Normalisation, for proposing candidate matches only -------------------
# Exact match after normalisation, never fuzzy or substring: substring matching
# confuses Roma with Lazio Roma and Sporting CP with Sporting Braga. Anything
# this does not resolve exactly is reported for human review.

#: Club-type and filler tokens that carry no identity.
TYPE_TOKENS = frozenset("""
fc afc ac acf as ss ssc sc cf cfc cd ud sd rc rcd cs sk fk pfk pfc kf kfc mfk msk kv kaa krc
rsc bk bc ff if il ik gif aik nk hnk ks lks mks vfb vfl tsg bsc fsv sv spvgg pae sfp losc osc
osk ogc sl cp ca club clube calcio
spor kulubu sportif sport futbol football fotball idrettslag ballspielverein association
de du da di of the and
""".split())

_LETTER_SUBS = str.maketrans({"ø": "o", "æ": "a", "å": "a", "ß": "s", "ł": "l", "đ": "d",
                              "ð": "d", "þ": "t", "ı": "i"})


def normalize_club_name(name: str) -> str:
    """Strip accents, punctuation, club-type tokens and founding years."""
    text = str(name).lower().translate(_LETTER_SUBS)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t not in TYPE_TOKENS]
    # Founding years and the German "1. FC" style ordinal carry no identity.
    tokens = [t for t in tokens if not re.fullmatch(r"(?:18|19|20)\d\d|\d{1,2}", t)]
    # Never return an empty key: "AIK" is all club-type tokens but still a club.
    return " ".join(tokens) or re.sub(r"[^a-z0-9]+", " ", text).strip()


def intersection_clubs(
    european: pd.DataFrame, fd_countries: set[str]
) -> dict[str, str]:
    """European club names whose country's domestic league is in the cache.

    These are the clubs that must be mapped: one appearing under two names
    would be fitted as two different teams.
    """
    clubs: dict[str, str] = {}
    for column, country_column in (("home", "home_country"), ("away", "away_country")):
        for name, country in zip(european[column], european[country_column]):
            clubs.setdefault(name, country)
    return {name: cc for name, cc in clubs.items() if association(cc) in fd_countries}


def unmapped_intersection(
    european: pd.DataFrame,
    fd_countries: set[str],
    *,
    path: Path | None = None,
) -> dict[str, str]:
    """Intersection clubs with no openfootball alias in the mapping table."""
    lookup = alias_to_canonical("openfootball", path=path)
    return {
        name: cc
        for name, cc in intersection_clubs(european, fd_countries).items()
        if name not in lookup
    }
