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

from pathlib import Path

import pandas as pd

from ucl.config import DATA_RAW

__all__ = [
    "MAPPING_PATH", "TEAMS_PATH", "AmbiguousAliasError",
    "load_mapping", "load_teams", "alias_to_canonical", "canonical_aliases",
    "add_canonical", "unmapped_names",
]

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
