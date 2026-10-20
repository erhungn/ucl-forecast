"""The decisions the weekly update makes, separated from running it.

``scripts/weekly_update.py`` is a thin CLI over these: the refusals, the
matchday arithmetic and the cache bookkeeping are all here so they can be tested
without running a five-minute simulation or touching the real caches.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ucl.data import clubelo, football_data, openfootball, transfermarkt, wikipedia
from ucl.data.manifest import (
    MANIFEST_NAME, read_manifest, require_exact_cache_names, write_manifest,
)

__all__ = [
    "Collision", "cache_directories", "cache_precedence", "check_cache_names",
    "collisions", "prune_superseded",
    "latest_complete_prefix", "partial_matchdays", "missing_before",
    "inputs_hash", "publish", "read_status", "write_status",
]


@dataclass(frozen=True)
class Collision:
    """One filename present in two caches, and which copy is superseded."""

    name: str
    superseded: Path
    winner: Path

    def __str__(self) -> str:
        return f"{self.superseded} (superseded by {self.winner})"


def cache_precedence() -> tuple[Path, ...]:
    """Caches from most to least current.

    ``football_data_current`` is the weekly refresh, so it wins; the original
    cache beats the 2016-21 history, which should never overlap it anyway.
    """
    return (
        football_data.CURRENT_CACHE_DIR,
        football_data.CACHE_DIR,
        football_data.HISTORY_CACHE_DIR,
    )


def cache_directories() -> tuple[Path, ...]:
    """Every directory under ``data/raw/`` that a loader expects by name.

    The list the name check sweeps. A source whose cache is optional belongs on
    it too: an optional cache is exactly the one whose absence raises nothing.
    """
    return (
        *football_data.CACHE_DIRS,
        openfootball.CACHE_DIR,
        wikipedia.CACHE_DIR,
        transfermarkt.CACHE_DIR,
        clubelo.CACHE_DIR,
    )


def check_cache_names(cache_dirs: tuple[Path, ...] | None = None) -> None:
    """Raise if any expected cache is on disk under a near-miss name.

    One sweep over every source, up front, so a run that is about to read a
    cache as absent says which folder it should have been reading instead.
    """
    require_exact_cache_names(*(cache_dirs or cache_directories()))


def collisions(cache_dirs: tuple[Path, ...] | None = None) -> list[Collision]:
    """Every filename present in more than one cache.

    ``football_data.cached_files`` raises on the *first* such file, from deep
    inside the ratings step. This finds all of them, up front, for the price of
    a directory listing.
    """
    order = [Path(d) for d in (cache_dirs or cache_precedence()) if Path(d).is_dir()]
    seen: dict[str, Path] = {}
    found: list[Collision] = []
    for directory in order:
        for group in football_data.available_files(directory).values():
            for name in group:
                if name in seen:
                    # ``order`` is most-current first, so the later one loses.
                    found.append(Collision(name, directory / name, seen[name] / name))
                else:
                    seen[name] = directory
    return sorted(found, key=lambda c: str(c.superseded))


def prune_superseded(found: list[Collision], *, dry_run: bool = False) -> list[Path]:
    """Delete the superseded copies and drop them from their manifests.

    Irreversible, which is why the caller must ask for it explicitly. Returns
    the paths removed (or that would be, under ``dry_run``).
    """
    removed: list[Path] = []
    by_directory: dict[Path, list[str]] = {}
    for collision in found:
        by_directory.setdefault(collision.superseded.parent, []).append(
            collision.superseded.name)
        removed.append(collision.superseded)
    if dry_run:
        return removed

    for directory, names in by_directory.items():
        for name in names:
            (directory / name).unlink(missing_ok=True)
        if (directory / MANIFEST_NAME).exists():
            manifest = read_manifest(directory)
            kept = [entry for key, entry in manifest.items() if key not in set(names)]
            write_manifest(directory, kept)
    return removed


def latest_complete_prefix(played: pd.DataFrame, fixtures: pd.DataFrame) -> int:
    """The largest ``k`` with matchdays ``1..k`` all complete.

    Not ``max(completed_matchdays(...))``: if matchday 2 has a postponed fixture
    while 3 finishes on time, the completed set is ``[1, 3]`` and the maximum is
    a matchday the simulator will refuse to run through. The contiguous prefix
    is the honest answer, and the gap is reported separately.
    """
    from ucl.data.league_results import completed_matchdays

    complete = set(completed_matchdays(played, fixtures))
    matchday = 0
    while matchday + 1 in complete:
        matchday += 1
    return matchday


def partial_matchdays(played: pd.DataFrame, fixtures: pd.DataFrame
                      ) -> dict[int, tuple[int, int]]:
    """Matchdays with some results but not all: ``{matchday: (have, want)}``."""
    expected = fixtures.groupby("matchday").size()
    actual = played.groupby("matchday").size()
    out = {}
    for matchday in expected.index:
        have = int(actual.get(matchday, 0))
        if 0 < have < int(expected[matchday]):
            out[int(matchday)] = (have, int(expected[matchday]))
    return out


def missing_before(played: pd.DataFrame, fixtures: pd.DataFrame, matchday: int
                   ) -> list[int]:
    """Completed matchdays beyond ``matchday`` - evidence a round was skipped."""
    from ucl.data.league_results import completed_matchdays

    return [m for m in completed_matchdays(played, fixtures) if m > matchday]


def inputs_hash(played: pd.DataFrame,
                cache_dirs: tuple[Path, ...] | None = None) -> str:
    """A digest of everything that can change a forecast.

    The played matches plus every cached domestic file's recorded hash. A
    corrected scoreline, a domestic refresh or a pruned cache all change it; a
    re-run that changes nothing does not, which is what makes the update
    idempotent.
    """
    digest = hashlib.sha256()
    digest.update(played.to_csv(index=False).encode("utf-8"))
    for directory in sorted({Path(d) for d in (cache_dirs or cache_precedence())
                             if Path(d).is_dir()}, key=str):
        if not (directory / MANIFEST_NAME).exists():
            continue
        manifest = read_manifest(directory)
        for name in sorted(manifest):
            digest.update(f"{directory.name}/{name}:{manifest[name].sha256}".encode())
    return f"sha256:{digest.hexdigest()}"


def publish(source: Path, destination: Path) -> Path:
    """Copy into ``live/`` atomically, so a reader never sees half a file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    return destination


def read_status(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_status(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)
    return path
