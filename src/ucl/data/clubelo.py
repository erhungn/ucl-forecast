"""ClubElo ratings — an optional cross-check, never a dependency.

The ratings we forecast with are computed from match results in
:mod:`ucl.models.elo`. ClubElo is fetched when reachable so the two can be
compared, but the API is http-only, has no mirror, and was returning HTTP 502
from its own server while this pipeline was built. So every failure here is
reported and returned, never raised: a ClubElo outage must not stop a forecast.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ucl.config import DATA_RAW
from ucl.data.manifest import entry_for, verify_cache, write_manifest

__all__ = ["CACHE_DIR", "FetchStatus", "fetch_snapshot", "load_snapshot"]

CACHE_DIR = DATA_RAW / "clubelo"
SNAPSHOT_URL = "http://api.clubelo.com/{day}"


@dataclass(frozen=True)
class FetchStatus:
    """Outcome of a fetch attempt; ``available`` is False on any failure."""

    day: date
    available: bool
    path: Path | None = None
    detail: str = ""
    attempts: int = 0

    def summary(self) -> str:
        if self.available:
            return f"snapshot {self.day} cached at {self.path}"
        return f"UNAVAILABLE for {self.day} after {self.attempts} attempt(s): {self.detail}"


def _snapshot_name(day: date) -> str:
    return f"clubelo_{day.isoformat()}.csv"


def fetch_snapshot(
    day: date,
    *,
    cache_dir: Path | None = None,
    attempts: int = 4,
    base_delay: float = 2.0,
    sleep=time.sleep,
) -> FetchStatus:
    """Fetch the rating snapshot for ``day``, retrying with exponential backoff.

    A cached snapshot is reused and no request is made. Failure is returned as
    ``FetchStatus(available=False)``.
    """
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / _snapshot_name(day)
    if target.exists():
        return FetchStatus(day, True, target, "already cached", 0)

    url = SNAPSHOT_URL.format(day=day.isoformat())
    detail = ""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            if not payload.strip():
                raise ValueError("empty response body")
            target.write_bytes(payload)
            existing = []
            manifest_path = cache_dir / "manifest.json"
            if manifest_path.exists():
                from ucl.data.manifest import read_manifest

                existing = [e for e in read_manifest(cache_dir).values() if e.file != target.name]
            write_manifest(cache_dir, [*existing, entry_for(target, url)])
            return FetchStatus(day, True, target, "fetched", attempt)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            detail = f"{type(error).__name__}: {error}"
            if attempt < attempts:
                sleep(base_delay * 2 ** (attempt - 1))
    return FetchStatus(day, False, None, detail, attempts)


def load_snapshot(
    day: date, *, cache_dir: Path | None = None, verify: bool = True
) -> pd.DataFrame | None:
    """Return the cached snapshot for ``day``, or None if it was never fetched."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    path = cache_dir / _snapshot_name(day)
    if not path.exists():
        return None
    if verify:
        verify_cache(cache_dir, files=[path.name])
    frame = pd.read_csv(path)
    frame.columns = [c.lower() for c in frame.columns]
    return frame
