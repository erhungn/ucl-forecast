"""Fetch the main football-data.co.uk cache. Run this from Google Colab.

football-data.co.uk is DNS-blocked on the author's network, so the loaders in
this repository never touch the network: they read a hash-verified cache. This
script builds that cache, and is the reason the repository works at all without
the payloads, which are gitignored.

It is deliberately **self-contained** - standard library plus ``requests``, no
import from ``ucl`` - so the whole file can be pasted into one Colab cell. The
manifest it writes is exactly what ``ucl.data.manifest`` expects: a JSON list of
``{file, source_url, downloaded_at_utc, sha256, bytes}``.

    python fetch_initial.py --out /content/drive/MyDrive/ucl/football_data

Then copy the directory to ``data/raw/football_data/`` in the repository and run
``.venv/bin/python -c "from ucl.data.manifest import verify_cache; \
verify_cache('data/raw/football_data')"``.

73 files: 11 main leagues x 6 seasons (2021-22 to 2026-27), plus 7 extra-league
files that each cover all seasons. The 2016-17 to 2020-21 history lives in its
own cache with its own manifest - see ``fetch_history.py`` - so this one is never
rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://www.football-data.co.uk"
#: Division codes, matching ucl.data.football_data.MAIN_LEAGUES.
MAIN_CODES = ["E0", "SC0", "D1", "I1", "SP1", "F1", "N1", "B1", "P1", "T1", "G1"]
#: Seasons as football-data spells them in its paths.
SEASONS = ["2122", "2223", "2324", "2425", "2526", "2627"]
#: Extra-league files, one per country, all seasons in one file.
EXTRA_CODES = ["AUT", "DNK", "NOR", "POL", "ROU", "SWE", "SWZ"]


def targets() -> list[tuple[str, str]]:
    """(filename, url) for everything this cache holds."""
    out = [(f"{code}_{season}.csv", f"{BASE}/mmz4281/{season}/{code}.csv")
           for season in SEASONS for code in MAIN_CODES]
    out += [(f"{code}_all.csv", f"{BASE}/new/{code}.csv") for code in EXTRA_CODES]
    return out


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, *, timeout: float, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            if not response.content:
                raise RuntimeError("empty body")
            path.write_bytes(response.content)
            return
        except Exception as error:                      # noqa: BLE001 - reported, not swallowed
            if attempt == attempts:
                raise
            print(f"    attempt {attempt} failed ({error}); retrying")
            time.sleep(2.0 * attempt)


def fetch(out_dir: Path, items: list[tuple[str, str]], *, delay: float,
          timeout: float) -> int:
    """Download ``items`` into ``out_dir`` and write the manifest. Returns exit code."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, failures = [], []
    for index, (name, url) in enumerate(items, 1):
        print(f"[{index:3}/{len(items)}] {name} <- {url}")
        path = out_dir / name
        try:
            download(url, path, timeout=timeout)
        except Exception as error:                      # noqa: BLE001
            failures.append(f"{name}: {error}")
            continue
        entries.append({
            "file": name,
            "source_url": url,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": sha256_of(path),
            "bytes": path.stat().st_size,
        })
        time.sleep(delay)

    manifest = out_dir / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump(sorted(entries, key=lambda e: e["file"]), handle, indent=2)
        handle.write("\n")
    print(f"\n{len(entries)} file(s) recorded in {manifest}")
    if failures:
        # Reported rather than hidden: a silently short cache would quietly
        # shrink the training set instead of failing.
        print(f"{len(failures)} FAILED - the cache is incomplete:")
        for line in failures:
            print(f"  {line}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/football_data",
                        help="destination directory (default: %(default)s)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="seconds between requests (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    return fetch(Path(args.out), targets(), delay=args.delay, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
