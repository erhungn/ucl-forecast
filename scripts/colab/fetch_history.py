"""Fetch the 2016-17 to 2020-21 football-data.co.uk history. Run from Colab.

The backtest needs five seasons of forward chaining, and a 2021-22 cutoff with
only one season of history behind it is not comparable to a 2026-27 cutoff with
five. This cache extends the main leagues back to 2016-17 so every fold starts
from a comparable burn-in.

It is a **second cache directory with its own manifest**: the original
``data/raw/football_data/manifest.json`` is never rewritten, and the loader
verifies each directory against its own record. A filename may appear in only
one cache, so the season ranges here and in ``fetch_initial.py`` must not
overlap.

Self-contained on purpose - standard library plus ``requests`` - so it pastes
into a single Colab cell.

    python fetch_history.py --out /content/drive/MyDrive/ucl/football_data_history

Then copy to ``data/raw/football_data_history/``. 55 files: 11 main leagues x 5
seasons. The extra-league files are not re-fetched here; each of those covers
every season in one file and already reaches back to 2012.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://www.football-data.co.uk"
MAIN_CODES = ["E0", "SC0", "D1", "I1", "SP1", "F1", "N1", "B1", "P1", "T1", "G1"]
SEASONS = ["1617", "1718", "1819", "1920", "2021"]


def targets() -> list[tuple[str, str]]:
    return [(f"{code}_{season}.csv", f"{BASE}/mmz4281/{season}/{code}.csv")
            for season in SEASONS for code in MAIN_CODES]


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
        except Exception as error:                      # noqa: BLE001
            if attempt == attempts:
                raise
            print(f"    attempt {attempt} failed ({error}); retrying")
            time.sleep(2.0 * attempt)


def fetch(out_dir: Path, items: list[tuple[str, str]], *, delay: float,
          timeout: float) -> int:
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
        print(f"{len(failures)} FAILED - the cache is incomplete:")
        for line in failures:
            print(f"  {line}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/football_data_history",
                        help="destination directory (default: %(default)s)")
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    return fetch(Path(args.out), targets(), delay=args.delay, timeout=args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
