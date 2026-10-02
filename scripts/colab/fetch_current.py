"""Refresh the current season's football-data files. Run from Colab, weekly.

The in-season update mode is driven by an as-of date, and it is only as current
as the domestic results behind it: every forecast prints how stale each source
is, and a gap of a week or more shows up as a NOTE in the information set. This
script closes that gap.

It writes into a **third** cache directory, ``data/raw/football_data_current/``,
which the loader reads alongside the other two. A filename may live in only one
cache, so a refreshed ``E0_2627.csv`` collides with the copy already in
``data/raw/football_data/``: the script reports exactly which stale files to
remove, and ``--prune-stale`` removes them (and their manifest entries) for you.

Self-contained on purpose - standard library plus ``requests`` - so it pastes
into a single Colab cell.

    python fetch_current.py --season 2627 \
        --out /content/drive/MyDrive/ucl/football_data_current \
        --cache /content/drive/MyDrive/ucl/football_data --prune-stale

The extra-league files are re-fetched too: each covers every season in one file,
so the current season only arrives through a fresh copy of the whole file.
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
EXTRA_CODES = ["AUT", "DNK", "NOR", "POL", "ROU", "SWE", "SWZ"]


def targets(season: str) -> list[tuple[str, str]]:
    out = [(f"{code}_{season}.csv", f"{BASE}/mmz4281/{season}/{code}.csv")
           for code in MAIN_CODES]
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
        except Exception as error:                      # noqa: BLE001
            if attempt == attempts:
                raise
            print(f"    attempt {attempt} failed ({error}); retrying")
            time.sleep(2.0 * attempt)


def fetch(out_dir: Path, items: list[tuple[str, str]], *, delay: float,
          timeout: float) -> tuple[int, list[str]]:
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
        print(f"{len(failures)} FAILED - the refresh is incomplete:")
        for line in failures:
            print(f"  {line}")
    return (1 if failures else 0), [e["file"] for e in entries]


def resolve_collisions(cache_dir: Path, fetched: list[str], *, prune: bool) -> None:
    """Report, or remove, the superseded copies in the original cache."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"\nno manifest in {cache_dir}; skipping the collision check")
        return
    with open(manifest_path, encoding="utf-8") as handle:
        rows = json.load(handle)
    stale = [name for name in fetched if (cache_dir / name).exists()]
    if not stale:
        print(f"\nno filename collides with {cache_dir}")
        return

    print(f"\n{len(stale)} file(s) now exist in both caches, which the loader "
          f"refuses (one season must live in exactly one cache):")
    for name in stale:
        print(f"  {cache_dir / name}")
    if not prune:
        print("  re-run with --prune-stale to remove these and their manifest "
              "entries, or delete them by hand")
        return

    for name in stale:
        (cache_dir / name).unlink()
    kept = [row for row in rows if row["file"] not in set(stale)]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(sorted(kept, key=lambda e: e["file"]), handle, indent=2)
        handle.write("\n")
    print(f"  pruned; {manifest_path} now records {len(kept)} file(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2627",
                        help="season as football-data spells it (default: %(default)s)")
    parser.add_argument("--out", default="data/raw/football_data_current")
    parser.add_argument("--cache", default="data/raw/football_data",
                        help="the original cache, checked for superseded copies")
    parser.add_argument("--prune-stale", action="store_true",
                        help="delete the superseded copies from --cache")
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    code, fetched = fetch(Path(args.out), targets(args.season),
                          delay=args.delay, timeout=args.timeout)
    resolve_collisions(Path(args.cache), fetched, prune=args.prune_stale)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
