"""Fetch the raw data this project is allowed to fetch, and record provenance.

football-data.co.uk is never fetched here: it is DNS-blocked on the author's
network, so its CSVs live in data/raw/football_data/ with a hand-made manifest
and are read from cache only.
"""

from __future__ import annotations

import argparse
import sys

from ucl.config import season as season_config
from ucl.data import clubelo, openfootball


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27", help="season label, e.g. 2026-27")
    parser.add_argument(
        "--skip-clubelo",
        action="store_true",
        help="skip the optional ClubElo cross-check snapshot",
    )
    args = parser.parse_args(argv)
    cfg = season_config(args.season)

    print(f"openfootball: fetching {len(openfootball.SOURCES)} files -> {openfootball.CACHE_DIR}")
    failed = openfootball.fetch()
    for line in failed:
        print(f"  FAILED {line}", file=sys.stderr)
    print(f"  done, {len(openfootball.SOURCES) - len(failed)} ok, {len(failed)} failed")
    for (season_label, competition), why in sorted(openfootball.MISSING.items()):
        print(f"  declared gap: {season_label} {competition}: {why}")

    if not args.skip_clubelo:
        print(f"clubelo: snapshot at {cfg.cutoff_date} (optional cross-check)")
        status = clubelo.fetch_snapshot(cfg.cutoff_date)
        print(f"  {status.summary()}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
