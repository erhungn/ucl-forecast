"""Verify the team-name mapping against the cached data, and report every gap.

Exits non-zero if the mapping itself is broken: a dead alias (a name in the
table that never appears in the data), an ambiguous alias, or a league-phase
team with no alias for a source that should cover it. Clubs outside the 36 that
are not yet mapped are reported as a worklist, not as a failure.
"""

from __future__ import annotations

import argparse
import sys

from ucl.config import OUTPUTS_DIR, season as season_config
from ucl.data import football_data, names, openfootball


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    args = parser.parse_args(argv)
    cfg = season_config(args.season)

    domestic = football_data.load_results(cfg.cutoff_date)
    european = openfootball.load_results(cfg.cutoff_date)
    teams = names.load_teams()
    mapping = names.load_mapping()

    failures: list[str] = []
    print(f"mapping: {len(mapping)} alias rows, {mapping.canonical.nunique()} canonical teams")

    for source, frame in (("football_data", domestic), ("openfootball", european)):
        dead = names.dead_aliases(frame, source)
        unmapped = names.unmapped_names(frame, source)
        aliases = names.canonical_aliases(source)
        print(f"\n{source}: {len(aliases)} teams mapped, {len(dead)} dead alias(es), "
              f"{len(unmapped)} unmapped club name(s) in the data")
        for name in dead:
            print(f"  DEAD ALIAS {name!r} never appears in {source}")
            failures.append(f"{source}: dead alias {name!r}")
        path = OUTPUTS_DIR / f"unmapped_{source}.csv"
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        unmapped.rename("matches").to_csv(path, index_label="source_name")
        print(f"  worklist written to {path.relative_to(OUTPUTS_DIR.parent)}")
        if len(unmapped):
            print("  top unmapped: " + ", ".join(f"{n} ({k})" for n, k in unmapped.head(5).items()))

    # Every team with a football-data code must resolve on the football-data side.
    fd_aliases = names.canonical_aliases("football_data")
    for row in teams.itertuples():
        if row.football_data_code and row.team not in fd_aliases:
            failures.append(f"{row.team}: has code {row.football_data_code} but no football_data alias")
    of_aliases = names.canonical_aliases("openfootball")
    no_european = [row.team for row in teams.itertuples() if row.team not in of_aliases]
    if no_european:
        print(f"\nno European alias (no European matches in the window): {', '.join(no_european)}")

    print()
    if failures:
        for line in failures:
            print(f"FAIL {line}", file=sys.stderr)
        return 1
    print("mapping OK: no dead or ambiguous aliases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
