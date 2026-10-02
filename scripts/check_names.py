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

    # The intersection gate: a European club whose domestic league is cached must
    # be mapped, or it would be fitted as two separate teams.
    fd_countries = {c for c, _ in football_data.MAIN_LEAGUES.values()} | set(
        football_data.EXTRA_LEAGUES.values()
    )
    unmapped_inter = names.unmapped_intersection(european, fd_countries)
    total_inter = len(names.intersection_clubs(european, fd_countries))
    print(f"\nintersection gate: {total_inter - len(unmapped_inter)}/{total_inter} "
          f"club names mapped")
    for name, country in sorted(unmapped_inter.items()):
        failures.append(f"intersection club unmapped: {name} ({country})")

    # --- Check A: reverse direction, with the fd spelling-variant escape hatch ---
    # A mapped club may have domestic data under a second football-data spelling.
    # If that second name is not mapped too, the club's own domestic matches are
    # silently ignored. Declaring the row's provenance as "fd-variant" is the
    # escape hatch for a deliberate second spelling.
    fd_alias = names.alias_to_canonical("football_data")
    variant_rows = set(
        mapping.loc[mapping.provenance.fillna("").str.startswith("fd-variant"), "source_name"]
    )
    fd_keys: dict[tuple[str, str], set[str]] = {}
    for country, home, away in zip(domestic.country, domestic.home, domestic.away):
        for team in (home, away):
            fd_keys.setdefault((country, names.normalize_club_name(team)), set()).add(team)

    check_a = 0
    for (country, key), spellings in sorted(fd_keys.items()):
        mapped = {s for s in spellings if s in fd_alias}
        if not mapped:
            continue
        for spelling in sorted(spellings - mapped):
            if spelling in variant_rows:
                continue
            check_a += 1
            failures.append(
                f"check A: {spelling!r} ({country}) normalises to {key!r} like mapped "
                f"{sorted(mapped)[0]!r} but is unmapped; add it or mark it fd-variant"
            )
    print(f"\ncheck A (fd spellings of mapped clubs): {check_a} unmapped spelling(s)")

    # --- Check B: one club, one id. Hard failure. ---
    check_b = 0
    per_canonical = (
        mapping[mapping.source == "football_data"].groupby("canonical").source_name.apply(list)
    )
    for canonical, spellings in per_canonical.items():
        # One spelling is the primary; every further one must be declared.
        unmarked = [s for s in spellings if s not in variant_rows]
        if len(unmarked) > 1:
            check_b += 1
            failures.append(
                f"check B: {canonical!r} has several football-data names {unmarked} "
                "and none but one may be left unmarked; mark the others fd-variant"
            )
    distinct = set(
        mapping.loc[mapping.provenance.fillna("").str.startswith("distinct-club"), "source_name"]
    )
    seen_keys: dict[tuple[str, str], tuple[str, str]] = {}
    for row in mapping[mapping.source == "football_data"].itertuples():
        key = (row.country, names.normalize_club_name(row.source_name))
        other, other_name = seen_keys.setdefault(key, (row.canonical, row.source_name))
        if other == row.canonical:
            continue
        # Escape hatch: a declared distinct club whose key only collides because
        # the normaliser strips founding years.
        years_differ = names.normalize_club_name(
            row.source_name, keep_years=True
        ) != names.normalize_club_name(other_name, keep_years=True)
        if (row.source_name in distinct or other_name in distinct) and years_differ:
            print(f"  check B escape hatch: {row.source_name!r} and {other_name!r} "
                  f"declared distinct clubs (keys collide only once years are stripped)")
            continue
        check_b += 1
        failures.append(
            f"check B: ids {other!r} and {row.canonical!r} both normalise to {key[1]!r} "
            f"in {key[0]} - one club split in two"
        )
    print(f"check B (one club, one id): {check_b} violation(s)")

    # --- Check C: merged names must never meet. Hard failure. ---
    # Any two names mapped to one id are claimed to be the same club, so if they
    # appear on opposite sides of a match the merge is wrong.
    check_c = 0
    for source, frame in (("football_data", domestic), ("openfootball", european)):
        for row in names.merge_conflicts(frame, source).itertuples():
            check_c += 1
            failures.append(
                f"check C: {source} has {row.home!r} v {row.away!r} on "
                f"{row.date.date()} but both map to {row.home_canonical!r}"
            )
    print(f"check C (merged names never meet): {check_c} violation(s) "
          f"across {len(domestic) + len(european)} matches")

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
        for line in failures[:60]:
            print(f"FAIL {line}", file=sys.stderr)
        if len(failures) > 60:
            print(f"FAIL ... and {len(failures) - 60} more", file=sys.stderr)
        print(f"\n{len(failures)} failure(s); unresolved names are listed in "
              f"outputs/name_review.csv", file=sys.stderr)
        return 1
    print("mapping OK: no dead or ambiguous aliases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
