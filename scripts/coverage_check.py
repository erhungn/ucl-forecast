"""Coverage check: can every league-phase team actually be fitted?

Run before trusting any model. For each of the 36 teams it reports whether the
football-data cache carries its domestic league and whether the European
results carry the club, and it names every gap instead of working around it.
Exits non-zero if a team has no usable data on either side.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ucl.config import OUTPUTS_DIR, season as season_config
from ucl.data import football_data, names, openfootball


def build(season_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = season_config(season_label)
    teams = names.load_teams()
    domestic = football_data.load_results(cfg.cutoff_date)
    european = openfootball.load_results(cfg.cutoff_date)
    fd_aliases = names.canonical_aliases("football_data")
    of_aliases = names.canonical_aliases("openfootball")

    rows = []
    for team in teams.itertuples():
        fd_names = fd_aliases.get(team.team, [])
        of_names = of_aliases.get(team.team, [])
        fd_hit = domestic[domestic.home.isin(fd_names) | domestic.away.isin(fd_names)]
        of_hit = european[european.home.isin(of_names) | european.away.isin(of_names)]

        gaps = []
        if not team.football_data_code:
            gaps.append(f"no football-data league for {team.country}")
        elif fd_hit.empty:
            gaps.append("football-data league cached but club not found")
        if of_hit.empty:
            gaps.append("no European matches in window")

        rows.append(
            {
                "team": team.team,
                "country": team.country,
                "domestic_league": team.domestic_league,
                "fd_file": f"{team.football_data_code}_*" if team.football_data_code else "",
                "fd_names": "|".join(fd_names),
                "fd_matches": len(fd_hit),
                "fd_first_match": fd_hit.date.min().date() if len(fd_hit) else "",
                "fd_last_match": fd_hit.date.max().date() if len(fd_hit) else "",
                # The extra-leagues files carry every season since 2012, the main
                # files only the six cached ones, so the window is not uniform.
                "fd_window": (
                    f"{fd_hit.date.min():%Y-%m}->{fd_hit.date.max():%Y-%m}" if len(fd_hit) else ""
                ),
                "euro_names": "|".join(of_names),
                "euro_matches": len(of_hit),
                "euro_last_match": of_hit.date.max().date() if len(of_hit) else "",
                "gap": "; ".join(gaps),
            }
        )

    coverage = pd.DataFrame(rows)
    missing = openfootball.missing_report()
    missing = missing[missing.season <= cfg.label]
    return coverage, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    args = parser.parse_args(argv)
    cfg = season_config(args.season)

    coverage, missing = build(args.season)
    show = coverage[[
        "team", "country", "domestic_league", "fd_file", "fd_matches", "fd_window",
        "euro_matches", "euro_last_match", "gap",
    ]]
    print(f"Coverage for {cfg.label}, cutoff {cfg.cutoff_date} "
          f"(matches counted are pre-cutoff only)\n")
    print(show.to_string(index=False))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / f"coverage_{cfg.label}.csv"
    coverage.to_csv(out, index=False)

    flagged = coverage[coverage.gap != ""]
    print(f"\n{len(flagged)} team(s) with a gap:")
    for row in flagged.itertuples():
        print(f"  {row.team:22s} {row.gap}")

    if len(missing):
        print("\nDeclared upstream gaps in the European source:")
        for row in missing.itertuples():
            print(f"  {row.season} {row.competition}: {row.gap}")

    unusable = coverage[(coverage.fd_matches == 0) & (coverage.euro_matches == 0)]
    print(f"\nwritten to {out.relative_to(OUTPUTS_DIR.parent)}")
    if len(unusable):
        for row in unusable.itertuples():
            print(f"UNUSABLE {row.team}: no data on either side", file=sys.stderr)
        return 1
    print(f"all {len(coverage)} teams have usable data on at least one side")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
