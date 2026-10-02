"""Apply reviewed name decisions from data/raw/name_decisions.csv to the mapping.

Every football-data name a decision refers to is verified against the cache
first; a name that is not there is reported and nothing is written. The three
kinds of decision become three provenance values:

* ``user-reviewed``  a reviewed alias for an existing club
* ``fd-variant``     a second football-data spelling of the same club
* ``distinct-club``  a club kept separate even though its normalised key
                     collides with another (the normaliser strips founding years)
"""

from __future__ import annotations

import argparse

import pandas as pd

from ucl.config import DATA_RAW, season as season_config
from ucl.data import fixtures as fx
from ucl.data import football_data, names

DECISIONS_PATH = DATA_RAW / "name_decisions.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    decisions = pd.read_csv(DECISIONS_PATH, keep_default_na=False)
    # "distinct-pair" rows record a reviewed "not the same club" decision; they
    # annotate the review report and never become mapping rows.
    pairs = decisions[decisions.kind == "distinct-pair"]
    decisions = decisions[decisions.kind != "distinct-pair"]
    mapping = names.load_mapping()
    domestic = football_data.load_results(season_config(args.season).cutoff_date)

    in_cache: dict[str, set[str]] = {}
    for country, home, away in zip(domestic.country, domestic.home, domestic.away):
        in_cache.setdefault(country, set()).update((home, away))

    # Verify every football-data name a decision relies on. A canonical that is
    # already in the mapping (the 36 league-phase teams use UEFA-style ids) needs
    # no football-data name of its own; a new one does, because for a club with
    # domestic data the football-data name is the id.
    known_canonicals = set(mapping.canonical)
    missing: list[str] = []
    for row in decisions.itertuples():
        wanted = []
        if row.canonical not in known_canonicals:
            wanted.append(row.canonical)
        if row.source == "football_data":
            wanted.append(row.source_name)
        for name in wanted:
            if name not in in_cache.get(row.country, set()):
                missing.append(f"{name!r} ({row.country}) from decision on {row.source_name!r}")
    if missing:
        print(f"{len(missing)} name(s) not in the football-data cache; nothing applied:")
        for line in sorted(set(missing)):
            print(f"  MISSING {line}")
        return 1
    print(f"verified {len(decisions)} decision(s): every football-data name is in the cache")
    print(f"plus {len(pairs)} reviewed distinct-pair annotation(s), which add no mapping rows")

    new_rows: list[tuple[str, str, str, str, str]] = []
    for row in decisions.itertuples():
        provenance = row.kind if row.kind == "user-reviewed" else f"{row.kind} (user-reviewed)"
        new_rows.append((row.canonical, row.country, row.source, row.source_name, provenance))
        if row.canonical not in known_canonicals:
            # New id: make it reachable from its own football-data name too.
            new_rows.append((row.canonical, row.country, "football_data", row.canonical,
                             "user-reviewed"))

    # The fixture spellings that already agree with a uefa.com name need no review.
    uefa_by_key = {
        names.normalize_club_name(alias): canonical
        for canonical, aliases in names.canonical_aliases("uefa").items()
        for alias in aliases
    }
    try:
        fixtures = pd.read_csv(fx.fixtures_path("2026-27"))
    except FileNotFoundError:
        fixtures = None
    auto_wiki = 0
    if fixtures is not None:
        countries = {}
        for row in fixtures.itertuples():
            countries.setdefault(row.home, row.home_country)
            countries.setdefault(row.away, row.away_country)
        for team, country in sorted(countries.items()):
            canonical = uefa_by_key.get(names.normalize_club_name(team))
            if canonical:
                new_rows.append((canonical, country, "wikipedia", team,
                                 f"exact-normalised:{names.normalize_club_name(team)}"))
                auto_wiki += 1

    added = pd.DataFrame(new_rows, columns=list(mapping.columns)).drop_duplicates()
    combined = pd.concat([mapping, added], ignore_index=True).drop_duplicates(
        subset=["source", "source_name"], keep="first"
    )
    print(f"decisions applied: {len(decisions)} reviewed rows, "
          f"{auto_wiki} fixture spellings matched to uefa.com names automatically")
    print(f"mapping: {len(mapping)} -> {len(combined)} alias rows, "
          f"{mapping.canonical.nunique()} -> {combined.canonical.nunique()} clubs")
    if args.dry_run:
        print("(dry run, not written)")
        return 0
    combined.to_csv(names.MAPPING_PATH, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
