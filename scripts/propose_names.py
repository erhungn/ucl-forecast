"""Propose canonical names for the clubs that must be mapped across sources.

Scope is the intersection: clubs that appear in European matches and whose
UEFA association has a league in the football-data cache. One of those
appearing under two names would be fitted as two separate teams.

Candidates come from exact matching after normalisation only — never fuzzy or
substring matching, which confuses Roma with Lazio Roma and Sporting CP with
Sporting Braga. Everything else goes to outputs/name_review.csv, in sections:

* ``european-unmapped``        openfootball club with no football-data counterpart
* ``wikipedia-euro-unmapped``  gap-data club in a covered association: needs an id

Three deterministic routes resolve a gap name before it reaches the review list:
an exact match after normalisation against a football-data name; the schedule
bridge, which pins a Wikipedia name to an openfootball one through the matches
both sources describe; and a name that normalises identically to an alias
already approved for a club in the same association.
* ``wikipedia-euro-own-id``    gap-data club from an uncached league: keeps its name
* ``fd-intra-source-variant``  two football-data names that may be one club
* ``wikipedia-fixture-name``   fixture team name with no canonical yet

Similarity is used only to *propose* a pair for review; nothing is auto-accepted.
With --apply, only the exact matches are written, each with its own provenance.
"""

from __future__ import annotations

import argparse
import difflib
import itertools

import pandas as pd

from ucl.config import DATA_RAW, OUTPUTS_DIR, season as season_config
from ucl.data import fixtures as fx
from ucl.data import football_data, names, openfootball, wikipedia
from ucl.data.european import schedule_bridge

REVIEW_COLUMNS = ["section", "item", "country", "detail", "proposal"]
DECISIONS_PATH = DATA_RAW / "name_decisions.csv"


def fd_countries() -> set[str]:
    return {c for c, _ in football_data.MAIN_LEAGUES.values()} | set(
        football_data.EXTRA_LEAGUES.values()
    )


def decided_pairs() -> set[frozenset[str]]:
    """Pairs already reviewed as distinct, so the report stops proposing them."""
    try:
        decisions = pd.read_csv(DECISIONS_PATH, keep_default_na=False)
    except FileNotFoundError:
        return set()
    rows = decisions[decisions.kind == "distinct-pair"]
    return {
        frozenset(part.strip() for part in row.source_name.split("|"))
        for row in rows.itertuples()
    }


def intra_source_variants(domestic: pd.DataFrame) -> list[dict]:
    """football-data names that may be one club recorded under two spellings.

    Such a pair splits a club in two inside a single source, and no mapping
    check catches it because both names are football-data names. The signal:
    their seasons never overlap (a club has one name at a time) and their
    normalised keys share a token or are close. Review only.
    """
    rows: list[dict] = []
    decided = decided_pairs()
    resolved = names.alias_to_canonical("football_data")
    for (country, league), group in domestic.groupby(["country", "league"]):
        seasons: dict[str, set[str]] = {}
        for row in group.itertuples():
            for team in (row.home, row.away):
                seasons.setdefault(team, set()).add(row.season)
        keys = {team: names.normalize_club_name(team) for team in seasons}
        for left, right in itertools.combinations(sorted(seasons), 2):
            if seasons[left] & seasons[right]:
                continue  # played in the same season, so two different clubs
            if frozenset((left, right)) in decided:
                continue  # reviewed and recorded as different clubs
            if left in resolved and resolved[left] == resolved.get(right):
                continue  # already merged into one id
            left_key, right_key = keys[left], keys[right]
            shared = set(left_key.split()) & set(right_key.split())
            ratio = difflib.SequenceMatcher(None, left_key, right_key).ratio()
            if not shared and ratio < 0.75:
                continue
            rows.append(
                {
                    "section": "fd-intra-source-variant",
                    "item": f"{left} | {right}",
                    "country": country,
                    "detail": (
                        f"{league}; seasons {min(seasons[left])}-{max(seasons[left])} "
                        f"vs {min(seasons[right])}-{max(seasons[right])}; "
                        f"shared token {sorted(shared) or 'none'}; similarity {ratio:.2f}"
                    ),
                    "proposal": f"same club? -> {left}",
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--apply", action="store_true", help="append exact matches to the mapping")
    args = parser.parse_args(argv)
    cutoff = season_config(args.season).cutoff_date

    european = openfootball.load_results(cutoff)
    domestic = football_data.load_results(cutoff)
    covered = fd_countries()

    inter = names.intersection_clubs(european, covered)
    known = names.alias_to_canonical("openfootball")

    fd_index: dict[tuple[str, str], set[str]] = {}
    for country, home, away in zip(domestic.country, domestic.home, domestic.away):
        for team in (home, away):
            fd_index.setdefault((country, names.normalize_club_name(team)), set()).add(team)

    already, matched, ambiguous, review = [], [], [], []
    for name, country in sorted(inter.items()):
        if name in known:
            already.append(name)
            continue
        key = (names.association(country), names.normalize_club_name(name))
        hit = fd_index.get(key, set())
        if len(hit) == 1:
            matched.append((name, names.association(country), next(iter(hit)), key[1]))
        elif hit:
            ambiguous.append((name, country, "|".join(sorted(hit))))
        else:
            review.append((name, names.association(country), key[1]))

    print(f"intersection: {len(inter)} European club names in the covered associations")
    print(f"  already mapped          : {len(already)}")
    print(f"  exact-normalised match  : {len(matched)}")
    print(f"  ambiguous (>1 fd club)  : {len(ambiguous)}")
    print(f"  need human review       : {len(review)}")

    # --- review list, in sections ---
    rows: list[dict] = []
    fd_mapped = set(names.alias_to_canonical("football_data"))
    claimed = {fd for _, _, fd, _ in matched}
    unmapped_fd: dict[str, list[str]] = {}
    for country, home, away in zip(domestic.country, domestic.home, domestic.away):
        for team in (home, away):
            if team not in fd_mapped and team not in claimed:
                unmapped_fd.setdefault(country, [])
                if team not in unmapped_fd[country]:
                    unmapped_fd[country].append(team)

    for name, country, key in review:
        candidates = sorted(unmapped_fd.get(country, []))
        # Propose only where normalisation is one abbreviation apart; a shared
        # token is a hint for the human, not an assertion of identity.
        tokens = set(key.split())
        close = [
            c for c in candidates
            if set(names.normalize_club_name(c).split()) & tokens
            or difflib.SequenceMatcher(None, key, names.normalize_club_name(c)).ratio() >= 0.75
        ]
        rows.append(
            {
                "section": "european-unmapped",
                "item": name,
                "country": country,
                "detail": f"normalised {key!r}; {len(candidates)} unmapped fd name(s) in {country}",
                "proposal": " | ".join(close) if close else "",
            }
        )

    rows.extend(intra_source_variants(domestic))

    # Clubs introduced by the Wikipedia gap data (2025-26 UEL/UECL main phase and
    # 2026-27 qualifying). Nothing here is auto-accepted, not even an exact
    # normalised match: these names have never been reviewed.
    try:
        gap = wikipedia.load_results(cutoff)
    except FileNotFoundError:
        gap = None
    wiki_exact: list[tuple[str, str, str, str]] = []
    wiki_bridged: list[tuple[str, str, str]] = []
    wiki_alias_identical: list[tuple[str, str, str, str]] = []
    if gap is not None and len(gap):
        mapping_now = names.load_mapping()
        wiki_mapped = set(names.alias_to_canonical("wikipedia"))
        of_canonical = names.alias_to_canonical("openfootball")
        # Every approved alias, keyed by association and normalised form: a gap
        # name identical to one of these is the same string a human already
        # accepted for that club.
        alias_index: dict[tuple[str, str], set[str]] = {}
        for row in mapping_now.itertuples():
            alias_index.setdefault(
                (row.country, names.normalize_club_name(row.source_name)), set()
            ).add(row.canonical)
        bridge, bridge_problems = schedule_bridge()
        for line in bridge_problems:
            print(f"  bridge problem: {line}")

        gap_clubs: dict[str, str] = {}
        for row in gap.itertuples():
            gap_clubs.setdefault(row.home, row.home_country)
            gap_clubs.setdefault(row.away, row.away_country)

        for name, country in sorted(gap_clubs.items()):
            assoc = names.association(country)
            if name in wiki_mapped:
                continue
            if assoc not in covered:
                rows.append(
                    {
                        "section": "wikipedia-euro-own-id",
                        "item": name, "country": assoc,
                        "detail": "league not cached, keeps its own name as id",
                        "proposal": "",
                    }
                )
                continue
            key = names.normalize_club_name(name)
            hit = fd_index.get((assoc, key), set())
            if len(hit) == 1:
                wiki_exact.append((name, assoc, next(iter(hit)), key))
                continue
            # Deterministic fallback: the schedule bridge pins this Wikipedia
            # name to an openfootball name, which already has an id.
            bridged = bridge.get(name)
            if bridged and bridged in of_canonical:
                wiki_bridged.append((name, assoc, of_canonical[bridged]))
                continue
            approved = alias_index.get((assoc, key), set())
            if len(approved) == 1:
                wiki_alias_identical.append((name, assoc, next(iter(approved)), key))
                continue
            candidates = sorted(
                unmapped_fd.get(assoc, []),
                key=lambda c: -difflib.SequenceMatcher(
                    None, key, names.normalize_club_name(c)
                ).ratio(),
            )[:3]
            rows.append(
                {
                    "section": "wikipedia-euro-unmapped",
                    "item": name, "country": assoc,
                    "detail": (
                        f"normalised {key!r}; not in the 2024-25 schedule bridge"
                        + (f"; bridges to {bridged!r} which has no id" if bridged else "")
                    ),
                    "proposal": " | ".join(candidates),
                }
            )
        print(f"\nwikipedia gap names: {len(wiki_exact)} exact-normalised, "
              f"{len(wiki_bridged)} schedule-bridged, "
              f"{len(wiki_alias_identical)} identical to an approved alias, "
              f"{sum(1 for r in rows if r['section'] == 'wikipedia-euro-unmapped')} unresolved")

    # Fixture team names (Wikipedia spellings) that have no canonical yet.
    # A fixture spelling is settled once it is a mapped wikipedia alias.
    wikipedia_aliases = set(names.alias_to_canonical("wikipedia"))
    try:
        fixtures = pd.read_csv(fx.fixtures_path("2026-27"))
    except FileNotFoundError:
        fixtures = None
    if fixtures is not None:
        for team in sorted(set(fixtures.home) | set(fixtures.away)):
            if team in wikipedia_aliases:
                continue
            rows.append(
                {
                    "section": "wikipedia-fixture-name",
                    "item": team,
                    "country": "",
                    "detail": "fixture spelling differs from the uefa.com short form",
                    "proposal": "",
                }
            )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [(n, c, fd, k) for n, c, fd, k in matched],
        columns=["openfootball_name", "country", "football_data_name", "normalised"],
    ).to_csv(OUTPUTS_DIR / "name_proposals.csv", index=False)
    review_frame = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    review_frame.to_csv(OUTPUTS_DIR / "name_review.csv", index=False)

    print("\nreview list by section:")
    for section, count in review_frame.section.value_counts().items():
        print(f"  {section:26s} {count}")
    if ambiguous:
        print("\nAMBIGUOUS, not applied:")
        for name, country, cands in ambiguous:
            print(f"  {name} ({country}) -> {cands}")

    if args.apply and (matched or wiki_exact or wiki_bridged or wiki_alias_identical):
        mapping = names.load_mapping()
        known_canonicals = set(mapping.canonical)
        fd_to_canonical = names.alias_to_canonical("football_data")
        new_rows = []
        for name, country, fd_name, key in wiki_exact:
            canonical = fd_to_canonical.get(fd_name, fd_name)
            if canonical not in known_canonicals and fd_name not in fd_to_canonical:
                new_rows.append((canonical, country, "football_data", fd_name,
                                 f"exact-normalised:{key}"))
            new_rows.append((canonical, country, "wikipedia", name,
                             f"exact-normalised:{key}"))
        for name, country, canonical in wiki_bridged:
            new_rows.append((canonical, country, "wikipedia", name, "schedule-bridged"))
        for name, country, canonical, key in wiki_alias_identical:
            new_rows.append((canonical, country, "wikipedia", name, f"alias-identical:{key}"))
        for name, country, fd_name, key in matched:
            # Per-row provenance: the key that actually matched, so each row can
            # be audited without re-running the matcher.
            provenance = f"exact-normalised:{key}"
            new_rows.append((fd_name, country, "football_data", fd_name, provenance))
            new_rows.append((fd_name, country, "openfootball", name, provenance))
        added = pd.DataFrame(new_rows, columns=list(mapping.columns)).drop_duplicates()
        combined = pd.concat([mapping, added], ignore_index=True).drop_duplicates(
            subset=["source", "source_name"], keep="first"
        )
        combined.to_csv(names.MAPPING_PATH, index=False)
        print(f"\napplied: mapping now has {len(combined)} alias rows "
              f"({combined.canonical.nunique()} clubs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
