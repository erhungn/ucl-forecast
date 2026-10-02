"""Map Transfermarkt club ids to our club ids by deterministic routes.

Written for the candidate-2 coverage fix. The first version of
``transfermarkt.club_map`` matched exact normalised names only, which reached
110 of 796 Transfermarkt clubs and left 10 of the 36 league-phase teams
unvalued - an implementation defect, not a property of the data.

Three routes, in order of how direct the evidence is:

* **A, exact-normalised** - the Transfermarkt name normalises to one of our ids.
* **B, alias-identical** - it normalises to an alias already approved for a club
  in ``team_names.csv``.
* **C, schedule bridge** - Transfermarkt's ``games.csv`` and our domestic
  results describe the same matches, so ``(date, home goals, away goals)`` pins
  the pairing. A club is accepted when its leading counterpart has at least
  ``MIN_VOTES`` matched fixtures and at least ``MARGIN`` times the runner-up.

Two guards, both of which caught real errors:

* an **association guard** - a name match may not cross borders (route B had
  mapped Romanian "FC Rapid 1923" onto Austrian "SK Rapid");
* **schedule beats alias** - where route C contradicts route B, the matched
  fixtures win, because they are direct evidence rather than a name guess
  (route B had mapped "Paris FC" onto Paris Saint-Germain through UEFA's short
  name "Paris").

Margin over the runner-up, rather than share of the total, is what route C
accepts on: same-date same-score collisions across 18 leagues put a noise floor
under every club, and a share rule rejected Real Betis at 96 votes to 2.

Anything the routes cannot settle is reported for review and left unmapped;
reviewed decisions live in ``data/raw/transfermarkt_decisions.csv``.
"""

from __future__ import annotations

import argparse
import collections
import sys
from datetime import date

import pandas as pd

from ucl.config import DATA_RAW
from ucl.data import football_data, names, transfermarkt

#: Route C acceptance: votes for the leader, and its margin over the runner-up.
MIN_VOTES = 5
MARGIN = 3.0

DECISIONS_PATH = DATA_RAW / "transfermarkt_decisions.csv"

#: Transfermarkt competition country -> our association code.
COUNTRY_TO_ASSOCIATION = {
    "England": "ENG", "Spain": "ESP", "Italy": "ITA", "Germany": "GER",
    "France": "FRA", "Netherlands": "NED", "Portugal": "POR", "Belgium": "BEL",
    "Turkey": "TUR", "Greece": "GRE", "Scotland": "SCO", "Austria": "AUT",
    "Denmark": "DEN", "Norway": "NOR", "Sweden": "SWE", "Switzerland": "SUI",
    "Poland": "POL", "Romania": "ROU", "Russia": "RUS", "Ukraine": "UKR",
    "Czech Republic": "CZE", "Croatia": "CRO", "Serbia": "SRB",
    "Hungary": "HUN", "Bulgaria": "BUL",
}


def load_decisions() -> pd.DataFrame:
    if not DECISIONS_PATH.exists():
        return pd.DataFrame(columns=["kind", "tm_club_id", "tm_name", "canonical", "note"])
    return pd.read_csv(DECISIONS_PATH)


def build(*, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return the mapping and a report of everything it could not settle."""
    clubs = pd.read_csv(transfermarkt.CACHE_DIR / "clubs.csv")
    competitions = pd.read_csv(transfermarkt.CACHE_DIR / "competitions.csv")
    games = pd.read_csv(
        transfermarkt.CACHE_DIR / "games.csv",
        usecols=["date", "home_club_id", "away_club_id", "home_club_goals",
                 "away_club_goals"], parse_dates=["date"])

    mapping = names.load_mapping()
    our_ids = sorted(set(mapping.canonical))
    name_of = {int(r.club_id): r.name for r in clubs.itertuples()}

    # --- associations, both sides -------------------------------------------
    country_of_competition = dict(
        zip(competitions.competition_id, competitions.country_name))
    tm_association = {
        int(r.club_id): COUNTRY_TO_ASSOCIATION.get(
            country_of_competition.get(r.domestic_competition_id))
        for r in clubs.itertuples()
    }
    our_association = dict(zip(mapping.canonical, mapping.country))

    # Our domestic results, with ids. A club whose name is not in the mapping
    # keeps its football-data name as its id, as it does everywhere else, so
    # route C can reach domestic-only clubs too - and their association comes
    # from the league file they appear in.
    domestic = football_data.load_results(date(2027, 7, 1))
    lookup = names.alias_to_canonical("football_data")
    domestic = domestic.assign(
        home_id=domestic.home.map(lookup).fillna(domestic.home),
        away_id=domestic.away.map(lookup).fillna(domestic.away))
    for row in domestic.itertuples():
        our_association.setdefault(row.home_id, row.country)
        our_association.setdefault(row.away_id, row.country)

    def compatible(club_id: int, canonical: str) -> bool:
        """A match may not cross associations when both sides are known."""
        left, right = tm_association.get(club_id), our_association.get(canonical)
        return left is None or right is None or left == right

    # --- route A: exact normalised name -------------------------------------
    # Collisions must not be resolved by dict iteration order: "Paris FC" and
    # UEFA's short name "Paris" for PSG both normalise to "paris".
    by_key: dict[str, set[str]] = collections.defaultdict(set)
    for club in our_ids:
        by_key[names.normalize_club_name(club)].add(club)

    route_a: dict[int, str] = {}
    for row in clubs.itertuples():
        club_id = int(row.club_id)
        hits = {h for h in by_key.get(names.normalize_club_name(row.name), set())
                if compatible(club_id, h)}
        if len(hits) == 1:
            route_a[club_id] = next(iter(hits))

    # --- route B: identical to an approved alias -----------------------------
    by_alias: dict[str, set[str]] = collections.defaultdict(set)
    for row in mapping.itertuples():
        by_alias[names.normalize_club_name(row.source_name)].add(row.canonical)

    route_b: dict[int, str] = {}
    for row in clubs.itertuples():
        club_id = int(row.club_id)
        if club_id in route_a:
            continue
        hits = {h for h in by_alias.get(names.normalize_club_name(row.name), set())
                if compatible(club_id, h)}
        if len(hits) == 1:
            route_b[club_id] = next(iter(hits))

    # --- route C: schedule bridge -------------------------------------------
    index: dict[tuple, list] = collections.defaultdict(list)
    for row in domestic.itertuples():
        if pd.isna(row.hg) or pd.isna(row.ag):
            continue
        index[(row.date.normalize(), int(row.hg), int(row.ag))].append(
            (row.home_id, row.away_id))

    votes: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in games.itertuples():
        if pd.isna(row.home_club_goals) or pd.isna(row.away_club_goals):
            continue
        bucket = index.get(
            (row.date.normalize(), int(row.home_club_goals), int(row.away_club_goals)))
        if not bucket or len(bucket) != 1:   # same day, same score: no evidence
            continue
        home_id, away_id = bucket[0]
        votes[int(row.home_club_id)][home_id] += 1
        votes[int(row.away_club_id)][away_id] += 1

    route_c: dict[int, str] = {}
    ambiguous: dict[int, list] = {}
    cross_border: dict[int, list] = {}
    for club_id, counter in votes.items():
        top = counter.most_common(2)
        leader, count = top[0]
        runner_up = top[1][1] if len(top) > 1 else 0
        if count < MIN_VOTES:
            continue
        if not compatible(club_id, leader):
            cross_border[club_id] = counter.most_common(3)
            continue
        if count >= MARGIN * max(runner_up, 1):
            route_c[club_id] = leader
        else:
            ambiguous[club_id] = counter.most_common(3)

    # --- resolve, most direct evidence first ---------------------------------
    resolved: dict[int, tuple[str, str, str]] = {}
    overrides, contradictions = [], []
    for club_id in sorted(set(route_a) | set(route_b) | set(route_c)):
        answers = {"A": route_a.get(club_id), "B": route_b.get(club_id),
                   "C": route_c.get(club_id)}
        if club_id in route_a:
            decided, route = route_a[club_id], "exact-normalised"
            if club_id in route_c and route_c[club_id] != route_a[club_id]:
                contradictions.append(
                    (club_id, route_a[club_id], route_c[club_id], votes[club_id].most_common(2)))
        elif club_id in route_c:
            decided, route = route_c[club_id], "schedule-bridged"
            if club_id in route_b and route_b[club_id] != route_c[club_id]:
                overrides.append((club_id, route_b[club_id], route_c[club_id],
                                  votes[club_id].most_common(2)))
        else:
            decided, route = route_b[club_id], "alias-identical"
        # Which other routes reached the same answer independently.
        agreeing = "".join(k for k, v in answers.items() if v == decided)
        resolved[club_id] = (decided, route, agreeing)

    # --- reviewed decisions --------------------------------------------------
    decisions = load_decisions()
    applied = []
    for row in decisions.itertuples():
        club_id = int(row.tm_club_id)
        if row.kind == "distinct-club":
            was = resolved.pop(club_id, (None,))[0]
            applied.append(f"dropped {name_of.get(club_id, club_id)!r} "
                           f"(was {was!r}): {row.note}")
        elif row.kind == "mapped":
            was = resolved.get(club_id, (None,))[0]
            resolved[club_id] = (row.canonical, "user-reviewed", "")
            applied.append(f"{name_of.get(club_id, club_id)!r} -> {row.canonical!r} "
                           f"(was {was!r}): {row.note}")
        else:
            raise ValueError(f"unknown decision kind {row.kind!r}")

    frame = pd.DataFrame(
        [{"tm_club_id": club_id, "tm_name": name_of.get(club_id, ""),
          "canonical": canonical, "route": route, "agreeing_routes": agreeing}
         for club_id, (canonical, route, agreeing) in sorted(resolved.items())])

    report = {
        "clubs": len(clubs), "route_a": len(route_a), "route_b": len(route_b),
        "route_c": len(route_c), "overrides": overrides,
        "contradictions": contradictions, "ambiguous": ambiguous,
        "cross_border": cross_border,
        "decisions": applied, "votes": votes, "name_of": name_of,
        "clubs_frame": clubs, "games": games, "tm_association": tm_association,
    }
    return frame, report


def run_checks(frame: pd.DataFrame, report: dict) -> list[str]:
    """Checks A, B and C, in the same sense as the team-name mapping."""
    failures: list[str] = []
    clubs, name_of = report["clubs_frame"], report["name_of"]
    mapped = dict(zip(frame.tm_club_id, frame.canonical))
    association = report["tm_association"]

    # Check A: a Transfermarkt name that normalises like a mapped one, in the
    # same association, but is itself unmapped - its players would be dropped.
    keys: dict[tuple, set[int]] = collections.defaultdict(set)
    for row in clubs.itertuples():
        club_id = int(row.club_id)
        keys[(association.get(club_id), names.normalize_club_name(row.name))].add(club_id)
    check_a = 0
    for (_, key), ids in sorted(keys.items(), key=lambda kv: str(kv[0])):
        known = {i for i in ids if i in mapped}
        if not known:
            continue
        for club_id in sorted(ids - known):
            check_a += 1
            failures.append(
                f"check A: {name_of[club_id]!r} normalises to {key!r} like mapped "
                f"{name_of[sorted(known)[0]]!r} but is unmapped")
    print(f"check A (unmapped spellings of mapped clubs): {check_a} violation(s)")

    # Check B: one club, one id.
    reverse: dict[str, list[int]] = collections.defaultdict(list)
    for club_id, canonical in mapped.items():
        reverse[canonical].append(int(club_id))
    check_b = 0
    for canonical, ids in sorted(reverse.items()):
        if len(ids) > 1:
            check_b += 1
            failures.append(f"check B: {canonical!r} claimed by "
                            f"{[name_of[i] for i in ids]} - one id, several clubs")
    print(f"check B (one club, one id): {check_b} violation(s)")

    # Check C: two Transfermarkt clubs mapped to one id must never meet.
    check_c = 0
    for row in report["games"].itertuples():
        home, away = int(row.home_club_id), int(row.away_club_id)
        if home in mapped and away in mapped and mapped[home] == mapped[away]:
            check_c += 1
            failures.append(
                f"check C: {name_of[home]!r} v {name_of[away]!r} on "
                f"{row.date.date()} but both map to {mapped[home]!r}")
    print(f"check C (merged clubs never meet): {check_c} violation(s) "
          f"across {len(report['games'])} Transfermarkt games")
    return failures


def legacy_map() -> dict[int, str]:
    """The exact-normalised-only map this script replaces, for the before column."""
    clubs = pd.read_csv(transfermarkt.CACHE_DIR / "clubs.csv")
    our_ids = sorted(set(names.load_mapping().canonical))
    key = {names.normalize_club_name(c): c for c in our_ids}
    out = {}
    for row in clubs.itertuples():
        canonical = key.get(names.normalize_club_name(row.name))
        if canonical:
            out[int(row.club_id)] = canonical
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="write the mapping to data/raw/")
    args = parser.parse_args(argv)

    frame, report = build()
    print(f"Transfermarkt clubs: {report['clubs']}")
    print(f"  route A exact-normalised : {report['route_a']}")
    print(f"  route B alias-identical  : {report['route_b']}")
    print(f"  route C schedule-bridged : {report['route_c']}")
    print(f"  mapped after resolution  : {len(frame)}")
    for route, count in frame.route.value_counts().items():
        print(f"    {route:18s} {count}")

    if report["overrides"]:
        print(f"\nschedule beats alias ({len(report['overrides'])}):")
        for club_id, alias, schedule, top in report["overrides"]:
            print(f"  {report['name_of'][club_id]!r}: alias said {alias!r}, "
                  f"{top[0][1]} matched fixtures say {schedule!r}")
    if report["contradictions"]:
        print(f"\nexact name contradicts the schedule ({len(report['contradictions'])}):")
        for club_id, exact, schedule, top in report["contradictions"]:
            print(f"  {report['name_of'][club_id]!r}: exact {exact!r} vs schedule "
                  f"{schedule!r} {top}")
    if report["decisions"]:
        print("\nreviewed decisions applied:")
        for line in report["decisions"]:
            print(f"  {line}")
    agreement = frame.agreeing_routes.value_counts().to_dict()
    print("  corroboration (routes agreeing on the same answer): "
          + ", ".join(f"{k or 'reviewed'}={v}" for k, v in sorted(agreement.items())))

    if report["cross_border"]:
        print(f"\nrejected by the association guard ({len(report['cross_border'])}):")
        for club_id, top in sorted(report["cross_border"].items(), key=lambda kv: -kv[1][0][1]):
            print(f"  {report['name_of'][club_id]!r:44} -> {top[0][0]!r} "
                  f"{top[0][1]} vote(s), different association")
    if report["ambiguous"]:
        print(f"\nstill ambiguous, left unmapped ({len(report['ambiguous'])}):")
        for club_id, top in sorted(report["ambiguous"].items(), key=lambda kv: -kv[1][0][1]):
            runner_up = top[1][1] if len(top) > 1 else 0
            print(f"  {report['name_of'][club_id]!r:44} -> {top[0][0]!r} "
                  f"{top[0][1]} vs {runner_up} (needs {MARGIN:g}x)")

    failures = run_checks(frame, report)
    for line in failures:
        print(f"  {line}")

    teams = set(names.load_teams().team)
    before = {v for v in legacy_map().values() if v in teams}
    after = {v for v in frame.canonical if v in teams}
    print(f"\nleague-phase teams valued: {len(before)}/36 before, {len(after)}/36 after")
    print("  still unmapped: " + ", ".join(sorted(teams - after)))

    if failures:
        print(f"\nFAILED: {len(failures)} check violation(s); nothing written")
        return 1
    if args.write:
        transfermarkt.MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(transfermarkt.MAP_PATH, index=False)
        print(f"\nwrote {len(frame)} rows to "
              f"{transfermarkt.MAP_PATH.relative_to(DATA_RAW.parent.parent)}")
    else:
        print("\n(dry run; pass --write to save)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
