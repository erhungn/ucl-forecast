"""Build and structurally validate the league-phase fixture list for a season.

2024-25 and 2025-26 come from the cached openfootball cl.txt; 2026-27 from the
Wikipedia league-phase article, which is cross-checked against uefa.com. Every
fetch has a timeout and a cache. Validation failures are fatal.
"""

from __future__ import annotations

import argparse
import sys

from ucl.config import SEASONS, season as season_config
from ucl.data import fixtures as fx
from ucl.data import names


def cross_check_uefa(frame) -> tuple[int, int, list[str]]:
    """Cross-check the fixture team set against the uefa.com-derived names.

    uefa.com publishes its fixtures through a JavaScript shell and its match API
    returns 404, so the pairings cannot be cross-checked there. The club list
    can: those names were captured from uefa.com into the mapping table.

    Returns (matched by normalisation, reconciled via a reviewed alias, missing).
    """
    uefa_keys = {
        names.normalize_club_name(alias): canonical
        for canonical, aliases in names.canonical_aliases("uefa").items()
        for alias in aliases
    }
    reviewed = names.alias_to_canonical("wikipedia")
    direct, via_alias, missing = 0, 0, []
    for team in sorted(set(frame.home) | set(frame.away)):
        if names.normalize_club_name(team) in uefa_keys:
            direct += 1
        elif team in reviewed:
            via_alias += 1
        else:
            missing.append(team)
    return direct, via_alias, missing


def build(season_label: str, *, timeout: float = 25.0):
    wikitext_path, reason = fx.fetch_wikitext(season_label, timeout=timeout)
    text = wikitext_path.read_text(encoding="utf-8") if wikitext_path else ""
    if reason:
        print(f"  wikipedia unavailable for {season_label}: {reason}", file=sys.stderr)

    if season_label == "2026-27":
        if not text:
            raise SystemExit(
                f"2026-27 fixtures need the Wikipedia league-phase article: {reason}"
            )
        frame = fx.from_wikitext(text, season_label)
        source = f"wikipedia ({wikitext_path.name})"
    else:
        frame = fx.from_openfootball(season_label)
        source = f"openfootball ({season_label}_cl.txt)"

    pots = fx.pots_from_wikitext(text, season_label) if text else None
    bridge_problems: list[str] = []
    if pots is not None and len(pots) and season_label != "2026-27":
        # Pots carry Wikipedia names while the fixtures carry openfootball ones.
        # The two sources list the same 144 matches, so the correspondence is
        # derived from the schedule rather than guessed.
        wiki_fixtures = fx.from_wikitext(text, season_label)
        mapping, bridge_problems = fx.bridge_names(frame, wiki_fixtures)
        inverse = {ref: src for src, ref in mapping.items()}
        pots = pots.assign(team=pots.team.map(inverse).fillna(pots.team))
        print(f"  bridged {len(mapping)}/36 team names from the reference source")
    return frame, pots, source, bridge_problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", action="append", help="repeatable; default all three")
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args(argv)
    labels = args.season or list(SEASONS)

    failed = False
    for label in labels:
        cfg = season_config(label)
        frame, pots, source, bridge_problems = build(label, timeout=args.timeout)
        print(f"\n=== {label} ===")
        print(f"source: {source}")
        print(f"matches: {len(frame)} | matchdays {frame.matchday.min()}-{frame.matchday.max()} "
              f"| {frame.date.min().date()} -> {frame.date.max().date()}")
        if pots is None or not len(pots):
            print("pots: UNAVAILABLE (wikitext not fetched) - pot checks skipped")
        else:
            print(f"pots: {len(pots)} teams from the seeding tables")

        for line in bridge_problems:
            print(f"  BRIDGE {line}", file=sys.stderr)
        problems = bridge_problems + fx.validate(frame, pots, season_label=label)

        if label == "2026-27":
            direct, via_alias, unmatched = cross_check_uefa(frame)
            print(f"  uefa.com cross-check: {direct + via_alias}/36 fixture teams reconciled "
                  f"with the uefa.com club list ({direct} by normalisation, {via_alias} by a "
                  f"reviewed alias); pairings not cross-checkable: JS shell, API 404")
            if unmatched:
                print(f"    UNRECONCILED: {', '.join(unmatched)}")
        checks = [
            ("144 matches", len(frame) == 144),
            ("36 teams", len(set(frame.home) | set(frame.away)) == 36),
            ("4 home / 4 away each", not any("home /" in p for p in problems)),
            ("no same-association pairing", not any("same-association" in p for p in problems)),
            ("no repeated pairing", not any("appears" in p for p in problems)),
            ("2 opponents per pot (1H/1A)", pots is not None and len(pots) > 0
             and not any("pot" in p for p in problems)),
        ]
        for name, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

        # The league phase is played after the cutoff; that is expected here.
        assert frame.date.min().date() >= cfg.league_phase_start, "fixture before matchday 1"

        if problems:
            failed = True
            print(f"  {len(problems)} violation(s):")
            for problem in problems[:20]:
                print(f"    {problem}", file=sys.stderr)
            if len(problems) > 20:
                print(f"    ... and {len(problems) - 20} more", file=sys.stderr)
            continue

        frame.to_csv(fx.fixtures_path(label), index=False)
        if pots is not None and len(pots):
            pots.to_csv(fx.pots_path(label), index=False)
        print(f"  written {fx.fixtures_path(label).name}"
              + (f" and {fx.pots_path(label).name}" if pots is not None and len(pots) else ""))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
