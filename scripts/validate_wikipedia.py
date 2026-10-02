"""Validate the Wikipedia parser against openfootball before trusting it.

Wikipedia is used only to fill gaps openfootball does not cover, so the parser
must first be checked where both sources overlap: the 2024-25 Europa and
Conference League main phases. Team names differ between the sources, so the
correspondence is derived from the schedule (the same technique the fixture
pot join uses) rather than guessed, and then every match is compared on date,
score, extra time and penalties.

Reports the agreement rate and every mismatch. Exits non-zero if agreement is
below --min-agreement.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ucl.config import OUTPUTS_DIR, season as season_config
from ucl.data import openfootball, wikipedia
from ucl.data.fixtures import bridge_names
from ucl.data.names import normalize_club_name

# The league phase has no extra time, so the knockout rounds are compared too:
# they are the only place a.e.t. and penalty handling can be checked, including
# the 90-minute score recovered from goal minutes.
PAIRS = [
    ("2024-25", "UEL", "league_phase"), ("2024-25", "UECL", "league_phase"),
    ("2024-25", "UEL", "knockout"), ("2024-25", "UECL", "knockout"),
    ("2024-25", "UEL", "final"), ("2024-25", "UECL", "final"),
]


def _wiki(season_label: str, competition: str, kinds: tuple[str, ...]) -> pd.DataFrame:
    pages = [
        p for p in wikipedia.PAGES
        if p.season == season_label and p.competition == competition and p.kind in kinds
    ]
    frames = [
        wikipedia.parse_matches(
            (wikipedia.CACHE_DIR / p.file_name).read_text(encoding="utf-8"),
            season=p.season, competition=p.competition, source_file=p.file_name,
        )
        for p in pages
        if (wikipedia.CACHE_DIR / p.file_name).exists()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _openfootball(season_label: str, competition: str) -> pd.DataFrame:
    path = openfootball.CACHE_DIR / f"{season_label}_{'el' if competition == 'UEL' else 'conf'}.txt"
    return openfootball.parse_file(path, season=season_label, competition=competition)


def _bridge(wiki: pd.DataFrame, of: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """Map Wikipedia names onto openfootball names via the shared schedule."""
    def shaped(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["home", "away", "home_country", "away_country"]].copy()
        out["matchday"] = 0  # a constant key: the (venue, opponent) set identifies a club
        return out

    return bridge_names(shaped(wiki), shaped(of))


def compare(
    season_label: str, competition: str, phase: str
) -> tuple[pd.DataFrame, list[str], dict]:
    wiki = _wiki(season_label, competition, (phase,))
    of_all = _openfootball(season_label, competition)
    # openfootball bundles the whole competition in one file; split it by phase.
    is_league = of_all.stage.str.contains("League|Group|Matchday", regex=True, na=False)
    is_final = of_all.stage.str.fullmatch("Final|Finals, Final", na=False)
    if phase == "league_phase":
        of = of_all[is_league].copy()
    elif phase == "final":
        of = of_all[is_final].copy()
    else:
        of = of_all[~is_league & ~is_final].copy()

    mapping, bridge_problems = _bridge(wiki, of)
    wiki = wiki.assign(
        home_of=wiki.home.map(mapping), away_of=wiki.away.map(mapping)
    )

    of_index = {(row.home, row.away): row for row in of.itertuples()}
    mismatches: list[str] = []
    agree = 0
    checked = 0
    for row in wiki.itertuples():
        key = (row.home_of, row.away_of)
        other = of_index.pop(key, None)
        if other is None:
            mismatches.append(
                f"{competition} {season_label}: no openfootball match for "
                f"{row.home} v {row.away} ({row.home_of} v {row.away_of})"
            )
            continue
        checked += 1
        problems = []
        if row.date.date() != other.date.date():
            problems.append(f"date {row.date.date()} vs {other.date.date()}")
        if (row.hg, row.ag) != (other.hg, other.ag):
            problems.append(
                f"90-min score {row.hg}-{row.ag} vs {other.hg}-{other.ag}"
            )
        if bool(row.aet) != bool(other.aet):
            problems.append(f"aet {row.aet} vs {other.aet}")
        if bool(row.aet) and bool(other.aet) and (row.et_hg, row.et_ag) != (other.et_hg, other.et_ag):
            problems.append(
                f"after-ET score {row.et_hg}-{row.et_ag} vs {other.et_hg}-{other.et_ag}"
            )
        wiki_pens = (row.pens_h, row.pens_a)
        of_pens = (other.pens_h, other.pens_a)
        if pd.notna(wiki_pens[0]) or pd.notna(of_pens[0]):
            if wiki_pens != of_pens:
                problems.append(f"pens {wiki_pens} vs {of_pens}")
        if problems:
            mismatches.append(
                f"{competition} {season_label}: {other.home} v {other.away}: "
                + "; ".join(problems)
            )
        else:
            agree += 1
    for leftover in of_index.values():
        mismatches.append(
            f"{competition} {season_label}: openfootball has "
            f"{leftover.home} v {leftover.away} with no Wikipedia match"
        )

    stats = {
        "competition": competition,
        "season": season_label,
        "phase": phase,
        "wikipedia_matches": len(wiki),
        "openfootball_matches": len(of),
        "paired": checked,
        "agree": agree,
        "mismatches": len(mismatches),
        "unbridged_names": len(bridge_problems),
        "aet_matches": int(wiki.aet.sum()),
        "pen_matches": int(wiki.pens_h.notna().sum()),
    }
    return wiki, mismatches + bridge_problems, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-agreement", type=float, default=0.99)
    args = parser.parse_args(argv)

    rows, all_mismatches = [], []
    for season_label, competition, phase in PAIRS:
        _, mismatches, stats = compare(season_label, competition, phase)
        rows.append(stats)
        all_mismatches.extend(mismatches)
        rate = stats["agree"] / stats["paired"] if stats["paired"] else 0.0
        print(f"{competition} {season_label} {phase:12s}: wikipedia {stats['wikipedia_matches']:3d}, "
              f"openfootball {stats['openfootball_matches']:3d}, paired {stats['paired']:3d}, "
              f"identical {stats['agree']:3d} ({rate:6.1%}), mismatches {stats['mismatches']:2d}, "
              f"unbridged {stats['unbridged_names']}, aet {stats['aet_matches']:2d}, "
              f"pens {stats['pen_matches']}")

    total_paired = sum(r["paired"] for r in rows)
    total_agree = sum(r["agree"] for r in rows)
    overall = total_agree / total_paired if total_paired else 0.0
    print(f"\noverall: {total_agree}/{total_paired} matches identical on teams, date, "
          f"score, a.e.t. and penalties = {overall:.2%}")

    if all_mismatches:
        print(f"\nevery mismatch ({len(all_mismatches)}):")
        for line in all_mismatches:
            print(f"  {line}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "wikipedia_validation.csv", index=False)
    if overall < args.min_agreement:
        print(f"\nFAIL agreement {overall:.2%} below {args.min_agreement:.0%}; "
              "Wikipedia must not be used for the gaps yet")
        return 1
    print("\nparser validated: Wikipedia may be used for the gaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
