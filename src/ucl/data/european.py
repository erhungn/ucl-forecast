"""All European club results, from openfootball plus the Wikipedia gap fills.

openfootball is the primary source. It does not publish the 2025-26 Europa and
Conference League main phases, nor anything for 2026-27, so those come from
Wikipedia, whose parser is validated against openfootball where they overlap
(see scripts/validate_wikipedia.py). Only Wikipedia's "gap" pages are loaded,
so no match is counted twice.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from ucl.data import openfootball, wikipedia

__all__ = ["load_results", "coverage", "is_main_phase", "PhaseError"]


class PhaseError(ValueError):
    """A stage string that cannot be sorted into qualifying or main phase."""


#: Only the openfootball ``world.more`` files need a stage rule: they carry
#: ``phase == "mixed"`` because one file holds the qualifying rounds and the main
#: draw together (2021-22 to 2023-24). Everywhere else ``phase`` is authoritative
#: and is used instead, which matters because the bare string "Playoffs" means
#: the qualifying play-off round in one place and the new format's knockout
#: play-off in another - 2024-25 carries both, so the string alone cannot decide.
_MIXED_QUALIFYING = re.compile(
    r"^(?:preliminary\s+round"
    r"|\d(?:st|nd|rd|th)\s+qualifying\s+round"
    r"|play-?offs)$",
    re.IGNORECASE,
)
#: The main-phase shapes those same files use. Listed rather than inferred as
#: "whatever is left": an unrecognised string means the source layout changed,
#: and silently calling it main phase would move qualifying matches into the
#: competition's own goal-scale term.
_MIXED_MAIN = re.compile(
    r"^(?:group|gruppe|knockout|round\s+of|quarter|semi|final)",
    re.IGNORECASE,
)


def is_main_phase(frame: pd.DataFrame) -> pd.Series:
    """True where a match belongs to a competition's main draw.

    The qualifying rounds are a different population from the draw they feed:
    across the cache the UCL main phase scores 3.13 goals a match against 2.79 in
    qualifying, so anything fitted per competition has to separate them.

    ``phase`` decides wherever it is "main" or "qualifying". For the "mixed"
    files the stage string decides, and a string matching neither list **raises**
    rather than defaulting - the same reasoning as the cache-name guards, where a
    near miss treated as a legitimate value was the whole failure.
    """
    phase = frame.phase.astype(str)
    stage = frame.stage.fillna("").astype(str).str.strip()
    mixed = phase == "mixed"
    qualifying = stage.str.match(_MIXED_QUALIFYING)
    main = stage.str.match(_MIXED_MAIN)
    unknown = mixed & ~qualifying & ~main
    if unknown.any():
        found = sorted(set(stage[unknown]))
        raise PhaseError(
            f"{int(unknown.sum())} match(es) in a mixed-phase file carry a stage "
            f"this cannot sort into qualifying or main phase: "
            f"{', '.join(repr(s) for s in found[:5])}"
            + (f" and {len(found) - 5} more" if len(found) > 5 else "")
            + ". Add it to european._MIXED_QUALIFYING or european._MIXED_MAIN."
        )
    return (~mixed & (phase == "main")) | (mixed & main)


def load_results(
    cutoff: date,
    *,
    competitions: list[str] | None = None,
    verify: bool = True,
) -> pd.DataFrame:
    """Every European club match strictly before ``cutoff``, from both sources."""
    frames = [
        openfootball.load_results(cutoff, competitions=competitions, verify=verify).assign(
            source="openfootball"
        )
    ]
    try:
        gaps = wikipedia.load_results(cutoff, verify=verify).assign(source="wikipedia")
    except FileNotFoundError:
        gaps = None
    if gaps is not None and len(gaps):
        if competitions is not None:
            gaps = gaps[gaps.competition.isin(competitions)]
        frames.append(gaps)

    combined = pd.concat(frames, ignore_index=True)
    # Guard against double-counting at the level the gap design works on: each
    # season-competition-phase is covered by exactly one source. Phase matters:
    # for 2025-26 UEL openfootball has the qualifying rounds and Wikipedia the
    # main phase, which is a split, not an overlap. Match-level keys
    # cannot be used, because the same pairing legitimately recurs within a
    # competition (Bodo/Glimt met Roma in both the 2021-22 UECL group stage and
    # the quarter-finals) and the two sources spell clubs differently anyway.
    overlap = combined.groupby(["season", "competition", "phase"]).source.nunique()
    clashes = overlap[overlap > 1]
    if len(clashes):
        raise ValueError(
            "these season-competitions are present in both sources, so matches "
            f"would be double-counted: {sorted(clashes.index)}"
        )
    return combined.sort_values(["date", "competition", "home"]).reset_index(drop=True)


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Matches per season, competition, phase and source, for reporting."""
    return (
        frame.groupby(["season", "competition", "phase", "source"])
        .size()
        .rename("matches")
        .reset_index()
        .sort_values(["season", "competition", "phase", "source"])
    )


def schedule_bridge(
    pairs: tuple[tuple[str, str], ...] = (("2024-25", "UEL"), ("2024-25", "UECL")),
) -> tuple[dict[str, str], list[str]]:
    """Map Wikipedia club names onto openfootball ones via a shared schedule.

    For a season-competition both sources cover, the two describe the same
    matches, so the name correspondence is determined by the fixtures rather
    than guessed: this is the same bridge the fixture pot join and the parser
    validation use. Returns ``(wikipedia name -> openfootball name, problems)``.
    """
    from ucl.data.fixtures import bridge_names

    def shaped(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["home", "away", "home_country", "away_country"]].copy()
        # A constant key: a club is identified by its set of (venue, opponent).
        out["matchday"] = 0
        return out

    mapping: dict[str, str] = {}
    problems: list[str] = []
    for season_label, competition in pairs:
        pages = [
            p for p in wikipedia.PAGES
            if p.season == season_label and p.competition == competition
        ]
        frames = [
            wikipedia.parse_matches(
                (wikipedia.CACHE_DIR / p.file_name).read_text(encoding="utf-8"),
                season=p.season, competition=p.competition, source_file=p.file_name,
            )
            for p in pages
            if (wikipedia.CACHE_DIR / p.file_name).exists()
        ]
        if not frames:
            problems.append(f"{season_label} {competition}: no cached Wikipedia pages")
            continue
        wiki = pd.concat(frames, ignore_index=True)
        stem = "el" if competition == "UEL" else "conf"
        path = openfootball.CACHE_DIR / f"{season_label}_{stem}.txt"
        if not path.exists():
            problems.append(f"{season_label} {competition}: {path.name} not cached")
            continue
        of = openfootball.parse_file(path, season=season_label, competition=competition)
        resolved, issues = bridge_names(shaped(wiki), shaped(of))
        for name, target in resolved.items():
            existing = mapping.setdefault(name, target)
            if existing != target:
                problems.append(
                    f"{name!r} bridges to {existing!r} and {target!r} in different competitions"
                )
        problems.extend(f"{season_label} {competition}: {issue}" for issue in issues)
    return mapping, problems
