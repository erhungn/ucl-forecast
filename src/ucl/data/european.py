"""All European club results, from openfootball plus the Wikipedia gap fills.

openfootball is the primary source. It does not publish the 2025-26 Europa and
Conference League main phases, nor anything for 2026-27, so those come from
Wikipedia, whose parser is validated against openfootball where they overlap
(see scripts/validate_wikipedia.py). Only Wikipedia's "gap" pages are loaded,
so no match is counted twice.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ucl.data import openfootball, wikipedia

__all__ = ["load_results", "coverage"]


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
