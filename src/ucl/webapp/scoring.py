"""Scoring the played matchdays, in the two senses that differ.

**Frozen at cutoff** is what ``scripts/evaluate_matchday.py`` produces: the
pre-tournament probabilities, fitted at the season's cutoff, scored on matches
played since. It never changes and it uses no hindsight at all.

**Pre-matchday** is the table ``simulate.py`` saved the week before each round,
whose ratings include every earlier matchday. It is what the model actually said
immediately before those matches.

They coincide for matchday 1 - both are fitted at the cutoff - which is the
equivalence check that the saved tables are what they claim to be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ucl.evaluate import log_loss, rps
from ucl.webapp import loaders, paths

__all__ = [
    "OUTCOMES", "outcome_codes", "frozen_series", "pre_matchday_series",
    "cumulative", "summarise",
]

#: 1X2 is ordered home > draw > away, which is what the RPS assumes.
OUTCOMES = ("home", "draw", "away")


def outcome_codes(frame: pd.DataFrame) -> np.ndarray:
    """Map the ``result`` column onto 0/1/2."""
    lookup = {name: index for index, name in enumerate(OUTCOMES)}
    return frame.result.map(lookup).to_numpy(dtype=int)


def frozen_series(season: str = paths.SEASON, matchdays=None,
                  **kwargs) -> pd.DataFrame:
    """Every published evaluation, stacked, with its matchday."""
    if matchdays is None:
        matchdays = paths.scored_matchdays(season, **kwargs)
    pieces = []
    for matchday in matchdays:
        frame = loaders.load_evaluation(matchday, season, **kwargs)
        frame.insert(0, "matchday", matchday)
        frame["series"] = "frozen at cutoff"
        pieces.append(frame)
    if not pieces:
        return pd.DataFrame(columns=["matchday", "home", "away", "series"])
    return pd.concat(pieces, ignore_index=True)


def pre_matchday_series(season: str = paths.SEASON, matchdays=None,
                        **kwargs) -> pd.DataFrame:
    """The saved pre-match tables joined to what actually happened.

    The join is on the club pair, never the date: a rescheduled match carries
    one date in the fixture list and another in the played table.
    """
    if matchdays is None:
        matchdays = paths.scored_matchdays(season, **kwargs)
    pieces = []
    for matchday in matchdays:
        try:
            forecast = loaders.load_next_matchday(matchday, season, **kwargs)
        except FileNotFoundError:
            # No table was saved before this round - a backfill gap, not an error.
            continue
        actual = loaders.load_evaluation(matchday, season, **kwargs)
        merged = forecast.merge(
            actual[["home", "away", "score", "result"]], on=["home", "away"],
            how="inner")
        if not len(merged):
            continue
        probabilities = merged[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        codes = outcome_codes(merged)
        merged["p_actual"] = probabilities[np.arange(len(merged)), codes]
        merged["log_loss"] = -np.log(np.clip(merged.p_actual, 1e-15, 1.0))
        merged["rps"] = [
            rps(probabilities[i:i + 1], codes[i:i + 1]) for i in range(len(merged))
        ]
        merged["series"] = "pre-matchday"
        pieces.append(merged)
    if not pieces:
        return pd.DataFrame(columns=["matchday", "home", "away", "series"])
    return pd.concat(pieces, ignore_index=True)


def summarise(frame: pd.DataFrame) -> dict:
    """Aggregate scores for one series, with the uniform baseline beside them.

    The model's log-loss and RPS are the means of the per-match columns, which
    is their definition; the uniform baseline needs the outcomes, so it is
    computed from ``result``.
    """
    if not len(frame):
        return {"n": 0}
    codes = outcome_codes(frame)
    uniform = np.full((len(frame), 3), 1 / 3)
    per_match = frame.log_loss.to_numpy(dtype=float)
    return {
        "n": int(len(frame)),
        "log_loss": float(per_match.mean()),
        "rps": float(frame.rps.mean()),
        "uniform_log_loss": log_loss(uniform, codes),
        "uniform_rps": rps(uniform, codes),
        # The same standard error scripts/evaluate_matchday.py reports: on 18
        # matches it is ~0.09, about a hundred times the differences the
        # five-season backtest was built to resolve.
        "standard_error": (float(per_match.std(ddof=1) / np.sqrt(len(per_match)))
                           if len(per_match) > 1 else float("nan")),
    }


def cumulative(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-matchday and running scores for one series."""
    if not len(frame):
        return pd.DataFrame(columns=["matchday", "n", "log_loss", "rps"])
    rows = []
    ordered = sorted(frame.matchday.unique())
    for matchday in ordered:
        this = frame[frame.matchday == matchday]
        so_far = frame[frame.matchday <= matchday]
        here, running = summarise(this), summarise(so_far)
        rows.append({
            "matchday": int(matchday), "n": here["n"],
            "log_loss": here["log_loss"], "rps": here["rps"],
            "cumulative_n": running["n"],
            "cumulative_log_loss": running["log_loss"],
            "cumulative_rps": running["rps"],
            "uniform_log_loss": running["uniform_log_loss"],
            "uniform_rps": running["uniform_rps"],
        })
    return pd.DataFrame(rows)
