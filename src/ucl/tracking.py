"""Prospective comparison of the candidate models on this season's matches.

The backtest decides; this does not. The forward chain accepted the shot
ratings and rejected minutes-weighting, and that is what promoted v2.

**This season cannot resolve v1 against v2, and is not a gate.** The paired
standard error after matchday 1 is 0.004, and over all 144 league-phase matches
it will be roughly 0.0014 - still about three times the 0.0004 the five-season
backtest measured. It is monitoring: it would catch something badly wrong, and
it accumulates, which is only worth anything if it is recorded as the season
goes rather than reconstructed afterwards.

Every model's probabilities are computed from the **same frozen information
set**: the 2026-09-07 cutoff, the day before matchday 1. No model sees a result
it is then scored on, and none sees more than the others, so the comparison is
between the models and not between their information.

Nothing here promotes anything. The published forecast stays model v1 until
that is a separate, declared decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ucl.config import SEASONS
from ucl.evaluate import log_loss, rps

__all__ = ["MODELS", "REFERENCE", "UNIFORM", "probabilities", "track",
           "running", "summary"]

#: The models being followed, and what each adds to the one before. The order
#: is the order they were accepted or rejected in.
MODELS: dict[str, str] = {
    "v2": "Elo + association offsets + squad value + GAP shot ratings (published)",
    "v1": "Elo + association offsets + squad value (the previous published model)",
    "v2+B": "v2 with minutes-weighted squad value instead of the total (rejected)",
}
#: The model every other is compared against: whatever is published.
REFERENCE = "v2"
UNIFORM = "uniform"


def _fitted(season_label: str, model: str):
    """The fitted model and the pieces needed to evaluate it, at the cutoff."""
    from ucl.backtest import GAP_COLUMNS, _fitted_gap, _fitted_squad_value

    cutoff = SEASONS[season_label].cutoff_date
    if model == "v1":
        ratings, values, fitted = _fitted_squad_value(cutoff, season_label)
        return ratings, fitted, values, None
    weighting = "minutes" if model == "v2+B" else "total"
    ratings, values, residuals, _, fitted = _fitted_gap(
        cutoff, season_label, weighting)
    return ratings, fitted, values, residuals


def probabilities(season_label: str, home, away, model: str) -> np.ndarray:
    """``(n, 3)`` home/draw/away probabilities for a model at the cutoff.

    Returns rows of NaN for a fixture whose clubs the Elo replay has never
    rated, which cannot happen for the league phase but keeps the function
    honest if it is ever pointed at something else.
    """
    from ucl.backtest import GAP_COLUMNS, probabilities_from_rates
    from ucl.data import transfermarkt
    from ucl.models import gap as gap_model

    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; have {sorted(MODELS)}")
    ratings, fitted, values, residuals = _fitted(season_label, model)
    home = np.asarray(home, dtype=object)
    away = np.asarray(away, dtype=object)

    known = np.array([one in ratings.ratings and other in ratings.ratings
                      for one, other in zip(home, away)])
    out = np.full((len(home), 3), np.nan)
    if not known.any():
        return out

    elo_home = np.array([ratings.ratings[club] for club in home[known]], dtype=float)
    elo_away = np.array([ratings.ratings[club] for club in away[known]], dtype=float)
    symmetric = {"d_value": transfermarkt.value_difference(
        values, home[known], away[known])}
    if residuals is not None:
        for measure, column in zip(gap_model.MEASURES, GAP_COLUMNS):
            symmetric[column] = gap_model.match_difference(
                residuals[measure], home[known], away[known])
    rate_home, rate_away = fitted.rates(elo_home, elo_away, symmetric=symmetric)
    out[known] = probabilities_from_rates(rate_home, rate_away)
    return out


def track(season_label: str = "2026-27", played: pd.DataFrame | None = None
          ) -> pd.DataFrame:
    """One row per played match per model, with its score.

    ``played`` defaults to the league-phase matches recorded so far.
    """
    from ucl.data import league_results

    if played is None:
        played = league_results.load_played(season_label)
    if not len(played):
        return pd.DataFrame(columns=["matchday", "date", "home", "away", "model"])

    outcomes = np.where(played.hg > played.ag, 0,
                        np.where(played.hg == played.ag, 1, 2))
    home = played.home_id.to_numpy()
    away = played.away_id.to_numpy()
    rows = []
    for model in [*MODELS, UNIFORM]:
        if model == UNIFORM:
            chances = np.full((len(played), 3), 1 / 3)
        else:
            chances = probabilities(season_label, home, away, model)
        actual = chances[np.arange(len(played)), outcomes]
        rows.append(pd.DataFrame({
            "matchday": played.matchday.to_numpy(),
            "date": played.date.dt.date.to_numpy(),
            "home": home, "away": away,
            "score": [f"{h}-{a}" for h, a in zip(played.hg, played.ag)],
            "result": [("home", "draw", "away")[code] for code in outcomes],
            "model": model,
            "p_home": chances[:, 0], "p_draw": chances[:, 1],
            "p_away": chances[:, 2], "p_actual": actual,
            "log_loss": -np.log(np.clip(actual, 1e-15, 1.0)),
            "rps": [rps(chances[index:index + 1], outcomes[index:index + 1])
                    for index in range(len(played))],
        }))
    return pd.concat(rows, ignore_index=True)


def running(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-matchday and cumulative scores for each model."""
    if not len(frame):
        return pd.DataFrame(columns=["matchday", "model", "n", "log_loss", "rps"])
    rows = []
    for model, group in frame.groupby("model"):
        for matchday in sorted(group.matchday.unique()):
            here = group[group.matchday == matchday]
            so_far = group[group.matchday <= matchday]
            rows.append({
                "matchday": int(matchday), "model": model, "n": len(here),
                "log_loss": float(here.log_loss.mean()),
                "rps": float(here.rps.mean()),
                "cumulative_n": len(so_far),
                "cumulative_log_loss": float(so_far.log_loss.mean()),
                "cumulative_rps": float(so_far.rps.mean()),
            })
    return pd.DataFrame(rows).sort_values(["matchday", "model"]).reset_index(drop=True)


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per model over everything played, with the gap to v1.

    The standard error is on the *paired* difference, not on each model's own
    log-loss: the models see the same matches, so what matters is how much
    their per-match losses differ, and that is far smaller than either level.
    Even so it cannot resolve the difference the backtest measured - see the
    module docstring.
    """
    if not len(frame):
        return pd.DataFrame(columns=["model", "n", "log_loss", "rps"])
    wide = frame.pivot_table(index=["matchday", "home", "away"],
                             columns="model", values="log_loss")
    rows = []
    for model in [*MODELS, UNIFORM]:
        group = frame[frame.model == model]
        if not len(group):
            continue
        row = {"model": model, "n": len(group),
               "log_loss": float(group.log_loss.mean()),
               "rps": float(group.rps.mean())}
        if model != REFERENCE and REFERENCE in wide.columns and model in wide.columns:
            paired = (wide[model] - wide[REFERENCE]).dropna()
            row["vs_reference"] = float(paired.mean())
            row["vs_reference_standard_error"] = (
                float(paired.std(ddof=1) / np.sqrt(len(paired)))
                if len(paired) > 1 else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)
