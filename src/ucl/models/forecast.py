"""Assembling the published models, in one place.

``v1`` is Elo plus the pre-season squad-value prior. ``v2`` adds the three GAP
shot-rating residuals, accepted by the forward chain in milestone 8 (pooled
delta -0.00042, cluster CI [-0.00076, -0.00008], all four folds negative).

Both are built here rather than in the scripts, so the backtest, the simulator
and the matchday evaluation cannot drift into assembling the same model three
slightly different ways - which is how a comparison between two models stops
measuring the models.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ucl.models import gap
from ucl.models.covariates import (
    ClubCovariate, ForecastModel, fit_model, value_covariate,
)

__all__ = ["MODELS", "GAP_COLUMNS", "gap_covariates", "fit_v1", "fit_v2", "fit"]

#: The published models, newest last.
MODELS = ("elo", "v1", "v2")

#: The GAP covariate columns, fixed so the fitted coefficients read back by name.
GAP_COLUMNS = tuple(f"d_gap_{measure}" for measure in gap.MEASURES)


def gap_covariates(residuals: dict[str, dict]) -> tuple[ClubCovariate, ...]:
    """One covariate per attacking measure, from residual ratings."""
    out = []
    for measure, column in zip(gap.MEASURES, GAP_COLUMNS):
        rated = residuals[measure]
        out.append(ClubCovariate(
            column,
            # Default argument binds this measure's table, not the loop's last.
            lambda home, away, rated=rated: gap.match_difference(rated, home, away),
            lambda club, rated=rated: club in rated,
        ))
    return tuple(out)


def gap_residuals(as_of: date, elo_ratings: dict[str, float]) -> tuple[dict, dict]:
    """GAP residual ratings per measure, and the parameters they were built on.

    Everything is fitted at ``as_of``: the shot ratings are replayed over
    matches before it, their learning rate and phis are fitted by least squares
    against observed attacking output, and the per-league regression on Elo
    uses the ratings as they stand then.
    """
    shots = gap.domestic_frame(as_of)
    residuals, parameters = {}, {}
    for measure in gap.MEASURES:
        parameters[measure] = gap.fit_params(shots, measure)
        state = gap.replay(shots, measure, parameters[measure])
        residuals[measure] = gap.residual_ratings(state, elo_ratings)
    return residuals, parameters


def fit_v1(pre_match: pd.DataFrame, values: dict[str, float], *,
           as_of=None) -> ForecastModel:
    """Elo plus the squad-value prior."""
    return fit_model(pre_match, (value_covariate(values),), label="v1",
                     as_of=as_of, details={"values": values})


def fit_v2(pre_match: pd.DataFrame, values: dict[str, float],
           elo_ratings: dict[str, float], *, as_of: date,
           value_as_of=None) -> ForecastModel:
    """Elo, the squad-value prior, and the three GAP shot-rating residuals."""
    residuals, parameters = gap_residuals(as_of, elo_ratings)
    covariates = (value_covariate(values), *gap_covariates(residuals))
    return fit_model(pre_match, covariates, label="v2", as_of=value_as_of,
                     details={"values": values, "residuals": residuals,
                              "gap_params": parameters, "gap_as_of": as_of})


def fit(label: str, pre_match: pd.DataFrame, values: dict[str, float],
        elo_ratings: dict[str, float], *, as_of: date,
        value_as_of=None) -> ForecastModel:
    """Build whichever published model ``label`` names."""
    if label == "v1":
        return fit_v1(pre_match, values, as_of=value_as_of)
    if label == "v2":
        return fit_v2(pre_match, values, elo_ratings, as_of=as_of,
                      value_as_of=value_as_of)
    raise ValueError(f"unknown model {label!r}; have {MODELS}")
