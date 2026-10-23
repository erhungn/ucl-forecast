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

__all__ = ["MODELS", "CANDIDATE_LEVELS", "GAP_COLUMNS", "gap_covariates",
           "fit_v1", "fit_v2", "fit_euro_scale", "fit"]

#: Every model this code can build, newest last. Not the same list as
#: :data:`ucl.config.MODEL_HISTORY`, which is what has been *published*: a
#: model is buildable here first, so that it can be fitted, simulated and
#: rehearsed before anything starts publishing it. "v3" is v2 plus candidate
#: C's per-competition European terms, and is the same fit the candidate was
#: accepted on - "v2+C" is accepted as an alias so the archived candidate's
#: name keeps working and its decision files stay readable.
MODELS = ("elo", "v1", "v2", "v3")

#: Candidate C's competition levels. Domestic is the baseline and gets no
#: intercept of its own, so naming it here would make the three indicators
#: collinear with `a` and the fit unidentified.
CANDIDATE_LEVELS = ("ucl_main", "other_european")

#: The GAP covariate columns, fixed so the fitted coefficients read back by name.
GAP_COLUMNS = tuple(f"d_gap_{measure}" for measure in gap.MEASURES)


def gap_covariates(residuals: dict[str, dict]) -> tuple[ClubCovariate, ...]:
    """One covariate per attacking measure present in ``residuals``."""
    out = []
    for measure, column in zip(gap.MEASURES, GAP_COLUMNS):
        if measure not in residuals:
            continue
        rated = residuals[measure]
        out.append(ClubCovariate(
            column,
            # Default argument binds this measure's table, not the loop's last.
            lambda home, away, rated=rated: gap.match_difference(rated, home, away),
            lambda club, rated=rated: club in rated,
        ))
    return tuple(out)


def gap_residuals(as_of: date, elo_ratings: dict[str, float],
                  measures: tuple[str, ...] | None = None) -> tuple[dict, dict]:
    """GAP residual ratings per measure, and the parameters they were built on.

    Everything is fitted at ``as_of``: the shot ratings are replayed over
    matches before it, their learning rate and phis are fitted by least squares
    against observed attacking output, and the per-league regression on Elo
    uses the ratings as they stand then.
    """
    shots = gap.domestic_frame(as_of)
    residuals, parameters = {}, {}
    for measure in (measures or tuple(gap.MEASURES)):
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
           value_as_of=None, measures: tuple[str, ...] | None = None
           ) -> ForecastModel:
    """Elo, the squad-value prior, and the GAP shot-rating residuals.

    ``measures`` restricts which attacking measures are used; the default is
    all three, which is the published model. Anything else is an ablation and
    is not promoted by running it.
    """
    residuals, parameters = gap_residuals(as_of, elo_ratings, measures)
    covariates = (value_covariate(values), *gap_covariates(residuals))
    return fit_model(pre_match, covariates, label="v2", as_of=value_as_of,
                     details={"values": values, "residuals": residuals,
                              "gap_params": parameters, "gap_as_of": as_of})


def fit_euro_scale(pre_match: pd.DataFrame, values: dict[str, float],
                   elo_ratings: dict[str, float], *, as_of: date,
                   value_as_of=None, measures: tuple[str, ...] | None = None,
                   label: str = "v2+C") -> ForecastModel:
    """Candidate C: model v2 plus a per-competition goal scale and home advantage.

    Identical to :func:`fit_v2` in every covariate, so the comparison isolates
    the four new intercept terms. Each level contributes a scale added to both
    log-rates and an extra home advantage added to the home side only; domestic
    is the baseline and carries neither.

    The motivation is in `docs/METHODOLOGY.md` section 3: 9.8% of the fitting
    sample is European, so `a` is set by domestic football at 2.78 goals a
    match while the UCL main phase scores 3.13.

    ``label`` names the fit: "v2+C" is the candidate, "v3" the published model.
    Nothing else differs - the same covariates, the same levels, the same
    starting point - so a v3 forecast and candidate C's accepted fit are the
    same arithmetic at the same cutoff, and a test asserts it.
    """
    residuals, parameters = gap_residuals(as_of, elo_ratings, measures)
    covariates = (value_covariate(values), *gap_covariates(residuals))
    return fit_model(pre_match, covariates, label=label, as_of=value_as_of,
                     levels=CANDIDATE_LEVELS, level="ucl_main",
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
    if label in ("v3", "v2+C"):
        # One fit, two names: "v3" is the published name, "v2+C" the name the
        # candidate was accepted under. The label is carried through so an
        # artefact says which of the two produced it.
        return fit_euro_scale(pre_match, values, elo_ratings, as_of=as_of,
                              value_as_of=value_as_of, label=label)
    raise ValueError(f"unknown model {label!r}; have {MODELS}")
