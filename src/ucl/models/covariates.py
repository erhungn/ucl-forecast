"""The baseline Poisson match model extended with extra covariates.

The accepted model is ``log λ_home = a + h + b·d_elo`` and
``log λ_away = a − b·d_elo``. A candidate information source adds terms:

* **symmetric** covariates act like the Elo difference - one value per match,
  added to the home rate and subtracted from the away rate. The market-implied
  rating difference is one of these.
* **per-side** covariates act on whichever team they describe, added to that
  team's own rate. Rest days are one of these: congestion tires one team, not
  both.

Setting a covariate's column to zero reproduces the baseline exactly, which is
how an unrated club falls back and how the tests pin the nesting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist

from ucl import config

__all__ = ["LEVELS", "CovariateModel", "fit_covariate_model", "ClubCovariate",
           "ForecastModel", "value_covariate", "fit_model",
           "fit_squad_value_model"]


@dataclass
class CovariateModel:
    a: float
    h: float
    b: float
    symmetric: dict[str, float] = field(default_factory=dict)
    per_side: dict[str, float] = field(default_factory=dict)
    #: Per-competition intercept shifts, keyed by level name. ``scale`` is added
    #: to BOTH log-rates, so it is a goal scale: a factor of e**c on every rate.
    #: ``home_extra`` is added to the home side only and is switched off by
    #: ``neutral`` exactly as ``h`` is - it is extra home advantage, and the
    #: neutral-venue final must not receive it.
    scale: dict[str, float] = field(default_factory=dict)
    home_extra: dict[str, float] = field(default_factory=dict)
    matches: int = 0
    converged: bool = True
    log_likelihood: float = float("nan")

    def rates(self, elo_home, elo_away, *, neutral: bool = False,
              symmetric: dict | None = None, per_side_home: dict | None = None,
              per_side_away: dict | None = None, levels: dict | None = None):
        if (self.scale or self.home_extra) and levels is None:
            raise TypeError(
                "this model carries per-competition intercepts "
                f"({sorted(set(self.scale) | set(self.home_extra))}) and cannot "
                "produce rates without knowing the competition; pass "
                "levels={'<level>': 1.0, ...}. Defaulting to zero would score a "
                "UCL match at the domestic goal scale and nothing downstream "
                "would look wrong."
            )
        d = (np.asarray(elo_home, dtype=float) - np.asarray(elo_away, dtype=float)) / 400.0
        home_term = self.a + (0.0 if neutral else self.h) + self.b * d
        away_term = self.a - self.b * d
        for name, coefficient in self.symmetric.items():
            value = np.asarray((symmetric or {}).get(name, 0.0), dtype=float)
            home_term = home_term + coefficient * value
            away_term = away_term - coefficient * value
        for name, coefficient in self.per_side.items():
            home_term = home_term + coefficient * np.asarray(
                (per_side_home or {}).get(name, 0.0), dtype=float)
            away_term = away_term + coefficient * np.asarray(
                (per_side_away or {}).get(name, 0.0), dtype=float)
        for name, coefficient in self.scale.items():
            value = np.asarray((levels or {}).get(name, 0.0), dtype=float)
            home_term = home_term + coefficient * value
            away_term = away_term + coefficient * value
        for name, coefficient in self.home_extra.items():
            value = np.asarray((levels or {}).get(name, 0.0), dtype=float)
            # Written as the same idiom as `h` above on purpose: one switch, so
            # it cannot come adrift from the thing it has to agree with.
            home_term = home_term + (0.0 if neutral else coefficient) * value
        return np.exp(home_term), np.exp(away_term)

    def as_dict(self) -> dict:
        out = {"a": self.a, "h": self.h, "b": self.b,
               **{f"b_{k}": v for k, v in self.symmetric.items()},
               **{f"c_{k}": v for k, v in self.per_side.items()},
               **{f"c_{k}": v for k, v in self.scale.items()},
               **{f"g_{k}": v for k, v in self.home_extra.items()}}
        expected = (3 + len(self.symmetric) + len(self.per_side)
                    + len(self.scale) + len(self.home_extra))
        if len(out) != expected:
            raise ValueError(
                "two coefficients publish the same key: `c_` is shared by "
                "per-side covariates and competition scales, so a name used "
                "for both would silently overwrite one. Rename one of "
                f"{sorted(set(self.per_side) & set(self.scale))}."
            )
        return out


def fit_covariate_model(
    frame: pd.DataFrame,
    *,
    symmetric: tuple[str, ...] = (),
    per_side: tuple[str, ...] = (),
    levels: tuple[str, ...] = (),
) -> CovariateModel:
    """Poisson MLE on pre-match ratings plus the named covariate columns.

    ``symmetric`` names columns holding one value per match; ``per_side`` names
    pairs, expected as ``<name>_home`` and ``<name>_away``.

    ``levels`` names 0/1 indicator columns, one per competition level, and each
    contributes **two** parameters: a goal scale added to both log-rates and an
    extra home advantage added to the home side. The parameter vector appends
    them after the existing blocks, so a fit with ``levels=()`` is the same
    optimisation over the same vector as before - which is what lets the
    archived models keep reproducing.
    """
    needed = ["elo_home", "elo_away", "hg", "ag", *symmetric]
    needed += [f"{n}_{side}" for n in per_side for side in ("home", "away")]
    needed += list(levels)
    data = frame.dropna(subset=needed)
    d = (data.elo_home.to_numpy(float) - data.elo_away.to_numpy(float)) / 400.0
    hg = data.hg.to_numpy(float)
    ag = data.ag.to_numpy(float)
    sym = [data[name].to_numpy(float) for name in symmetric]
    side_home = [data[f"{name}_home"].to_numpy(float) for name in per_side]
    side_away = [data[f"{name}_away"].to_numpy(float) for name in per_side]
    lev = [data[name].to_numpy(float) for name in levels]

    n_sym, n_side, n_lev = len(symmetric), len(per_side), len(levels)
    at_side = 3 + n_sym
    at_scale = at_side + n_side
    at_extra = at_scale + n_lev

    def objective(theta):
        a, h, b = theta[0], theta[1], theta[2]
        sym_c = theta[3:at_side]
        side_c = theta[at_side:at_scale]
        scale_c = theta[at_scale:at_extra]
        extra_c = theta[at_extra:]
        log_h = a + h + b * d
        log_a = a - b * d
        for value, coefficient in zip(sym, sym_c):
            log_h = log_h + coefficient * value
            log_a = log_a - coefficient * value
        for home_value, away_value, coefficient in zip(side_home, side_away, side_c):
            log_h = log_h + coefficient * home_value
            log_a = log_a + coefficient * away_value
        for value, coefficient in zip(lev, scale_c):
            log_h = log_h + coefficient * value
            log_a = log_a + coefficient * value
        for value, coefficient in zip(lev, extra_c):
            log_h = log_h + coefficient * value
        rate_h, rate_a = np.exp(log_h), np.exp(log_a)
        value = np.sum(rate_h - hg * log_h) + np.sum(rate_a - ag * log_a)

        residual_h, residual_a = rate_h - hg, rate_a - ag
        gradient = np.zeros_like(theta)
        gradient[0] = residual_h.sum() + residual_a.sum()
        gradient[1] = residual_h.sum()
        gradient[2] = np.sum(d * (residual_h - residual_a))
        for i, column in enumerate(sym):
            gradient[3 + i] = np.sum(column * (residual_h - residual_a))
        for i, (home_value, away_value) in enumerate(zip(side_home, side_away)):
            gradient[3 + n_sym + i] = np.sum(
                home_value * residual_h + away_value * residual_a)
        for i, column in enumerate(lev):
            # The scale sits on both sides, so both residuals enter with the
            # same sign; the home extra sees only the home residual.
            gradient[at_scale + i] = np.sum(column * (residual_h + residual_a))
            gradient[at_extra + i] = np.sum(column * residual_h)
        return value, gradient

    start = np.zeros(3 + n_sym + n_side + 2 * n_lev)
    start[0] = config.POISSON_FALLBACK["a"]
    start[1] = config.POISSON_FALLBACK["h"]
    start[2] = config.POISSON_FALLBACK["b"]
    result = minimize(objective, start, jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-12, "gtol": 1e-10})
    a, h, b = result.x[0], result.x[1], result.x[2]
    sym_c = dict(zip(symmetric, result.x[3:at_side]))
    side_c = dict(zip(per_side, result.x[at_side:at_scale]))
    scale_c = dict(zip(levels, result.x[at_scale:at_extra]))
    extra_c = dict(zip(levels, result.x[at_extra:]))

    log_h = a + h + b * d
    log_a = a - b * d
    for value, name in zip(sym, symmetric):
        log_h = log_h + sym_c[name] * value
        log_a = log_a - sym_c[name] * value
    for home_value, away_value, name in zip(side_home, side_away, per_side):
        log_h = log_h + side_c[name] * home_value
        log_a = log_a + side_c[name] * away_value
    for value, name in zip(lev, levels):
        log_h = log_h + scale_c[name] * value + extra_c[name] * value
        log_a = log_a + scale_c[name] * value
    full = float(np.sum(poisson_dist.logpmf(hg, np.exp(log_h)))
                 + np.sum(poisson_dist.logpmf(ag, np.exp(log_a))))
    return CovariateModel(
        a=float(a), h=float(h), b=float(b),
        symmetric={k: float(v) for k, v in sym_c.items()},
        per_side={k: float(v) for k, v in side_c.items()},
        scale={k: float(v) for k, v in scale_c.items()},
        home_extra={k: float(v) for k, v in extra_c.items()},
        matches=len(data), converged=bool(result.success), log_likelihood=full,
    )


#: Every competition level the model knows. A typo must raise rather than
#: silently become the baseline, so the set is closed.
LEVELS: tuple[str, ...] = ("domestic", "ucl_main", "other_european")


@dataclass(frozen=True)
class ClubCovariate:
    """One covariate the simulator must look up by club id.

    ``difference`` turns a pair of club ids into the per-match value, zero
    wherever either club is uncovered - the convention every accepted candidate
    was scored under, so the forecast is never better informed than the test
    that accepted it. ``covers`` answers whether a single club has the data.
    """

    name: str
    difference: Callable[[Any, Any], np.ndarray]
    covers: Callable[[Any], bool]


@dataclass(frozen=True)
class ForecastModel:
    """The Poisson match model with any number of club-keyed covariates.

    Model v1 is this with one covariate (squad value); model v2 adds the three
    GAP shot-rating residuals. The simulator works in club ids, so the
    covariates are looked up per match rather than passed in as columns:
    :meth:`rates_for` takes the two clubs.

    ``details`` carries whatever the builder wants to publish alongside - the
    value snapshot, the GAP residuals, the fitted rating parameters - so the
    caller does not have to thread them separately.
    """

    model: CovariateModel
    covariates: tuple[ClubCovariate, ...]
    label: str = "v1"
    as_of: object = None
    details: dict = field(default_factory=dict)
    #: Which competition this model forecasts, when the fitted model carries
    #: per-competition intercepts. Required in that case and refused otherwise:
    #: a level that defaulted to "domestic" would publish a 2.8-goal UCL
    #: forecast instead of a 3.3-goal one, and every check the simulator prints
    #: - probabilities summing to one, the 8/16/12 band counts, monotone stage
    #: probabilities - would still pass.
    level: str | None = None

    def __post_init__(self) -> None:
        carries = bool(self.model.scale or self.model.home_extra)
        if carries and self.level is None:
            raise TypeError(
                f"{type(self).__name__} was fitted with per-competition "
                f"intercepts {sorted(set(self.model.scale) | set(self.model.home_extra))} "
                "but no level was declared; pass level='ucl_main' for a "
                "Champions League forecast")
        if self.level is not None and self.level not in LEVELS:
            raise ValueError(f"unknown competition level {self.level!r}; "
                             f"expected one of {LEVELS}")

    @property
    def a(self) -> float:
        return self.model.a

    @property
    def h(self) -> float:
        return self.model.h

    @property
    def b(self) -> float:
        return self.model.b

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(covariate.name for covariate in self.covariates)

    def differences(self, home, away) -> dict[str, np.ndarray]:
        return {covariate.name: covariate.difference(home, away)
                for covariate in self.covariates}

    def rates_for(self, home, away, elo_home, elo_away, *, neutral: bool = False):
        """Expected goals for (home, away), every covariate included.

        The competition level comes from ``self.level``, so no caller changes:
        every match the simulator draws belongs to the competition the model was
        built to forecast. Only the backtest scores more than one competition,
        and it goes through :meth:`CovariateModel.rates` with a per-row level.
        """
        scalar = np.ndim(elo_home) == 0 and np.ndim(home) == 0
        symmetric = {
            name: (value[0] if scalar else value)
            for name, value in self.differences(home, away).items()
        }
        return self.model.rates(elo_home, elo_away, neutral=neutral,
                                symmetric=symmetric, levels=self.levels())

    def levels(self) -> dict[str, float] | None:
        """The one-hot level dict, or None when the model carries no levels."""
        if not (self.model.scale or self.model.home_extra):
            return None
        return {name: float(name == self.level) for name in LEVELS}

    def rates(self, elo_home, elo_away, *, neutral: bool = False):
        """Refused: without the club ids the covariates cannot be evaluated.

        Silently dropping them would make the forecast quietly Elo-only, which
        is the kind of bug that cost two re-runs already.
        """
        raise TypeError(
            f"{type(self).__name__} needs the club ids; "
            "call rates_for(home, away, ...)")

    def as_dict(self) -> dict:
        return self.model.as_dict()

    def coverage(self, home, away) -> dict[str, float]:
        """Share of the given fixtures each covariate actually reaches."""
        home, away = np.atleast_1d(home), np.atleast_1d(away)
        out = {}
        for covariate in self.covariates:
            known = np.array([covariate.covers(club) for club in home])
            known_away = np.array([covariate.covers(club) for club in away])
            out[covariate.name] = float(np.mean(known & known_away))
        return out


def value_covariate(values: dict[str, float]) -> ClubCovariate:
    """The squad-value covariate: log(value_home) - log(value_away)."""
    def difference(home, away) -> np.ndarray:
        home_value = np.array([values.get(c, np.nan) for c in np.atleast_1d(home)])
        away_value = np.array([values.get(c, np.nan) for c in np.atleast_1d(away)])
        known = ~np.isnan(home_value) & ~np.isnan(away_value)
        out = np.zeros(len(home_value))
        out[known] = np.log(home_value[known]) - np.log(away_value[known])
        return out

    return ClubCovariate("d_value", difference, lambda club: club in values)


def fit_model(pre_match: pd.DataFrame, covariates: tuple[ClubCovariate, ...],
              *, label: str = "v1", as_of=None, details: dict | None = None,
              levels: tuple[str, ...] = (), level: str | None = None
              ) -> ForecastModel:
    """Fit the Poisson model with these covariates on pre-match ratings.

    Each covariate is evaluated on the training frame, then all of them are
    fitted jointly by MLE. Uncovered matches carry a zero rather than a NaN, so
    they stay in the fit instead of being dropped from it.

    ``levels`` names the competition levels to give their own intercepts, read
    from the frame's ``level`` column as one-hot indicators; ``level`` is the
    competition the resulting model forecasts. Both default to off, so every
    published model fits exactly the vector it always did.
    """
    frame = pre_match.copy()
    home = frame.home_id.to_numpy()
    away = frame.away_id.to_numpy()
    for covariate in covariates:
        frame[covariate.name] = covariate.difference(home, away)
    names = tuple(covariate.name for covariate in covariates)
    if levels:
        if "level" not in frame:
            raise KeyError(
                "fitting per-competition intercepts needs a `level` column on "
                "the training frame; ucl.models.elo.match_frame provides it")
        unknown = set(frame.level.unique()) - set(LEVELS)
        if unknown:
            raise ValueError(f"unknown competition level(s) {sorted(unknown)}")
        for name in levels:
            frame[name] = (frame.level == name).astype(float)
    fitted = fit_covariate_model(frame, symmetric=names, levels=levels)
    return ForecastModel(fitted, tuple(covariates), label=label, as_of=as_of,
                         details=details or {}, level=level)


def fit_squad_value_model(pre_match: pd.DataFrame, values: dict[str, float],
                          *, as_of=None) -> ForecastModel:
    """Model v1: Elo plus the pre-season squad-value prior, and nothing else.

    Kept as its own entry point because v1 is still the published forecast and
    the frozen archive it has to keep reproducing.
    """
    return fit_model(pre_match, (value_covariate(values),), label="v1",
                     as_of=as_of, details={"values": values})
