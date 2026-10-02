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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist

from ucl import config

__all__ = ["CovariateModel", "fit_covariate_model", "SquadValueModel",
           "fit_squad_value_model"]


@dataclass
class CovariateModel:
    a: float
    h: float
    b: float
    symmetric: dict[str, float] = field(default_factory=dict)
    per_side: dict[str, float] = field(default_factory=dict)
    matches: int = 0
    converged: bool = True
    log_likelihood: float = float("nan")

    def rates(self, elo_home, elo_away, *, neutral: bool = False,
              symmetric: dict | None = None, per_side_home: dict | None = None,
              per_side_away: dict | None = None):
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
        return np.exp(home_term), np.exp(away_term)

    def as_dict(self) -> dict:
        return {"a": self.a, "h": self.h, "b": self.b,
                **{f"b_{k}": v for k, v in self.symmetric.items()},
                **{f"c_{k}": v for k, v in self.per_side.items()}}


def fit_covariate_model(
    frame: pd.DataFrame,
    *,
    symmetric: tuple[str, ...] = (),
    per_side: tuple[str, ...] = (),
) -> CovariateModel:
    """Poisson MLE on pre-match ratings plus the named covariate columns.

    ``symmetric`` names columns holding one value per match; ``per_side`` names
    pairs, expected as ``<name>_home`` and ``<name>_away``.
    """
    needed = ["elo_home", "elo_away", "hg", "ag", *symmetric]
    needed += [f"{n}_{side}" for n in per_side for side in ("home", "away")]
    data = frame.dropna(subset=needed)
    d = (data.elo_home.to_numpy(float) - data.elo_away.to_numpy(float)) / 400.0
    hg = data.hg.to_numpy(float)
    ag = data.ag.to_numpy(float)
    sym = [data[name].to_numpy(float) for name in symmetric]
    side_home = [data[f"{name}_home"].to_numpy(float) for name in per_side]
    side_away = [data[f"{name}_away"].to_numpy(float) for name in per_side]

    n_sym, n_side = len(symmetric), len(per_side)

    def objective(theta):
        a, h, b = theta[0], theta[1], theta[2]
        sym_c = theta[3:3 + n_sym]
        side_c = theta[3 + n_sym:]
        log_h = a + h + b * d
        log_a = a - b * d
        for value, coefficient in zip(sym, sym_c):
            log_h = log_h + coefficient * value
            log_a = log_a - coefficient * value
        for home_value, away_value, coefficient in zip(side_home, side_away, side_c):
            log_h = log_h + coefficient * home_value
            log_a = log_a + coefficient * away_value
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
        return value, gradient

    start = np.zeros(3 + n_sym + n_side)
    start[0] = config.POISSON_FALLBACK["a"]
    start[1] = config.POISSON_FALLBACK["h"]
    start[2] = config.POISSON_FALLBACK["b"]
    result = minimize(objective, start, jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-12, "gtol": 1e-10})
    a, h, b = result.x[0], result.x[1], result.x[2]
    sym_c = dict(zip(symmetric, result.x[3:3 + n_sym]))
    side_c = dict(zip(per_side, result.x[3 + n_sym:]))

    log_h = a + h + b * d
    log_a = a - b * d
    for value, name in zip(sym, symmetric):
        log_h = log_h + sym_c[name] * value
        log_a = log_a - sym_c[name] * value
    for home_value, away_value, name in zip(side_home, side_away, per_side):
        log_h = log_h + side_c[name] * home_value
        log_a = log_a + side_c[name] * away_value
    full = float(np.sum(poisson_dist.logpmf(hg, np.exp(log_h)))
                 + np.sum(poisson_dist.logpmf(ag, np.exp(log_a))))
    return CovariateModel(
        a=float(a), h=float(h), b=float(b),
        symmetric={k: float(v) for k, v in sym_c.items()},
        per_side={k: float(v) for k, v in side_c.items()},
        matches=len(data), converged=bool(result.success), log_likelihood=full,
    )


@dataclass(frozen=True)
class SquadValueModel:
    """Model v1: the Elo Poisson model plus the pre-season squad-value prior.

    The simulator works in club ids, so the covariate has to be looked up per
    match rather than passed in as a column: :meth:`rates_for` takes the two
    clubs. Where either club has no value the covariate is **zero for that
    match** - the same convention the backtest scored, so the forecast is never
    better informed than the test that accepted it.

    The values are a snapshot at the season's as-of date (6 July), frozen: the
    source has been paused since 2026-07-06, so a weekly update refreshes the
    ratings and not this.
    """

    model: CovariateModel
    values: dict[str, float]
    as_of: object = None

    @property
    def a(self) -> float:
        return self.model.a

    @property
    def h(self) -> float:
        return self.model.h

    @property
    def b(self) -> float:
        return self.model.b

    def difference(self, home, away) -> np.ndarray:
        """log(value_home) - log(value_away), zero unless both are valued."""
        home_value = np.array([self.values.get(c, np.nan) for c in np.atleast_1d(home)])
        away_value = np.array([self.values.get(c, np.nan) for c in np.atleast_1d(away)])
        known = ~np.isnan(home_value) & ~np.isnan(away_value)
        out = np.zeros(len(home_value))
        out[known] = np.log(home_value[known]) - np.log(away_value[known])
        return out

    def rates_for(self, home, away, elo_home, elo_away, *, neutral: bool = False):
        """Expected goals for (home, away), covariate included."""
        difference = self.difference(home, away)
        scalar = np.ndim(elo_home) == 0 and np.ndim(home) == 0
        rate_home, rate_away = self.model.rates(
            elo_home, elo_away, neutral=neutral,
            symmetric={"d_value": difference[0] if scalar else difference})
        return rate_home, rate_away

    def rates(self, elo_home, elo_away, *, neutral: bool = False):
        """Refused: without the club ids the covariate cannot be evaluated.

        Silently dropping it would make the forecast quietly Elo-only, which is
        the kind of bug that cost two re-runs already.
        """
        raise TypeError(
            "SquadValueModel needs the club ids; call rates_for(home, away, ...)")

    def as_dict(self) -> dict:
        return self.model.as_dict()

    def coverage(self, home, away) -> float:
        """Share of the given fixtures where both clubs are valued."""
        home = np.atleast_1d(home)
        away = np.atleast_1d(away)
        known = np.array([c in self.values for c in home])
        known_away = np.array([c in self.values for c in away])
        return float(np.mean(known & known_away))


def fit_squad_value_model(pre_match: pd.DataFrame, values: dict[str, float],
                          *, as_of=None) -> SquadValueModel:
    """Fit model v1 on pre-match ratings and a squad-value snapshot."""
    frame = pre_match.copy()
    model = SquadValueModel(CovariateModel(0.0, 0.0, 0.0, {"d_value": 0.0}), values, as_of)
    frame["d_value"] = model.difference(
        frame.home_id.to_numpy(), frame.away_id.to_numpy())
    fitted = fit_covariate_model(frame, symmetric=("d_value",))
    return SquadValueModel(fitted, values, as_of)
