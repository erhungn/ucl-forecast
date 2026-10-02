"""The baseline match model: Elo difference to a scoreline.

The tiebreakers need goal difference, so the simulator needs scorelines rather
than win probabilities:

    d = (elo_home - elo_away) / 400          # no Elo home advantage here
    log(lambda_home) = a + h + b * d
    log(lambda_away) = a     - b * d

Home advantage is carried by ``h`` alone. The Elo home advantage belongs to the
rating update (:mod:`ucl.models.elo`); including it in ``d`` as well would count
it twice, once in Elo points and once in goals.

``a``, ``h`` and ``b`` are fitted by Poisson maximum likelihood on the ratings as
they stood *before* each match, never on post-match ratings, which would leak the
result being predicted into its own covariate.

Dixon-Coles (milestone 4) replaces this with the same linear predictor plus
attack/defence deviations, a low-score correction and time decay, so the
comparison isolates exactly those additions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from ucl import config

__all__ = ["PoissonModel", "fit_poisson", "expected_goals"]


@dataclass(frozen=True)
class PoissonModel:
    a: float
    h: float
    b: float
    matches: int = 0
    converged: bool = True
    log_likelihood: float = float("nan")

    def rates(self, elo_home: float, elo_away: float, *, neutral: bool = False
              ) -> tuple[float, float]:
        """Expected goals for (home, away). ``neutral`` drops home advantage."""
        d = (np.asarray(elo_home) - np.asarray(elo_away)) / 400.0
        home_term = self.a + (0.0 if neutral else self.h) + self.b * d
        away_term = self.a - self.b * d
        return np.exp(home_term), np.exp(away_term)

    def rates_for(self, home, away, elo_home, elo_away, *, neutral: bool = False):
        """The simulator's entry point. This model does not use the club ids."""
        return self.rates(elo_home, elo_away, neutral=neutral)

    def as_dict(self) -> dict[str, float]:
        return {"a": self.a, "h": self.h, "b": self.b}


def expected_goals(
    model: PoissonModel, elo_home, elo_away, *, neutral: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    return model.rates(elo_home, elo_away, neutral=neutral)


def fit_poisson(pre_match: pd.DataFrame, *, neutral_column: str | None = None) -> PoissonModel:
    """Fit ``a``, ``h``, ``b`` on pre-match ratings by Poisson maximum likelihood.

    ``pre_match`` is :attr:`ucl.models.elo.EloResult.pre_match`: one row per
    match with ``elo_home``/``elo_away`` as they stood before kick-off.
    """
    frame = pre_match.dropna(subset=["elo_home", "elo_away", "hg", "ag"])
    d = (frame.elo_home.to_numpy() - frame.elo_away.to_numpy()) / 400.0
    hg = frame.hg.to_numpy(dtype=float)
    ag = frame.ag.to_numpy(dtype=float)
    at_home = (
        np.ones(len(frame))
        if neutral_column is None
        else (~frame[neutral_column].to_numpy(dtype=bool)).astype(float)
    )

    def negative_log_likelihood(theta: np.ndarray) -> float:
        a, h, b = theta
        log_home = a + h * at_home + b * d
        log_away = a - b * d
        # Poisson log-likelihood, constant terms dropped.
        return float(
            np.sum(np.exp(log_home) - hg * log_home)
            + np.sum(np.exp(log_away) - ag * log_away)
        )

    start = np.array([
        config.POISSON_FALLBACK["a"],
        config.POISSON_FALLBACK["h"],
        config.POISSON_FALLBACK["b"],
    ])
    # L-BFGS-B converges cleanly here; plain BFGS reports precision loss on a
    # likelihood this flat near the optimum.
    result = minimize(negative_log_likelihood, start, method="L-BFGS-B",
                      options={"ftol": 1e-12, "gtol": 1e-10})
    a, h, b = result.x
    full = float(
        np.sum(poisson.logpmf(hg, np.exp(a + h * at_home + b * d)))
        + np.sum(poisson.logpmf(ag, np.exp(a - b * d)))
    )
    return PoissonModel(
        a=float(a), h=float(h), b=float(b),
        matches=len(frame), converged=bool(result.success), log_likelihood=full,
    )
