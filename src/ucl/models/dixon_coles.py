"""Dixon-Coles with an Elo covariate, ridge-shrunk attack and defence.

For a match between club *i* at home and club *j* away, with Elo taken strictly
before kick-off::

    d         = (elo_i - elo_j) / 400
    log(l_h)  = mu + home + beta * d + att_i - def_j
    log(l_a)  = mu        - beta * d + att_j - def_i

Goals are Poisson with the Dixon-Coles correction tau applied to the four
low-score cells, fitted on a time-decayed weighted log-likelihood with a ridge
penalty on att and def.

Two things about the parameterisation matter:

**Identification.** Adding a constant to every ``att`` and every ``def`` leaves
every rate unchanged, so the model is translation-invariant and the ridge
penalty is what pins it down, centring both at zero. No sum-to-zero constraint
is needed.

**Shrinkage is the fallback to Elo.** A club's att/def is pulled toward zero by
roughly ``n / (n + ridge)`` where ``n`` is its decay-weighted match count, and at
zero the rate is exactly ``mu + home + beta * d`` - the Elo-only baseline. At the
2026-27 cutoff the median club has under 5 effective matches at a one-year
half-life, so most clubs sit close to that fallback by construction, with no
special-casing for thin records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = ["DixonColesModel", "fit_dixon_coles", "tau", "score_matrix", "decay_weights",
           "build_objective"]

#: Keeps tau positive so the log-likelihood stays finite.
TAU_FLOOR = 1e-6
RHO_BOUNDS = (-0.25, 0.25)
MAX_GOALS = 10


def decay_weights(dates, cutoff: date, half_life_days: float) -> np.ndarray:
    """0.5 ** (age / half_life), so a match that old counts half as much."""
    age = (np.datetime64(cutoff) - np.asarray(dates, dtype="datetime64[D]"))
    age = age.astype("timedelta64[D]").astype(float)
    return 0.5 ** (age / float(half_life_days))


def tau(hg, ag, rate_home, rate_away, rho: float) -> np.ndarray:
    """The Dixon-Coles correction, 1 everywhere except the four low scorelines."""
    hg = np.asarray(hg)
    ag = np.asarray(ag)
    out = np.ones(np.broadcast(hg, ag, rate_home, rate_away).shape, dtype=float)
    out = np.where((hg == 0) & (ag == 0), 1.0 - rate_home * rate_away * rho, out)
    out = np.where((hg == 0) & (ag == 1), 1.0 + rate_home * rho, out)
    out = np.where((hg == 1) & (ag == 0), 1.0 + rate_away * rho, out)
    out = np.where((hg == 1) & (ag == 1), 1.0 - rho, out)
    return np.maximum(out, TAU_FLOOR)


@dataclass
class DixonColesModel:
    mu: float
    home: float
    beta: float
    rho: float
    attack: dict[str, float]
    defence: dict[str, float]
    half_life_days: float
    ridge: float
    effective_matches: dict[str, float] = field(default_factory=dict)
    converged: bool = True
    iterations: int = 0
    log_likelihood: float = float("nan")

    def rates(self, elo_home, elo_away, home_id, away_id, *, neutral: bool = False):
        """Expected goals for (home, away). Unknown clubs fall back to att=def=0."""
        d = (np.asarray(elo_home, dtype=float) - np.asarray(elo_away, dtype=float)) / 400.0
        attack_home = np.array([self.attack.get(c, 0.0) for c in np.atleast_1d(home_id)])
        attack_away = np.array([self.attack.get(c, 0.0) for c in np.atleast_1d(away_id)])
        defence_home = np.array([self.defence.get(c, 0.0) for c in np.atleast_1d(home_id)])
        defence_away = np.array([self.defence.get(c, 0.0) for c in np.atleast_1d(away_id)])
        advantage = 0.0 if neutral else self.home
        return (
            np.exp(self.mu + advantage + self.beta * d + attack_home - defence_away),
            np.exp(self.mu - self.beta * d + attack_away - defence_home),
        )

    def shrinkage(self, club: str) -> float:
        """n / (n + ridge): how much of the unpenalised deviation survives."""
        n = self.effective_matches.get(club, 0.0)
        return float(n / (n + self.ridge)) if self.ridge > 0 else 1.0

    def as_dict(self) -> dict:
        return {"mu": self.mu, "home": self.home, "beta": self.beta, "rho": self.rho,
                "half_life_days": self.half_life_days, "ridge": self.ridge}


def score_matrix(rate_home: float, rate_away: float, rho: float,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    """The corrected joint scoreline distribution, normalised."""
    from scipy.stats import poisson

    goals = np.arange(max_goals + 1)
    joint = np.outer(poisson.pmf(goals, rate_home), poisson.pmf(goals, rate_away))
    grid_h, grid_a = np.meshgrid(goals, goals, indexing="ij")
    joint = joint * tau(grid_h, grid_a, rate_home, rate_away, rho)
    return joint / joint.sum()


def build_objective(
    matches: pd.DataFrame,
    cutoff: date,
    *,
    half_life_days: float,
    ridge: float,
    weight_floor: float = 1e-4,
):
    """The penalised negative log-likelihood and its analytic gradient.

    Returned separately from the fit so the gradient can be checked against a
    numerical one in the tests.
    """
    weights = decay_weights(matches.date.to_numpy(), cutoff, half_life_days)
    keep = weights >= weight_floor
    frame = matches.loc[keep]
    weights = weights[keep]

    clubs = sorted(set(frame.home_id) | set(frame.away_id))
    index = {club: i for i, club in enumerate(clubs)}
    n_clubs = len(clubs)
    home_index = frame.home_id.map(index).to_numpy()
    away_index = frame.away_id.map(index).to_numpy()
    hg = frame.hg.to_numpy(dtype=float)
    ag = frame.ag.to_numpy(dtype=float)
    d = (frame.elo_home.to_numpy(dtype=float) - frame.elo_away.to_numpy(dtype=float)) / 400.0

    low_00 = (hg == 0) & (ag == 0)
    low_01 = (hg == 0) & (ag == 1)
    low_10 = (hg == 1) & (ag == 0)
    low_11 = (hg == 1) & (ag == 1)

    def unpack(theta):
        return (theta[0], theta[1], theta[2], theta[3],
                theta[4:4 + n_clubs], theta[4 + n_clubs:])

    def objective(theta):
        mu, home, beta, rho, attack, defence = unpack(theta)
        log_h = mu + home + beta * d + attack[home_index] - defence[away_index]
        log_a = mu - beta * d + attack[away_index] - defence[home_index]
        rate_h, rate_a = np.exp(log_h), np.exp(log_a)

        # Poisson part.
        value = np.sum(weights * (rate_h - hg * log_h + rate_a - ag * log_a))
        residual_h = weights * (rate_h - hg)
        residual_a = weights * (rate_a - ag)

        # Dixon-Coles correction, only the four low-score cells.
        correction = np.ones_like(rate_h)
        correction = np.where(low_00, 1.0 - rate_h * rate_a * rho, correction)
        correction = np.where(low_01, 1.0 + rate_h * rho, correction)
        correction = np.where(low_10, 1.0 + rate_a * rho, correction)
        correction = np.where(low_11, 1.0 - rho, correction)
        correction = np.maximum(correction, TAU_FLOOR)
        value -= np.sum(weights * np.log(correction))

        # d log(tau) / d log(rate_h) and / d log(rate_a), and / d rho.
        inverse = weights / correction
        grad_log_h = np.zeros_like(rate_h)
        grad_log_a = np.zeros_like(rate_a)
        grad_rho = np.zeros_like(rate_h)
        grad_log_h = np.where(low_00, -rate_h * rate_a * rho, grad_log_h)
        grad_log_a = np.where(low_00, -rate_h * rate_a * rho, grad_log_a)
        grad_rho = np.where(low_00, -rate_h * rate_a, grad_rho)
        grad_log_h = np.where(low_01, rate_h * rho, grad_log_h)
        grad_rho = np.where(low_01, rate_h, grad_rho)
        grad_log_a = np.where(low_10, rate_a * rho, grad_log_a)
        grad_rho = np.where(low_10, rate_a, grad_rho)
        grad_rho = np.where(low_11, -1.0, grad_rho)

        residual_h = residual_h - inverse * grad_log_h
        residual_a = residual_a - inverse * grad_log_a

        gradient = np.zeros_like(theta)
        gradient[0] = residual_h.sum() + residual_a.sum()
        gradient[1] = residual_h.sum()
        gradient[2] = np.sum(d * (residual_h - residual_a))
        gradient[3] = -np.sum(inverse * grad_rho)
        gradient[4:4 + n_clubs] = (
            np.bincount(home_index, residual_h, n_clubs)
            + np.bincount(away_index, residual_a, n_clubs)
        )
        gradient[4 + n_clubs:] = -(
            np.bincount(away_index, residual_h, n_clubs)
            + np.bincount(home_index, residual_a, n_clubs)
        )

        # Ridge on att/def only; mu, home, beta and rho are unpenalised.
        value += 0.5 * ridge * (np.sum(attack ** 2) + np.sum(defence ** 2))
        gradient[4:4 + n_clubs] += ridge * attack
        gradient[4 + n_clubs:] += ridge * defence
        return value, gradient

    theta = np.zeros(4 + 2 * n_clubs)
    theta[0] = float(np.log(max(np.average(
        np.concatenate([hg, ag]), weights=np.concatenate([weights, weights])), 0.1)))
    theta[1] = 0.25
    theta[2] = 0.8
    effective = (
        np.bincount(home_index, weights, n_clubs) + np.bincount(away_index, weights, n_clubs)
    )
    return objective, theta, index, effective


def fit_dixon_coles(
    matches: pd.DataFrame,
    cutoff: date,
    *,
    half_life_days: float,
    ridge: float,
    weight_floor: float = 1e-4,
    start: DixonColesModel | None = None,
    maxiter: int = 500,
) -> DixonColesModel:
    """Fit by penalised maximum likelihood with analytic gradients.

    ``matches`` needs ``date``, ``home_id``, ``away_id``, ``hg``, ``ag``,
    ``elo_home`` and ``elo_away`` - the pre-match ratings, never post-match.
    """
    objective, theta, index, effective = build_objective(
        matches, cutoff, half_life_days=half_life_days, ridge=ridge,
        weight_floor=weight_floor,
    )
    n_clubs = len(index)
    if start is not None:
        theta[0], theta[1], theta[2], theta[3] = start.mu, start.home, start.beta, start.rho
        for club, position in index.items():
            theta[4 + position] = start.attack.get(club, 0.0)
            theta[4 + n_clubs + position] = start.defence.get(club, 0.0)

    bounds = [(None, None)] * 3 + [RHO_BOUNDS] + [(None, None)] * (2 * n_clubs)
    result = minimize(objective, theta, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": maxiter, "maxfun": maxiter * 2})
    mu, home, beta, rho = result.x[0], result.x[1], result.x[2], result.x[3]
    attack, defence = result.x[4:4 + n_clubs], result.x[4 + n_clubs:]

    return DixonColesModel(
        mu=float(mu), home=float(home), beta=float(beta), rho=float(rho),
        attack={club: float(attack[i]) for club, i in index.items()},
        defence={club: float(defence[i]) for club, i in index.items()},
        half_life_days=float(half_life_days), ridge=float(ridge),
        effective_matches={club: float(effective[i]) for club, i in index.items()},
        converged=bool(result.success), iterations=int(result.nit),
        log_likelihood=float(-result.fun),
    )
