"""Dixon-Coles: the gradient, parameter recovery, and the fallback to Elo."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from ucl.models.dixon_coles import (
    build_objective, decay_weights, fit_dixon_coles, score_matrix, tau,
)

CUTOFF = date(2026, 9, 7)


def synthetic(n=600, clubs=16, seed=0, attack=None, defence=None,
              mu=0.2, home=0.25, beta=0.8):
    rng = np.random.default_rng(seed)
    names = [f"C{i:02d}" for i in range(clubs)]
    attack = np.zeros(clubs) if attack is None else attack
    defence = np.zeros(clubs) if defence is None else defence
    hi = rng.integers(0, clubs, n)
    ai = rng.integers(0, clubs, n)
    keep = hi != ai
    hi, ai = hi[keep], ai[keep]
    elo_home = rng.normal(1700, 90, len(hi))
    elo_away = rng.normal(1700, 90, len(hi))
    d = (elo_home - elo_away) / 400.0
    rate_h = np.exp(mu + home + beta * d + attack[hi] - defence[ai])
    rate_a = np.exp(mu - beta * d + attack[ai] - defence[hi])
    return pd.DataFrame({
        "date": pd.to_datetime("2026-06-01") - pd.to_timedelta(
            rng.integers(0, 500, len(hi)), unit="D"),
        "home_id": [names[i] for i in hi], "away_id": [names[i] for i in ai],
        "hg": rng.poisson(rate_h), "ag": rng.poisson(rate_a),
        "elo_home": elo_home, "elo_away": elo_away,
    })


def test_the_analytic_gradient_matches_a_numerical_one():
    frame = synthetic(n=300, clubs=10)
    objective, theta0, _, _ = build_objective(
        frame, CUTOFF, half_life_days=365, ridge=7.0)
    rng = np.random.default_rng(1)
    theta = theta0 + rng.normal(0, 0.1, len(theta0))
    theta[3] = 0.05
    _, analytic = objective(theta)
    numerical = np.zeros_like(theta)
    step = 1e-6
    for i in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[i] += step
        down[i] -= step
        numerical[i] = (objective(up)[0] - objective(down)[0]) / (2 * step)
    assert np.abs(analytic - numerical).max() < 1e-5


def test_a_fit_recovers_the_parameters_it_was_generated_from():
    rng = np.random.default_rng(3)
    clubs = 14
    attack = rng.normal(0, 0.35, clubs)
    defence = rng.normal(0, 0.35, clubs)
    frame = synthetic(n=12000, clubs=clubs, seed=4, attack=attack, defence=defence,
                      mu=0.2, home=0.26, beta=0.75)
    model = fit_dixon_coles(frame, CUTOFF, half_life_days=100000, ridge=0.5)
    assert model.converged
    assert model.home == pytest.approx(0.26, abs=0.05)
    assert model.beta == pytest.approx(0.75, abs=0.10)
    names = sorted(model.attack)
    fitted = np.array([model.attack[c] for c in names])
    # Ridge shrinks, so compare the pattern rather than the level.
    assert np.corrcoef(fitted, attack - attack.mean())[0, 1] > 0.9


def test_attack_and_defence_are_centred_by_the_ridge():
    model = fit_dixon_coles(synthetic(), CUTOFF, half_life_days=365, ridge=5.0)
    assert abs(np.mean(list(model.attack.values()))) < 1e-4
    assert abs(np.mean(list(model.defence.values()))) < 1e-4


def test_a_huge_ridge_collapses_the_model_onto_the_elo_baseline():
    frame = synthetic(n=2000, clubs=20, seed=7)
    model = fit_dixon_coles(frame, CUTOFF, half_life_days=365, ridge=1e7)
    assert max(abs(v) for v in model.attack.values()) < 1e-3
    assert max(abs(v) for v in model.defence.values()) < 1e-3
    # With att = def = 0 the rate is exactly mu + home + beta*d.
    rate_home, _ = model.rates([1800.0], [1700.0], ["C00"], ["C01"])
    expected = np.exp(model.mu + model.home + model.beta * (100.0 / 400.0))
    # Not exactly zero at a finite ridge, so compare to within a rounding error.
    assert rate_home[0] == pytest.approx(expected, rel=1e-4)


def test_an_unseen_club_falls_back_to_elo_only():
    model = fit_dixon_coles(synthetic(), CUTOFF, half_life_days=365, ridge=5.0)
    assert "NEVER_SEEN" not in model.attack
    rate_home, rate_away = model.rates([1800.0], [1700.0], ["NEVER_SEEN"], ["ALSO_NEW"])
    d = 100.0 / 400.0
    assert rate_home[0] == pytest.approx(np.exp(model.mu + model.home + model.beta * d))
    assert rate_away[0] == pytest.approx(np.exp(model.mu - model.beta * d))


def test_shrinkage_is_stronger_for_a_club_with_less_data():
    frame = synthetic(n=2000, clubs=20, seed=11)
    # Give one club a single match.
    thin = frame.iloc[:1].copy()
    thin["home_id"] = "THIN"
    frame = pd.concat([frame, thin], ignore_index=True)
    model = fit_dixon_coles(frame, CUTOFF, half_life_days=365, ridge=20.0)
    assert model.shrinkage("THIN") < model.shrinkage("C00")
    assert abs(model.attack["THIN"]) < max(abs(v) for v in model.attack.values())


def test_tau_is_one_away_from_the_low_scores():
    assert tau(2, 1, 1.4, 1.1, 0.1) == pytest.approx(1.0)
    assert tau(0, 0, 1.4, 1.1, 0.0) == pytest.approx(1.0)


def test_rho_zero_reproduces_independent_poisson():
    matrix = score_matrix(1.4, 1.1, 0.0, max_goals=12)
    goals = np.arange(13)
    independent = np.outer(poisson.pmf(goals, 1.4), poisson.pmf(goals, 1.1))
    independent = independent / independent.sum()
    assert np.abs(matrix - independent).max() < 1e-12


def test_rho_moves_probability_between_the_low_scores():
    base = score_matrix(1.3, 1.1, 0.0)
    positive = score_matrix(1.3, 1.1, 0.12)
    assert positive[0, 0] < base[0, 0]
    assert positive[1, 1] < base[1, 1]
    assert positive[0, 1] > base[0, 1]
    assert positive[1, 0] > base[1, 0]
    assert positive.sum() == pytest.approx(1.0)


def test_decay_weights_halve_at_the_half_life():
    dates = np.array(["2026-09-07", "2025-09-07", "2024-09-07"], dtype="datetime64[D]")
    weights = decay_weights(dates, CUTOFF, 365)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5, abs=0.01)
    assert weights[2] == pytest.approx(0.25, abs=0.01)


def test_old_matches_are_dropped_by_the_weight_floor():
    frame = synthetic(n=400, clubs=10, seed=5)
    ancient = frame.iloc[:5].copy()
    ancient["date"] = pd.to_datetime("1990-01-01")
    model = fit_dixon_coles(
        pd.concat([frame, ancient], ignore_index=True), CUTOFF,
        half_life_days=365, ridge=5.0, weight_floor=1e-4)
    # The ancient rows carry weight ~0, so they cannot shift the effective counts.
    assert sum(model.effective_matches.values()) < 2 * len(frame)
