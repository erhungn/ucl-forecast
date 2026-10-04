"""Model v1: the Elo Poisson model plus the pre-season squad-value prior.

The covariate has to reach the simulator through club ids, and the ways that can
go wrong are all silent - a dropped covariate, a covariate applied to a club
with no value, or one applied twice. These pin it down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ucl.models.covariates import (
    ClubCovariate, CovariateModel, ForecastModel, fit_covariate_model,
    fit_squad_value_model, value_covariate,
)
from ucl.models.poisson import fit_poisson
from ucl.sim.league_phase import simulate_match

VALUES = {"rich": 800e6, "poor": 50e6}


def model(b_value: float = 0.04) -> ForecastModel:
    """Model v1: the general covariate model carrying one covariate."""
    return ForecastModel(
        CovariateModel(a=0.16, h=0.23, b=0.70, symmetric={"d_value": b_value}),
        (value_covariate(VALUES),), label="v1")


def difference(v1: ForecastModel, home, away):
    return v1.differences(home, away)["d_value"]


def test_rates_for_matches_the_covariate_model_given_the_difference():
    v1 = model()
    gap = np.log(VALUES["rich"]) - np.log(VALUES["poor"])
    expected = v1.model.rates(1900.0, 1700.0, symmetric={"d_value": gap})
    actual = v1.rates_for("rich", "poor", 1900.0, 1700.0)
    assert actual == pytest.approx(expected)


def test_the_covariate_is_zero_unless_both_clubs_are_valued():
    """The backtest scored it that way, so the forecast must not do better."""
    v1 = model()
    elo_only = CovariateModel(a=0.16, h=0.23, b=0.70).rates(1900.0, 1700.0)
    for home, away in (("rich", "unknown"), ("unknown", "poor"), ("x", "y")):
        assert v1.rates_for(home, away, 1900.0, 1700.0) == pytest.approx(elo_only)


def test_the_difference_is_antisymmetric():
    v1 = model()
    assert difference(v1, "rich", "poor") == pytest.approx(
        -difference(v1, "poor", "rich"))


def test_rates_without_club_ids_is_refused_rather_than_silently_elo_only():
    with pytest.raises(TypeError, match="club ids"):
        model().rates(1900.0, 1700.0)


# --- model v2: the same machinery carrying four covariates ----------------

def _v2(b_value: float = 0.04, b_shots: float = 0.023) -> ForecastModel:
    shots = {"rich": 2.0, "poor": -1.0}
    return ForecastModel(
        CovariateModel(a=0.16, h=0.23, b=0.70,
                       symmetric={"d_value": b_value, "d_gap_sot": b_shots}),
        (value_covariate(VALUES),
         ClubCovariate("d_gap_sot",
                       lambda home, away: np.array(
                           [shots.get(h, 0.0) - shots.get(a, 0.0)
                            for h, a in zip(np.atleast_1d(home),
                                            np.atleast_1d(away))]),
                       lambda club: club in shots)),
        label="v2")


def test_several_covariates_are_all_applied():
    v2 = _v2()
    one = model().rates_for("rich", "poor", 1900.0, 1700.0)
    both = v2.rates_for("rich", "poor", 1900.0, 1700.0)
    # The shot term adds to the home rate and subtracts from the away rate.
    assert both[0] > one[0] and both[1] < one[1]
    assert v2.names == ("d_value", "d_gap_sot")


def test_coverage_is_reported_for_each_covariate_separately():
    v2 = _v2()
    shares = v2.coverage(["rich", "rich", "other"], ["poor", "unknown", "poor"])
    assert set(shares) == {"d_value", "d_gap_sot"}
    assert shares["d_value"] == pytest.approx(1 / 3)
    assert shares["d_gap_sot"] == pytest.approx(1 / 3)


def test_a_zero_coefficient_covariate_changes_nothing():
    """Adding a covariate the fit gave no weight must leave v1's rates alone."""
    v1 = model()
    v2 = _v2(b_shots=0.0)
    assert v2.rates_for("rich", "poor", 1900.0, 1700.0) == pytest.approx(
        v1.rates_for("rich", "poor", 1900.0, 1700.0))


def test_rates_for_is_vectorised():
    v1 = model()
    home = np.array(["rich", "rich"])
    away = np.array(["poor", "unknown"])
    rate_home, rate_away = v1.rates_for(home, away, np.array([1900.0, 1900.0]),
                                        np.array([1700.0, 1700.0]))
    assert rate_home[0] > rate_home[1]      # the covariate helps only the first
    assert rate_away[0] < rate_away[1]


def test_coverage_counts_fixtures_where_both_clubs_are_valued():
    v1 = model()
    # Now reported per covariate, since a model may carry several.
    assert v1.coverage(["rich", "rich"], ["poor", "unknown"]) == {"d_value": 0.5}


def _synthetic(b_value: float, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 8000
    clubs = np.array(["rich", "poor", "mid"])
    values = {"rich": 800e6, "poor": 50e6, "mid": 200e6}
    home = rng.choice(clubs, n)
    away = np.array([rng.choice(clubs[clubs != h]) for h in home])
    elo_home = rng.normal(1700, 120, n)
    elo_away = rng.normal(1700, 120, n)
    d_elo = (elo_home - elo_away) / 400.0
    d_value = np.log([values[c] for c in home]) - np.log([values[c] for c in away])
    rate_home = np.exp(0.16 + 0.23 + 0.70 * d_elo + b_value * d_value)
    rate_away = np.exp(0.16 - 0.70 * d_elo - b_value * d_value)
    return pd.DataFrame({
        "home_id": home, "away_id": away, "elo_home": elo_home, "elo_away": elo_away,
        "hg": rng.poisson(rate_home), "ag": rng.poisson(rate_away),
    }), values


def test_a_known_coefficient_is_recovered():
    frame, values = _synthetic(b_value=0.12)
    fitted = fit_squad_value_model(frame, values)
    assert fitted.model.symmetric["d_value"] == pytest.approx(0.12, abs=0.02)
    assert fitted.b == pytest.approx(0.70, abs=0.03)


def test_with_no_values_the_fit_collapses_to_the_elo_only_model():
    frame, _ = _synthetic(b_value=0.0)
    plain = fit_poisson(frame)
    collapsed = fit_squad_value_model(frame, {})
    assert collapsed.a == pytest.approx(plain.a, abs=1e-4)
    assert collapsed.h == pytest.approx(plain.h, abs=1e-4)
    assert collapsed.b == pytest.approx(plain.b, abs=1e-4)


def test_the_simulator_uses_the_covariate_and_not_only_the_ratings():
    """Same ratings, different squad values, different scorelines."""
    ratings = {"rich": 1800.0, "poor": 1800.0}
    draws = []
    for b_value in (0.0, 0.5):
        rng = np.random.default_rng(11)
        v1 = model(b_value)
        goals = [simulate_match("rich", "poor", dict(ratings), v1, rng, update=False)
                 for _ in range(400)]
        draws.append(np.mean([h - a for h, a in goals]))
    assert draws[1] > draws[0] + 0.5


def test_the_elo_update_ignores_the_covariate():
    """Squad value shifts the goal rates, never the Elo dynamics."""
    rng = np.random.default_rng(3)
    ratings = {"rich": 1800.0, "poor": 1800.0}
    v1 = model(0.0)
    hg, ag = simulate_match("rich", "poor", ratings, v1, rng)
    moved = ratings["rich"] - 1800.0
    # The same scoreline through the plain model must move the rating identically.
    plain_ratings = {"rich": 1800.0, "poor": 1800.0}
    rng2 = np.random.default_rng(3)
    plain = ForecastModel(CovariateModel(0.16, 0.23, 0.70, {"d_value": 0.0}),
                          (value_covariate({}),), label="v1")
    hg2, ag2 = simulate_match("rich", "poor", plain_ratings, plain, rng2)
    assert (hg, ag) == (hg2, ag2)
    assert plain_ratings["rich"] - 1800.0 == pytest.approx(moved)
