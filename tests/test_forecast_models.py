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


# --- candidate C: per-competition intercepts -------------------------------
#
# Four new parameters, two of which must behave differently at a neutral venue
# from the two others. The goal scale sits on both log-rates and is a property
# of the competition, so it applies at the final; the extra home advantage is a
# property of playing at home and must vanish there, exactly as `h` does.

LEVELLED = dict(a=0.16, h=0.23, b=0.70,
                scale={"ucl_main": 0.10, "other_european": -0.03},
                home_extra={"ucl_main": 0.05, "other_european": 0.09})
UCL = {"domestic": 0.0, "ucl_main": 1.0, "other_european": 0.0}


def test_the_goal_scale_lands_on_both_sides_and_the_home_extra_on_one():
    fitted = CovariateModel(**LEVELLED)
    home, away = fitted.rates(1800.0, 1800.0, levels=UCL)
    assert home == pytest.approx(np.exp(0.16 + 0.23 + 0.10 + 0.05))
    assert away == pytest.approx(np.exp(0.16 + 0.10))


def test_the_neutral_venue_final_keeps_the_scale_and_loses_the_home_extra():
    """The trap this design exists to avoid.

    `neutral=True` zeroes `h`. A home-only per-side covariate would NOT have
    been zeroed by it, so the final - the one match of the tournament played at
    a neutral ground - would have carried European home advantage for the club
    nominally listed first.
    """
    fitted = CovariateModel(**LEVELLED)
    home, away = fitted.rates(1800.0, 1800.0, neutral=True, levels=UCL)
    assert home == pytest.approx(np.exp(0.16 + 0.10)), "h and g must both be gone"
    assert away == pytest.approx(np.exp(0.16 + 0.10)), "the scale must remain"
    assert home == pytest.approx(away), "a neutral venue is symmetric"


def test_a_domestic_match_gets_neither_term():
    fitted = CovariateModel(**LEVELLED)
    domestic = {"domestic": 1.0, "ucl_main": 0.0, "other_european": 0.0}
    plain = CovariateModel(a=0.16, h=0.23, b=0.70)
    assert fitted.rates(1800.0, 1700.0, levels=domestic) == pytest.approx(
        plain.rates(1800.0, 1700.0))


def test_a_levelled_model_refuses_to_guess_the_competition():
    """Defaulting to zero would forecast the league phase at the domestic scale.

    Nothing downstream would look wrong: the probabilities would still sum to
    one and the band counts would still be 8/16/12.
    """
    with pytest.raises(TypeError, match="cannot produce rates"):
        CovariateModel(**LEVELLED).rates(1800.0, 1700.0)


def test_a_forecast_model_carrying_levels_must_declare_one():
    with pytest.raises(TypeError, match="no level was declared"):
        ForecastModel(CovariateModel(**LEVELLED), (value_covariate(VALUES),))


def test_an_unknown_level_is_refused_rather_than_silently_baselined():
    with pytest.raises(ValueError, match="unknown competition level"):
        ForecastModel(CovariateModel(**LEVELLED), (value_covariate(VALUES),),
                      level="ucl-main")


def test_a_model_without_levels_is_unchanged_by_any_of_this():
    """The reproduction gate, at the arithmetic level.

    v1 and v2 carry empty scale and home_extra dicts, so neither new loop runs
    and `levels` is never consulted - the published code path is the same
    operations in the same order, not merely the same answer.
    """
    plain = CovariateModel(a=0.16, h=0.23, b=0.70, symmetric={"d_value": 0.04})
    assert plain.scale == {} and plain.home_extra == {}
    assert plain.rates(1900.0, 1700.0, symmetric={"d_value": 0.5}) == pytest.approx(
        plain.rates(1900.0, 1700.0, symmetric={"d_value": 0.5}, levels=UCL))


def test_zero_coefficients_reproduce_the_model_without_the_terms():
    zeroed = CovariateModel(a=0.16, h=0.23, b=0.70,
                            scale={"ucl_main": 0.0}, home_extra={"ucl_main": 0.0})
    plain = CovariateModel(a=0.16, h=0.23, b=0.70)
    assert zeroed.rates(1800.0, 1700.0, levels=UCL) == pytest.approx(
        plain.rates(1800.0, 1700.0))


def test_the_published_keys_cannot_collide():
    """`c_` is shared by per-side covariates and competition scales."""
    clash = CovariateModel(a=0.1, h=0.2, b=0.7, per_side={"ucl_main": 0.01},
                           scale={"ucl_main": 0.10})
    with pytest.raises(ValueError, match="same key"):
        clash.as_dict()


def test_the_levelled_model_publishes_c_and_g_keys():
    published = CovariateModel(**LEVELLED).as_dict()
    assert published["c_ucl_main"] == pytest.approx(0.10)
    assert published["g_ucl_main"] == pytest.approx(0.05)
    assert published["c_other_european"] == pytest.approx(-0.03)
    assert published["g_other_european"] == pytest.approx(0.09)


def test_fitting_levels_recovers_a_planted_goal_scale():
    """End to end through the fitter, including the new gradient blocks."""
    rng = np.random.default_rng(3)
    n = 20000
    level = rng.choice(["domestic", "ucl_main"], size=n, p=[0.6, 0.4])
    frame = pd.DataFrame({
        "elo_home": 1700.0, "elo_away": 1700.0, "level": level,
        "ucl_main": (level == "ucl_main").astype(float),
    })
    # log lambda = a + h (+ 0.20 and + 0.10 on the home side for ucl_main)
    rate_home = np.exp(0.16 + 0.23 + 0.20 * frame.ucl_main + 0.10 * frame.ucl_main)
    rate_away = np.exp(0.16 + 0.20 * frame.ucl_main)
    frame["hg"] = rng.poisson(rate_home)
    frame["ag"] = rng.poisson(rate_away)
    fitted = fit_covariate_model(frame, levels=("ucl_main",))
    assert fitted.converged
    assert fitted.scale["ucl_main"] == pytest.approx(0.20, abs=0.03)
    assert fitted.home_extra["ucl_main"] == pytest.approx(0.10, abs=0.04)
    assert fitted.a == pytest.approx(0.16, abs=0.03)
    assert fitted.h == pytest.approx(0.23, abs=0.04)
