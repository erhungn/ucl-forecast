"""Quantities read off the scoreline distribution.

Two implementations of the same arithmetic, checked against each other: the
closed forms (Poisson for the total, Skellam for the margin) against sums over
the scoreline matrix the simulator has always used. If they ever disagree, one
of them is wrong, and the card is reporting a number nobody can justify.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from ucl.backtest import probabilities_from_rates
from ucl.models import scorelines

#: Real rates from the 2026-27 fixtures: a close game, a mismatch, a rout.
RATES = [(1.27, 1.40), (1.61, 1.10), (2.69, 0.66), (3.15, 0.56), (0.91, 1.95)]


def test_the_matrix_is_a_probability_distribution():
    joint = scorelines.joint_matrix([1.5, 2.6], [1.1, 0.7], max_goals=30)
    assert joint.shape == (2, 31, 31)
    assert joint.sum(axis=(1, 2)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("home,away", RATES)
def test_probability_over_matches_the_matrix(home, away):
    """P(total > 2.5) analytically, against summing every scoreline."""
    analytic = scorelines.probability_over(2.5, home, away)[0]
    joint = scorelines.joint_matrix(home, away, max_goals=40)[0]
    goals = np.arange(joint.shape[0])
    totals = goals[:, None] + goals[None, :]
    summed = joint[totals > 2.5].sum()
    assert analytic == pytest.approx(summed, abs=1e-10)


@pytest.mark.parametrize("home,away", RATES)
def test_probability_margin_matches_the_matrix(home, away):
    """P(home wins by 3+) analytically, against summing every scoreline."""
    analytic = scorelines.probability_margin_at_least(3, home, away)[0]
    joint = scorelines.joint_matrix(home, away, max_goals=40)[0]
    goals = np.arange(joint.shape[0])
    margin = goals[:, None] - goals[None, :]
    summed = joint[margin >= 3].sum()
    assert analytic == pytest.approx(summed, abs=1e-10)


def test_the_truncated_grid_loses_a_measured_amount_of_mass():
    """How much the 0-10 grid misses, stated rather than assumed.

    This is why the closed forms are used for the reported numbers. The grid is
    short by 4e-7 at 1.3 goals a side and by 4.4e-4 at 3.15 - four hundredths
    of a percentage point, invisible on a card but not nothing, and it grows
    with the rate. The modal scoreline still comes off the grid, because it has
    no closed form and an argmax does not care about the missing tail.
    """
    errors = {}
    for home, away in RATES:
        analytic = scorelines.probability_over(2.5, home, away)[0]
        joint = scorelines.joint_matrix(home, away, max_goals=10)[0]
        goals = np.arange(11)
        totals = goals[:, None] + goals[None, :]
        errors[(home, away)] = abs(joint[totals > 2.5].sum() - analytic)
        # The grid always understates, never overstates: it drops the tail.
        assert joint[totals > 2.5].sum() <= analytic + 1e-12
    assert max(errors.values()) < 5e-4
    assert errors[(1.27, 1.40)] < 1e-6
    # A wider grid converges on the closed form.
    joint = scorelines.joint_matrix(3.15, 0.56, max_goals=40)[0]
    goals = np.arange(41)
    totals = goals[:, None] + goals[None, :]
    assert joint[totals > 2.5].sum() == pytest.approx(
        scorelines.probability_over(2.5, 3.15, 0.56)[0], abs=1e-12)


def test_probability_over_uses_the_betting_convention():
    """Over 2.5 means three or more goals, not more than two and a half."""
    home, away = 1.4, 1.2
    assert scorelines.probability_over(2.5, home, away)[0] == pytest.approx(
        1 - poisson.cdf(2, home + away))
    # Over 0.5 is simply "at least one goal".
    assert scorelines.probability_over(0.5, home, away)[0] == pytest.approx(
        1 - np.exp(-(home + away)))


def test_a_margin_of_one_is_just_winning():
    """P(win by 1 or more) must be P(win), by two different routes.

    The published 1X2 probabilities are sums over the truncated grid, then
    renormalised; the Skellam form is exact. They agree to 5.6e-5 at the
    highest rates in the draw and to 2.4e-8 at the lowest, which is the size of
    the approximation sitting behind every 1X2 number the app shows.
    """
    for home, away in RATES:
        exact = scorelines.probability_margin_at_least(1, home, away)[0]
        published = probabilities_from_rates([home], [away])[0][0]
        assert exact == pytest.approx(published, abs=1e-4)


def test_the_modal_scoreline_reproduces_the_simulators_own_arithmetic():
    """The refactor must not have moved a single published modal scoreline."""
    saved = pd.read_csv("outputs/next_matchday_1_2026-27.csv")
    home, away, probability = scorelines.modal_scoreline(saved.xg_home, saved.xg_away)
    assert [f"{h}-{a}" for h, a in zip(home, away)] == saved.modal_score.tolist()
    assert probability == pytest.approx(saved.p_modal.to_numpy(), abs=1e-6)


def test_the_summary_agrees_with_the_published_table():
    saved = pd.read_csv("outputs/next_matchday_1_2026-27.csv")
    derived = scorelines.summarise_fixture(saved.xg_home, saved.xg_away)
    assert derived["probability_over"] == pytest.approx(
        saved.p_over_2_5.to_numpy(), abs=1e-6)
    assert derived["probability_favourite_margin"] == pytest.approx(
        saved.p_favourite_by_3.to_numpy(), abs=1e-6)
    favoured = np.where(derived["home_favoured"], saved.home, saved.away)
    assert favoured.tolist() == saved.favourite.tolist()


def test_the_favourite_is_the_side_more_likely_to_win():
    saved = pd.read_csv("outputs/next_matchday_1_2026-27.csv")
    for row in saved.itertuples():
        expected = row.home if row.p_home >= row.p_away else row.away
        assert row.favourite == expected


def test_the_margin_reported_belongs_to_the_favoured_side():
    """An away favourite's 3+ chance is its own, not the home side's."""
    home, away = 0.91, 1.95                  # Napoli v Arsenal: away favoured
    derived = scorelines.summarise_fixture(home, away)
    assert not derived["home_favoured"][0]
    assert derived["probability_favourite_margin"][0] == pytest.approx(
        scorelines.probability_margin_at_least(3, away, home)[0])
    # And it is not the home side's, which here is much smaller.
    assert derived["probability_favourite_margin"][0] > \
        scorelines.probability_margin_at_least(3, home, away)[0]


def test_the_tail_numbers_separate_fixtures_the_modal_score_cannot():
    """The reason these columns exist, as an assertion.

    Lille v Betis and PSG v Slovan both have a modal scoreline carrying about
    12%, and the modal score alone says nothing about how different they are.
    """
    close = scorelines.summarise_fixture(1.40, 1.26)
    rout = scorelines.summarise_fixture(3.15, 0.56)
    assert close["probability_modal"][0] == pytest.approx(0.123, abs=0.01)
    assert rout["probability_modal"][0] == pytest.approx(0.127, abs=0.01)
    assert close["probability_favourite_margin"][0] < 0.10
    assert rout["probability_favourite_margin"][0] > 0.45
    assert rout["probability_over"][0] - close["probability_over"][0] > 0.20


def test_everything_is_vectorised_over_fixtures():
    home = [h for h, _ in RATES]
    away = [a for _, a in RATES]
    derived = scorelines.summarise_fixture(home, away)
    assert len(derived["probability_over"]) == len(RATES)
    for index, (one, other) in enumerate(RATES):
        single = scorelines.summarise_fixture(one, other)
        assert derived["probability_over"][index] == pytest.approx(
            single["probability_over"][0])
        assert derived["probability_favourite_margin"][index] == pytest.approx(
            single["probability_favourite_margin"][0])


def test_a_goalless_model_is_handled_rather_than_dividing_by_zero():
    derived = scorelines.summarise_fixture(0.0, 0.0)
    assert derived["probability_over"][0] == pytest.approx(0.0)
    assert derived["probability_modal"][0] == pytest.approx(1.0)
    assert derived["modal_home"][0] == 0 and derived["modal_away"][0] == 0
