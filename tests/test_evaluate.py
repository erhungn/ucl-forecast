"""The metrics, against values worked out by hand."""

from __future__ import annotations

import numpy as np
import pytest

from ucl.evaluate import brier, calibration, log_loss, rps


def test_log_loss_of_a_certain_correct_forecast_is_zero():
    assert log_loss([[1.0, 0.0, 0.0]], [0]) == pytest.approx(0.0, abs=1e-12)


def test_log_loss_matches_the_hand_calculation():
    # -log(0.5) for the first, -log(0.25) for the second.
    expected = (-np.log(0.5) - np.log(0.25)) / 2
    assert log_loss([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]], [0, 1]) == pytest.approx(expected)


def test_a_uniform_forecast_scores_log_three():
    assert log_loss([[1 / 3, 1 / 3, 1 / 3]], [1]) == pytest.approx(np.log(3))


def test_rps_is_zero_for_a_certain_correct_forecast():
    assert rps([[1.0, 0.0, 0.0]], [0]) == pytest.approx(0.0)


def test_rps_punishes_a_distant_miss_more_than_a_near_one():
    # The home team wins. Predicting a draw is a smaller error than away.
    near = rps([[0.0, 1.0, 0.0]], [0])
    far = rps([[0.0, 0.0, 1.0]], [0])
    assert far > near


def test_rps_matches_the_hand_calculation():
    # p = (0.5, 0.3, 0.2), outcome home. Cumulative (0.5, 0.8) vs (1, 1).
    expected = ((0.5 - 1) ** 2 + (0.8 - 1) ** 2) / 2
    assert rps([[0.5, 0.3, 0.2]], [0]) == pytest.approx(expected)


def test_brier_matches_the_hand_calculation():
    expected = (0.5 - 1) ** 2 + 0.3 ** 2 + 0.2 ** 2
    assert brier([[0.5, 0.3, 0.2]], [0]) == pytest.approx(expected)


def test_a_perfectly_calibrated_forecast_has_near_zero_error():
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet([2.0, 2.0, 2.0], size=20000)
    outcomes = [rng.choice(3, p=row) for row in probabilities]
    _, error = calibration(probabilities, outcomes, bins=10)
    assert error < 0.01


def test_a_badly_calibrated_forecast_is_caught():
    rng = np.random.default_rng(1)
    probabilities = np.tile([0.8, 0.1, 0.1], (2000, 1))
    outcomes = rng.choice(3, size=2000, p=[0.34, 0.33, 0.33])
    _, error = calibration(probabilities, outcomes, bins=10)
    assert error > 0.1


def test_probabilities_are_renormalised_before_scoring():
    assert log_loss([[2.0, 1.0, 1.0]], [0]) == pytest.approx(log_loss([[0.5, 0.25, 0.25]], [0]))
