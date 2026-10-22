"""Quantities read off the scoreline distribution.

The match model gives two Poisson rates, and everything a fixture card shows
follows from them. Two of these have exact closed forms, which is worth using
rather than summing a truncated grid:

* **total goals.** The sum of independent Poissons is Poisson, so
  ``H + A ~ Poisson(rate_home + rate_away)`` and P(over 2.5) is exact.
* **winning margin.** The difference of independent Poissons is Skellam, so
  P(win by three or more) is exact.

The scoreline matrix is still built for the modal scoreline, which has no
closed form, and the tests check the analytic answers against sums over that
matrix - two implementations of the same arithmetic, which is the point.

A caution worth keeping attached to these numbers: the modal scoreline is
almost always 1-0, 1-1 or 2-0 whatever the fixture, because no single scoreline
carries much probability. The expected goals and the tail probabilities are
what actually separate a one-sided fixture from a close one.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson, skellam

__all__ = [
    "MAX_GOALS", "BANDS", "joint_matrix", "folded_matrix", "modal_scoreline",
    "probability_over", "probability_margin_at_least", "scoreline_log_loss",
    "summarise_fixture"]

#: The grid the forecast has always used for scorelines.
MAX_GOALS = 10

#: Goals shown individually before the tail is gathered: 0-4, then "5 or more".
BANDS = 5


def _rates(rate_home, rate_away) -> tuple[np.ndarray, np.ndarray]:
    return (np.atleast_1d(np.asarray(rate_home, dtype=float)),
            np.atleast_1d(np.asarray(rate_away, dtype=float)))


def joint_matrix(rate_home, rate_away, *, max_goals: int = MAX_GOALS) -> np.ndarray:
    """``(n, max_goals + 1, max_goals + 1)`` of P(home scores h, away scores a)."""
    home, away = _rates(rate_home, rate_away)
    goals = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goals[None, :], home[:, None])
    away_pmf = poisson.pmf(goals[None, :], away[:, None])
    return home_pmf[:, :, None] * away_pmf[:, None, :]


def folded_matrix(rate_home, rate_away, *, bands: int = BANDS) -> np.ndarray:
    """``(n, bands + 1, bands + 1)`` of scoreline probabilities summing to 1.

    Goals 0..bands-1 keep a row and column of their own and everything above
    is gathered into the last one, whose mass is the exact Poisson survival
    function rather than a truncated sum. That matters for display: a bare
    0-4 x 0-4 grid holds only about 85% of the mass on a one-sided fixture, and
    a grid of absolute percentages that does not add up is worse than one that
    says "5 or more".
    """
    home, away = _rates(rate_home, rate_away)
    goals = np.arange(bands)
    home_pmf = poisson.pmf(goals[None, :], home[:, None])
    away_pmf = poisson.pmf(goals[None, :], away[:, None])
    # The tail, exactly: P(X >= bands) = sf(bands - 1).
    home_full = np.concatenate([home_pmf, poisson.sf(bands - 1, home)[:, None]], axis=1)
    away_full = np.concatenate([away_pmf, poisson.sf(bands - 1, away)[:, None]], axis=1)
    return home_full[:, :, None] * away_full[:, None, :]


def modal_scoreline(rate_home, rate_away, *, max_goals: int = MAX_GOALS):
    """The single likeliest scoreline, and its probability.

    Returns ``(home_goals, away_goals, probability)`` as arrays. It is reported
    because it is asked for, not because it is informative: the modal scoreline
    rarely clears 13% and is 1-0, 1-1 or 2-0 for most fixtures.
    """
    joint = joint_matrix(rate_home, rate_away, max_goals=max_goals)
    flat = joint.reshape(len(joint), -1).argmax(axis=1)
    home_goals, away_goals = np.unravel_index(flat, joint.shape[1:])
    probability = joint.reshape(len(joint), -1).max(axis=1)
    return home_goals, away_goals, probability


def probability_over(line: float, rate_home, rate_away) -> np.ndarray:
    """P(total goals above ``line``), exactly.

    ``line`` is a betting line such as 2.5, so "over" means at least the next
    whole goal above it.
    """
    home, away = _rates(rate_home, rate_away)
    threshold = int(np.floor(line))
    return 1.0 - poisson.cdf(threshold, home + away)


def scoreline_log_loss(rate_home, rate_away, hg, ag) -> np.ndarray:
    """-log P(exact score) under the independent-Poisson model, exactly.

    This is the quantity the model is fitted on, one match at a time, so it is
    the natural metric for a candidate whose claim is about the goal scale: a
    1X2 triple is a coarse functional of the same distribution and a scale shift
    barely moves it.

    Deliberately **not** read off :func:`joint_matrix`, which truncates at
    ``MAX_GOALS`` and whose 1X2 projection is renormalised. This is the model's
    own likelihood with no grid and no clipping, so it and the 1X2 loss are not
    two views of one truncated table - a 12-1 scoreline has a loss here and no
    cell there.
    """
    home, away = _rates(rate_home, rate_away)
    return -(poisson.logpmf(np.asarray(hg), home)
             + poisson.logpmf(np.asarray(ag), away))


def probability_margin_at_least(margin: int, rate_home, rate_away) -> np.ndarray:
    """P(home wins by ``margin`` or more), exactly, via the Skellam law."""
    home, away = _rates(rate_home, rate_away)
    return 1.0 - skellam.cdf(margin - 1, home, away)


def summarise_fixture(rate_home, rate_away, *, line: float = 2.5,
                      margin: int = 3, max_goals: int = MAX_GOALS) -> dict:
    """Everything a fixture card reports beyond home/draw/away.

    ``favourite`` is the side with the better chance of winning, and
    ``probability_favourite_margin`` is that side's chance of winning by
    ``margin`` or more - the number that distinguishes a likely win from a
    likely rout.
    """
    home, away = _rates(rate_home, rate_away)
    home_goals, away_goals, modal = modal_scoreline(home, away, max_goals=max_goals)

    joint = joint_matrix(home, away, max_goals=max_goals)
    upper = np.triu(np.ones((max_goals + 1, max_goals + 1)), 1)
    home_wins = (joint * upper.T).sum(axis=(1, 2))
    away_wins = (joint * upper).sum(axis=(1, 2))
    home_favoured = home_wins >= away_wins

    by_margin = np.where(
        home_favoured,
        probability_margin_at_least(margin, home, away),
        probability_margin_at_least(margin, away, home),
    )
    return {
        "expected_home": home,
        "expected_away": away,
        "modal_home": home_goals,
        "modal_away": away_goals,
        "probability_modal": modal,
        "probability_over": probability_over(line, home, away),
        "home_favoured": home_favoured,
        "probability_favourite_margin": by_margin,
    }
