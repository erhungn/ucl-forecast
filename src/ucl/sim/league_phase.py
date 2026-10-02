"""Simulation of the 144-match league phase and the single table it produces.

Ratings update inside the simulation after every match, as CLAUDE.md requires,
so a club that over- or under-performs early carries that into its later
fixtures. The ratings passed in are never mutated: each iteration works on its
own copy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ucl import config
from ucl.models.elo import goal_difference_multiplier
from ucl.models.poisson import PoissonModel
from ucl.sim.tiebreakers import rank_table

__all__ = ["simulate_league_phase", "simulate_match"]


def simulate_match(
    home: str,
    away: str,
    ratings: dict[str, float],
    model: PoissonModel,
    rng: np.random.Generator,
    *,
    neutral: bool = False,
    scale: float = 1.0,
    update: bool = True,
) -> tuple[int, int]:
    """Draw one scoreline and, by default, apply the Elo update for it."""
    elo_home, elo_away = ratings[home], ratings[away]
    rate_home, rate_away = model.rates(elo_home, elo_away, neutral=neutral)
    hg = int(rng.poisson(float(rate_home) * scale))
    ag = int(rng.poisson(float(rate_away) * scale))

    if update:
        advantage = 0.0 if neutral else config.HOME_ADVANTAGE
        expected = 1.0 / (1.0 + 10.0 ** (-(elo_home + advantage - elo_away) / 400.0))
        actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        change = (
            config.K_EUROPEAN
            * goal_difference_multiplier(hg - ag)
            * (actual - expected)
        )
        ratings[home] = elo_home + change
        ratings[away] = elo_away - change
    return hg, ag


def simulate_league_phase(
    fixtures: pd.DataFrame,
    ratings: dict[str, float],
    model: PoissonModel,
    rng: np.random.Generator,
    *,
    coefficients: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Play all 144 fixtures in matchday order and rank the resulting table.

    Returns the ranked table and the ratings as they stand afterwards, which the
    knockout phase then continues from.
    """
    live = dict(ratings)
    rows = []
    for row in fixtures.itertuples():
        hg, ag = simulate_match(row.home_id, row.away_id, live, model, rng)
        rows.append({"home": row.home_id, "away": row.away_id, "hg": hg, "ag": ag})

    results = pd.DataFrame(rows, columns=["home", "away", "hg", "ag"])
    table = rank_table(results, coefficients=coefficients, seed=int(rng.integers(1 << 32)))
    return table, live
