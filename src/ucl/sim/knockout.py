"""Two-legged ties, extra time, penalties, and the run through the bracket.

Article 20.03: the greater aggregate over the two legs qualifies. Article 21:
otherwise two 15-minute periods of extra time at the end of the second leg, then
a penalty shoot-out. There is no away-goals rule.

Extra time is simulated by scaling the goal rates to 30 minutes
(:data:`ucl.config.EXTRA_TIME_SCALE`); a shoot-out is close to a coin flip, with
a small tilt toward the better-rated side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ucl import config
from ucl.models.poisson import PoissonModel
from ucl.sim.bracket import Bracket, Tie
from ucl.sim.league_phase import simulate_match

__all__ = ["TieResult", "play_tie", "play_final", "run_knockout"]


@dataclass(frozen=True)
class TieResult:
    winner: str
    loser: str
    winner_seed: int
    decided_by: str


def play_tie(
    tie: Tie,
    ratings: dict[str, float],
    model: PoissonModel,
    rng: np.random.Generator,
) -> TieResult:
    """Play a two-legged tie. The lower seeding position hosts the second leg."""
    low, high = tie.team_low, tie.team_high       # low = worse seed, hosts leg 1
    first_home, first_away = simulate_match(low, high, ratings, model, rng)
    second_home, second_away = simulate_match(high, low, ratings, model, rng)

    aggregate_high = first_away + second_home
    aggregate_low = first_home + second_away
    if aggregate_high != aggregate_low:
        winner = high if aggregate_high > aggregate_low else low
        return TieResult(
            winner=winner,
            loser=low if winner == high else high,
            winner_seed=tie.seed_high if winner == high else tie.seed_low,
            decided_by="aggregate",
        )

    # Extra time at the end of the second leg, so the second-leg host is at home.
    extra_home, extra_away = simulate_match(
        high, low, ratings, model, rng, scale=config.EXTRA_TIME_SCALE
    )
    if extra_home != extra_away:
        winner = high if extra_home > extra_away else low
        return TieResult(
            winner=winner,
            loser=low if winner == high else high,
            winner_seed=tie.seed_high if winner == high else tie.seed_low,
            decided_by="extra time",
        )

    difference = ratings[high] - ratings[low]
    probability = float(
        np.clip(0.5 + config.PENALTY_TILT * difference, *config.PENALTY_TILT_CLIP)
    )
    winner = high if rng.random() < probability else low
    return TieResult(
        winner=winner,
        loser=low if winner == high else high,
        winner_seed=tie.seed_high if winner == high else tie.seed_low,
        decided_by="penalties",
    )


def play_final(
    team_a: str,
    seed_a: int,
    team_b: str,
    seed_b: int,
    ratings: dict[str, float],
    model: PoissonModel,
    rng: np.random.Generator,
) -> TieResult:
    """One match at a neutral venue, then extra time, then penalties."""
    goals_a, goals_b = simulate_match(team_a, team_b, ratings, model, rng, neutral=True)
    if goals_a == goals_b:
        extra_a, extra_b = simulate_match(
            team_a, team_b, ratings, model, rng, neutral=True,
            scale=config.EXTRA_TIME_SCALE,
        )
        goals_a += extra_a
        goals_b += extra_b
        decided = "extra time"
    else:
        decided = "90 minutes"

    if goals_a != goals_b:
        winner = team_a if goals_a > goals_b else team_b
        return TieResult(
            winner=winner,
            loser=team_b if winner == team_a else team_a,
            winner_seed=seed_a if winner == team_a else seed_b,
            decided_by=decided,
        )
    difference = ratings[team_a] - ratings[team_b]
    probability = float(
        np.clip(0.5 + config.PENALTY_TILT * difference, *config.PENALTY_TILT_CLIP)
    )
    winner = team_a if rng.random() < probability else team_b
    return TieResult(
        winner=winner,
        loser=team_b if winner == team_a else team_a,
        winner_seed=seed_a if winner == team_a else seed_b,
        decided_by="penalties",
    )


def run_knockout(
    bracket: Bracket,
    ratings: dict[str, float],
    model: PoissonModel,
    rng: np.random.Generator,
) -> dict[str, list[str] | str]:
    """Play the whole knockout phase and report who reached each round."""
    play_off_winner: dict[int, str] = {}
    for tie in bracket.play_offs:
        result = play_tie(tie, ratings, model, rng)
        play_off_winner[tie.seed_high] = result.winner

    # Fill each round-of-16 slot with the qualifier the draw sent there.
    slots: list[tuple[str, int, str, int]] = []
    for tie in bracket.slots:
        qualifier = play_off_winner[tie.seed_low]
        slots.append((tie.team_high, tie.seed_high, qualifier, tie.seed_low))

    round_of_16 = [team for slot in slots for team in (slot[0], slot[2])]
    survivors: list[tuple[str, int]] = []
    for team_high, seed_high, team_low, seed_low in slots:
        result = play_tie(
            Tie(team_high, team_low, seed_high, seed_low), ratings, model, rng
        )
        # The eliminator inherits the beaten seed's position (Article 19.04).
        survivors.append((result.winner, seed_high))

    quarter_finalists = [team for team, _ in survivors]
    semi: list[tuple[str, int]] = []
    for left, right in bracket.quarter_final_slots():
        (team_a, seed_a), (team_b, seed_b) = survivors[left], survivors[right]
        high, low = ((team_a, seed_a), (team_b, seed_b)) if seed_a < seed_b else (
            (team_b, seed_b), (team_a, seed_a))
        result = play_tie(Tie(high[0], low[0], high[1], low[1]), ratings, model, rng)
        semi.append((result.winner, min(seed_a, seed_b)))

    semi_finalists = [team for team, _ in semi]
    finalists: list[tuple[str, int]] = []
    for left, right in ((0, 1), (2, 3)):
        (team_a, seed_a), (team_b, seed_b) = semi[left], semi[right]
        high, low = ((team_a, seed_a), (team_b, seed_b)) if seed_a < seed_b else (
            (team_b, seed_b), (team_a, seed_a))
        result = play_tie(Tie(high[0], low[0], high[1], low[1]), ratings, model, rng)
        finalists.append((result.winner, min(seed_a, seed_b)))

    champion = play_final(
        finalists[0][0], finalists[0][1], finalists[1][0], finalists[1][1],
        ratings, model, rng,
    )
    return {
        "round_of_16": round_of_16,
        "quarter_finals": quarter_finalists,
        "semi_finals": semi_finalists,
        "final": [team for team, _ in finalists],
        "champion": champion.winner,
    }
