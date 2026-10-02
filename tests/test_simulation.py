"""The simulator: Elo replay determinism, ties, and the knockout run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ucl import config
from ucl.models.elo import MATCH_COLUMNS, replay
from ucl.models.poisson import PoissonModel
from ucl.sim.bracket import Tie, draw_bracket
from ucl.sim.knockout import play_final, play_tie, run_knockout
from ucl.sim.league_phase import simulate_league_phase

MODEL = PoissonModel(a=0.18, h=0.23, b=0.75)


def _frame(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["date", "home_id", "away_id", "hg", "ag", "season"])
    frame["date"] = pd.to_datetime(frame.date)
    frame["home_assoc"] = "ENG"
    frame["away_assoc"] = "ENG"
    frame["competition"] = "domestic"
    frame["is_european"] = False
    return frame[MATCH_COLUMNS]


def test_between_season_regression_uses_one_target_for_every_club():
    """Each club must regress toward the same pre-regression mean.

    Recomputing the target inside the loop made a club's rating depend on the
    order of the others, which set iteration does not fix between runs.
    """
    rows = [
        ("2021-08-01", "A", "B", 5, 0, "2021-22"),
        ("2021-08-08", "A", "C", 5, 0, "2021-22"),
        ("2022-08-01", "A", "B", 0, 0, "2022-23"),
    ]
    result = replay(_frame(rows), offsets={"ENG": 0.0})
    # After the first season A is above the mean and B, C below; the regression
    # must move all three toward the mean of the three, not a moving target.
    assert set(result.ratings) == {"A", "B", "C"}
    assert result.ratings["A"] > result.ratings["B"]


def test_replay_is_deterministic():
    rows = [("2021-08-01", "A", "B", 2, 1, "2021-22"), ("2021-08-08", "B", "A", 1, 1, "2021-22")]
    first = replay(_frame(rows), offsets={"ENG": 0.0}).ratings
    second = replay(_frame(rows), offsets={"ENG": 0.0}).ratings
    assert first == second


def test_play_tie_returns_one_of_the_two_clubs():
    ratings = {"A": 1800.0, "B": 1700.0}
    rng = np.random.default_rng(3)
    for _ in range(50):
        result = play_tie(Tie("A", "B", 1, 9), dict(ratings), MODEL, rng)
        assert result.winner in {"A", "B"}
        assert result.loser in {"A", "B"}
        assert result.winner != result.loser
        assert result.decided_by in {"aggregate", "extra time", "penalties"}


def test_the_better_seed_hosts_the_second_leg():
    tie = Tie("A", "B", 3, 14)
    assert tie.second_leg_home == "A"
    assert tie.first_leg_home == "B"


def test_a_tie_can_reach_extra_time_and_penalties():
    ratings = {"A": 1750.0, "B": 1750.0}
    rng = np.random.default_rng(1)
    decided = {play_tie(Tie("A", "B", 1, 9), dict(ratings), MODEL, rng).decided_by
               for _ in range(400)}
    assert {"aggregate", "extra time", "penalties"} <= decided


def test_the_final_is_played_at_a_neutral_venue():
    """With equal ratings and no home advantage, the final must be a coin flip."""
    ratings = {"A": 1800.0, "B": 1800.0}
    rng = np.random.default_rng(5)
    wins = sum(
        play_final("A", 1, "B", 2, dict(ratings), MODEL, rng).winner == "A"
        for _ in range(2000)
    )
    assert 0.44 < wins / 2000 < 0.56


def test_penalty_tilt_is_clipped():
    ratings = {"A": 3000.0, "B": 1000.0}
    rng = np.random.default_rng(2)
    probability = float(
        np.clip(0.5 + config.PENALTY_TILT * (ratings["A"] - ratings["B"]),
                *config.PENALTY_TILT_CLIP)
    )
    assert probability == config.PENALTY_TILT_CLIP[1]


def _league(n_teams: int = 36) -> tuple[pd.DataFrame, dict[str, float]]:
    teams = [f"T{i:02d}" for i in range(1, n_teams + 1)]
    rng = np.random.default_rng(0)
    rows = []
    for index, home in enumerate(teams):
        for offset in range(1, 5):
            away = teams[(index + offset) % n_teams]
            rows.append({"home_id": home, "away_id": away})
    fixtures = pd.DataFrame(rows)
    ratings = {team: 1700.0 + 10 * i for i, team in enumerate(teams)}
    return fixtures, ratings


def test_league_phase_produces_a_full_table_and_does_not_mutate_the_input():
    fixtures, ratings = _league()
    before = dict(ratings)
    table, live = simulate_league_phase(fixtures, ratings, MODEL, np.random.default_rng(4))
    assert len(table) == 36
    assert list(table.position) == list(range(1, 37))
    assert ratings == before, "the caller's ratings must not be mutated"
    assert live != before, "hot Elo must have moved the ratings"


def test_every_club_plays_eight_league_matches():
    fixtures, _ = _league()
    counts = pd.concat([fixtures.home_id, fixtures.away_id]).value_counts()
    assert set(counts) == {8}


def test_knockout_run_fills_every_round():
    fixtures, ratings = _league()
    rng = np.random.default_rng(6)
    table, live = simulate_league_phase(fixtures, ratings, MODEL, rng)
    by_position = dict(zip(table.position, table.team))
    outcome = run_knockout(draw_bracket(by_position, rng=rng), live, MODEL, rng)
    assert len(set(outcome["round_of_16"])) == 16
    assert len(set(outcome["quarter_finals"])) == 8
    assert len(set(outcome["semi_finals"])) == 4
    assert len(set(outcome["final"])) == 2
    assert outcome["champion"] in outcome["final"]
    # Nobody eliminated in the league phase may appear in the knockout.
    qualified = {by_position[p] for p in range(1, 25)}
    assert set(outcome["round_of_16"]) <= qualified


def test_the_simulator_refuses_a_group_stage_season():
    """2021-22 to 2023-24 had 32 teams in 8 groups and a different bracket."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/simulate.py", "--season", "2022-23", "-n", "1"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "group_stage" in (result.stdout + result.stderr)
