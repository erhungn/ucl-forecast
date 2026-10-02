"""The in-season update: played matches are fixed, the rest is simulated."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ucl.data import fixtures as fx
from ucl.data import league_results
from ucl.models.poisson import PoissonModel
from ucl.sim.league_phase import simulate_league_phase

MODEL = PoissonModel(a=0.18, h=0.23, b=0.75)


def test_played_results_enter_the_table_unchanged():
    fixtures = pd.DataFrame({"home_id": ["C", "D"], "away_id": ["D", "C"]})
    played = pd.DataFrame({"home_id": ["A"], "away_id": ["B"], "hg": [4], "ag": [0]})
    ratings = {t: 1700.0 for t in "ABCD"}
    table, _ = simulate_league_phase(
        fixtures, ratings, MODEL, np.random.default_rng(0), played=played)
    a = table[table.team == "A"].iloc[0]
    b = table[table.team == "B"].iloc[0]
    assert (a.gf, a.ga, a.points) == (4, 0, 3)
    assert (b.gf, b.ga, b.points) == (0, 4, 0)


def test_a_club_never_drops_below_the_points_it_has_actually_earned():
    fixtures = pd.DataFrame({"home_id": ["A"], "away_id": ["C"]})
    played = pd.DataFrame({"home_id": ["A", "A"], "away_id": ["B", "D"],
                           "hg": [2, 1], "ag": [0, 0]})
    ratings = {t: 1700.0 for t in "ABCD"}
    for seed in range(20):
        table, _ = simulate_league_phase(
            fixtures, ratings, MODEL, np.random.default_rng(seed), played=played)
        assert table[table.team == "A"].iloc[0].points >= 6


def test_played_matches_are_not_resimulated():
    fixtures = pd.DataFrame({"home_id": [], "away_id": []}, dtype=object)
    played = pd.DataFrame({"home_id": ["A"], "away_id": ["B"], "hg": [1], "ag": [1]})
    ratings = {"A": 1700.0, "B": 1700.0}
    first, _ = simulate_league_phase(
        fixtures, ratings, MODEL, np.random.default_rng(0), played=played)
    second, _ = simulate_league_phase(
        fixtures, ratings, MODEL, np.random.default_rng(99), played=played)
    assert first.sort_values("team").points.tolist() == \
           second.sort_values("team").points.tolist()


def _run(*arguments):
    return subprocess.run([sys.executable, "scripts/simulate.py", *arguments],
                          capture_output=True, text=True)


def test_matchday_zero_matches_a_plain_run():
    """--through-matchday 0 must be the pre-tournament forecast, unchanged."""
    plain = _run("--season", "2026-27", "-n", "25", "--out", "t_plain.csv")
    zero = _run("--season", "2026-27", "-n", "25", "--through-matchday", "0",
                "--out", "t_zero.csv")
    assert plain.returncode == 0 and zero.returncode == 0, plain.stderr + zero.stderr
    a = pd.read_csv("outputs/t_plain.csv")
    b = pd.read_csv("outputs/t_zero.csv")
    pd.testing.assert_frame_equal(a, b)


def test_an_incomplete_matchday_is_refused():
    fixtures = pd.read_csv(fx.fixtures_path("2026-27"))
    played = league_results.load_played("2026-27")
    highest = int(played.matchday.max())
    result = _run("--season", "2026-27", "-n", "5",
                  "--through-matchday", str(highest + 1))
    assert result.returncode != 0
    assert "not complete" in (result.stdout + result.stderr)


def test_the_update_run_fixes_the_played_matches_and_simulates_the_rest():
    result = _run("--season", "2026-27", "-n", "25", "--through-matchday", "1",
                  "--out", "t_md1.csv")
    assert result.returncode == 0, result.stderr
    assert "fixed at their actual scores" in result.stdout
    assert "126 fixtures to simulate" in result.stdout
    frame = pd.read_csv("outputs/t_md1.csv")
    assert len(frame) == 36
    assert frame.p_champion.sum() == pytest.approx(1.0, abs=1e-9)


def test_the_information_set_is_printed():
    result = _run("--season", "2026-27", "-n", "5", "--through-matchday", "1")
    assert "information set" in result.stdout
    assert "domestic/football_data" in result.stdout
    assert "european/wikipedia" in result.stdout
