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


def test_a_forecast_carries_no_retroactive_column_unless_asked():
    """The label travels in the file, and only when it is true.

    A snapshot re-forecast after its matchday was played is not a forecast
    anyone could have acted on. It is published to fill the gap a model switch
    leaves, so it has to be readable as what it is - and an ordinary run must
    not carry the column at all, or the byte-identity gate on the published
    models would be measuring this change instead of the model.
    """
    plain = _run("--season", "2026-27", "-n", "5", "--through-matchday", "1",
                 "--out", "t_plain_md1.csv")
    marked = _run("--season", "2026-27", "-n", "5", "--through-matchday", "1",
                  "--retroactive", "--out", "t_retro_md1.csv")
    assert plain.returncode == 0 and marked.returncode == 0, plain.stderr + marked.stderr
    before = pd.read_csv("outputs/t_plain_md1.csv")
    after = pd.read_csv("outputs/t_retro_md1.csv")
    assert "retroactive" not in before.columns
    assert after.retroactive.all()
    assert set(after.retroactive_model) == {"v2"}
    assert "marked retroactive" in marked.stdout
    # Nothing else moves: the same seed, the same iterations, the same numbers.
    pd.testing.assert_frame_equal(
        before, after.drop(columns=["retroactive", "retroactive_model"]))


def test_the_information_set_is_printed():
    # --out matters here: without it this run would overwrite the real
    # forecast_2026-27_md1.csv with five iterations, and simulate.py reads the
    # previous matchday's file to compute its deltas.
    result = _run("--season", "2026-27", "-n", "5", "--through-matchday", "1",
                  "--out", "t_info.csv")
    assert "information set" in result.stdout
    assert "domestic/football_data" in result.stdout
    assert "european/wikipedia" in result.stdout
