"""The significance analysis: its reproduction gate and its determinism.

The gate is the load-bearing test. Statistics computed from per-match losses
that do not add up to the archived deltas would be statistics about the wrong
numbers, and they would look perfectly reasonable.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ucl import significance as sig
from ucl.config import BACKTEST_SEASONS, PROJECT_ROOT

SEASONS = tuple(BACKTEST_SEASONS[1:])
ARCHIVES = {
    ("v1", "elo"): "results/model_v1/candidate_squad_value_decision.json",
    ("v2", "v1"): "results/candidate_gap/candidate_gap_decision.json",
    ("v2+B", "v2"): "results/candidate_gap_minutes/candidate_gap_minutes_decision.json",
}


@pytest.fixture(scope="module")
def losses() -> pd.DataFrame:
    return sig.per_match_losses(SEASONS, ("elo", "v1", "v2", "v2+B"))


@pytest.mark.parametrize("pair", list(ARCHIVES))
def test_the_per_match_losses_reproduce_the_archived_delta(pair, losses):
    """To the last digit, or the export is wrong."""
    model, reference = pair
    archived = json.load(
        open(PROJECT_ROOT / ARCHIVES[pair]))["cluster"]["difference"]
    differences = sig.paired_differences(losses, model, reference)
    got = sig.weighted_mean(differences.d, differences.weight)
    assert got == pytest.approx(archived, abs=1e-12)


def test_the_models_are_scored_on_identical_matches(losses):
    counts = losses.groupby("model").match_id.nunique()
    assert counts.nunique() == 1
    assert int(counts.iloc[0]) == 19188
    first = set(losses[losses.model == "v1"].match_id)
    assert set(losses[losses.model == "v2"].match_id) == first


def test_a_mismatched_pairing_is_refused(losses):
    trimmed = pd.concat([losses[losses.model == "v1"],
                         losses[losses.model == "v2"].iloc[:-5]])
    with pytest.raises(ValueError, match="same matches"):
        sig.paired_differences(trimmed, "v2", "v1")


def test_the_blocks_are_season_weeks_and_there_are_enough_of_them(losses):
    differences = sig.paired_differences(losses, "v2", "v1")
    assert differences.block.nunique() == 155
    # A block must not straddle a refit: the ratings are rebuilt per season.
    for block, group in differences.groupby("block"):
        assert group.season.nunique() == 1, block
    assert differences.cluster.nunique() == 1309


def test_the_bootstrap_is_deterministic_under_a_fixed_seed(losses):
    differences = sig.paired_differences(losses, "v2", "v1")
    first = sig.block_bootstrap(differences, draws=400, seed=20260907)
    second = sig.block_bootstrap(differences, draws=400, seed=20260907)
    assert first == second
    other = sig.block_bootstrap(differences, draws=400, seed=1)
    assert other["mean"] == pytest.approx(first["mean"])      # the point estimate
    assert other["low"] != first["low"]                       # the interval resamples


def test_more_draws_tighten_the_bootstrap_standard_error(losses):
    differences = sig.paired_differences(losses, "v2", "v1")
    few = sig.block_bootstrap(differences, draws=200, seed=7)
    many = sig.block_bootstrap(differences, draws=4000, seed=7)
    assert few["mean"] == pytest.approx(many["mean"])
    assert abs(few["standard_error"] - many["standard_error"]) < 5e-5


def test_the_bootstrap_mean_is_the_weighted_mean_it_reports(losses):
    differences = sig.paired_differences(losses, "v2", "v1")
    out = sig.block_bootstrap(differences, draws=200, seed=3)
    assert out["mean"] == pytest.approx(
        sig.weighted_mean(differences.d, differences.weight))
    unweighted = sig.block_bootstrap(differences, draws=200, seed=3,
                                     weighted=False)
    assert unweighted["mean"] == pytest.approx(differences.d.mean())


def test_the_newey_west_lag_rule_is_the_published_one():
    # floor(4 (n/100)^(2/9)): 930 days -> 6, 19,188 matches -> 12.
    assert sig.newey_west_lags(930) == 6
    assert sig.newey_west_lags(19188) == 12
    assert sig.newey_west_lags(100) == 4


def test_diebold_mariano_recovers_a_known_signal():
    """A constant improvement with no noise must be overwhelmingly significant."""
    dates = pd.date_range("2022-09-01", periods=400, freq="D")
    frame = pd.DataFrame({"date": dates, "d": -0.01, "weight": 1.0,
                          "season": "2022-23", "competition": "domestic"})
    out = sig.diebold_mariano(frame, daily=True)
    assert out["mean"] == pytest.approx(-0.01)
    assert out["p_value"] < 1e-6


def test_diebold_mariano_finds_nothing_in_pure_noise():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2022-09-01", periods=600, freq="D")
    frame = pd.DataFrame({"date": dates, "d": rng.normal(0, 0.5, 600),
                          "weight": 1.0, "season": "2022-23",
                          "competition": "domestic"})
    out = sig.diebold_mariano(frame, daily=True)
    assert out["p_value"] > 0.05


def test_the_daily_basis_tests_a_different_quantity_than_the_weighted_mean(losses):
    """Why the two tests can disagree, asserted rather than asserted away.

    Days carry between 1 and 82 matches. Averaging within a day and then across
    days gives a light midweek fixture the same weight as a full Saturday, and
    drops the x3 European weighting, so it is a smaller quantity measured less
    precisely - not a contradiction of the bootstrap.
    """
    differences = sig.paired_differences(losses, "v2", "v1")
    per_day = differences.groupby(differences.date.dt.date).d.agg(["mean", "size"])
    assert per_day["size"].min() < 5 and per_day["size"].max() > 50
    weighted = abs(sig.weighted_mean(differences.d, differences.weight))
    daily = abs(per_day["mean"].mean())
    assert daily < weighted


def test_the_two_way_bootstrap_is_wider_than_either_one_way(losses):
    """Admitting both margins cannot tighten the interval.

    A club's rating error travels with it whether it is at home or away, so a
    one-way cluster bootstrap over home clubs treats as independent two halves
    of the same dependence. The pigeonhole scheme admits both and must pay for
    it in width.
    """
    differences = sig.paired_differences(losses, "v2", "v1")
    one_way = sig.cluster_bootstrap(differences, draws=600, seed=11)
    two_way = sig.two_way_bootstrap(differences, draws=600, seed=11)
    assert two_way["mean"] == pytest.approx(one_way["mean"])
    assert (two_way["high"] - two_way["low"]) > (one_way["high"] - one_way["low"])
    assert two_way["home_clusters"] == 1309
    assert two_way["away_clusters"] >= 1300


def test_the_two_way_bootstrap_is_deterministic_under_a_fixed_seed(losses):
    differences = sig.paired_differences(losses, "v2", "v1")
    first = sig.two_way_bootstrap(differences, draws=300, seed=20260907)
    second = sig.two_way_bootstrap(differences, draws=300, seed=20260907)
    assert first == second


def test_club_names_containing_a_slash_survive_clustering(losses):
    """Bodo/Glimt must cluster as itself, not as "Bodo"."""
    differences = sig.paired_differences(losses, "v2", "v1")
    clusters = set(differences.home_cluster) | set(differences.away_cluster)
    assert any(name.endswith("/Bodo/Glimt") for name in clusters)
    assert not any(name.endswith("/Bodo") for name in clusters)


def test_per_season_covers_every_fold(losses):
    frame = sig.per_season(sig.paired_differences(losses, "v2", "v1"))
    assert list(frame.season) == list(SEASONS)
    assert frame.n.sum() == 19188
    assert (frame.european > 0).all()


def test_standardised_coefficients_put_the_terms_in_one_unit():
    frame = sig.standardised_coefficients(("2026-27",))
    by_name = frame.set_index("covariate")
    # Elo dominates; squad value next; the shot terms are the smallest.
    assert by_name.loc["d_elo", "standardised"] > \
        by_name.loc["d_value", "standardised"]
    assert by_name.loc["d_value", "standardised"] > \
        by_name.loc["d_gap_sot", "standardised"]
    # And the standardisation is exactly coefficient times spread.
    for row in frame.itertuples():
        assert row.standardised == pytest.approx(row.coefficient * row.covariate_sd)
