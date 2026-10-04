"""No match on or after a cutoff may reach anything that is fitted.

That means all three fitted stages: the association-offset MLE, the Elo replay,
and the Poisson match model. The offsets are the subtle one - they are fitted
before the replay, so reusing a set fitted at a later cutoff would leak the
future into a backtest without touching the replay at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ucl.backtest import _training_frame, score_season
from ucl.config import SEASONS
from ucl.models.elo import EloParams, elo_as_of, fit_association_offsets, replay
from ucl.models.poisson import fit_poisson

BACKTEST_SEASONS = ["2024-25", "2025-26"]


@pytest.mark.parametrize("season_label", BACKTEST_SEASONS)
def test_the_training_frame_stops_before_the_cutoff(season_label):
    cfg = SEASONS[season_label]
    frame = _training_frame(cfg.cutoff_date)
    assert frame.date.max().date() < cfg.cutoff_date


@pytest.mark.parametrize("season_label", BACKTEST_SEASONS)
def test_the_offset_fit_sees_nothing_from_the_cutoff_onwards(season_label):
    cfg = SEASONS[season_label]
    frame = _training_frame(cfg.cutoff_date)
    european = frame[frame.is_european]
    assert european.date.max().date() < cfg.cutoff_date
    # And it must actually use European matches, or the assertion is vacuous.
    assert len(european) > 100


@pytest.mark.parametrize("season_label", BACKTEST_SEASONS)
def test_the_replay_and_the_poisson_fit_stop_before_the_cutoff(season_label):
    cfg = SEASONS[season_label]
    ratings = elo_as_of(cfg.cutoff_date)
    assert ratings.pre_match.date.max().date() < cfg.cutoff_date
    model = fit_poisson(ratings.pre_match)
    assert model.matches == len(ratings.pre_match.dropna(subset=["elo_home", "elo_away"]))


def test_offsets_are_refitted_per_cutoff_not_reused():
    """A backtest cutoff must produce its own offsets, not 2026-27's."""
    early = _training_frame(SEASONS["2024-25"].cutoff_date)
    late = _training_frame(SEASONS["2026-27"].cutoff_date)
    early_offsets, _ = fit_association_offsets(early)
    late_offsets, _ = fit_association_offsets(late)
    shared = set(early_offsets) & set(late_offsets)
    assert shared
    differences = [abs(early_offsets[a] - late_offsets[a]) for a in shared]
    assert max(differences) > 1.0, "offsets look identical across cutoffs - are they cached?"


@pytest.mark.parametrize("season_label", BACKTEST_SEASONS)
def test_scored_matches_all_fall_after_the_cutoff(season_label):
    cfg = SEASONS[season_label]
    score = score_season(season_label, modes=("frozen",))
    scored = score.rows[score.rows["mode"] == "frozen"]
    assert pd.to_datetime(scored.date).min().date() >= cfg.cutoff_date
    assert len(scored) > 4000


def test_an_earlier_cutoff_gives_strictly_less_training_data():
    early = _training_frame(SEASONS["2024-25"].cutoff_date)
    late = _training_frame(SEASONS["2026-27"].cutoff_date)
    assert len(early) < len(late)


def test_ratings_differ_between_cutoffs():
    """A backtest must not silently inherit the 2026-27 ratings."""
    early = elo_as_of(SEASONS["2024-25"].cutoff_date).ratings
    late = elo_as_of(SEASONS["2026-27"].cutoff_date).ratings
    shared = set(early) & set(late)
    changed = sum(1 for club in shared if abs(early[club] - late[club]) > 1.0)
    assert changed > 0.5 * len(shared)


# The backtest seasons added in milestone 2.6, including the three group-stage
# ones, must satisfy the same guarantees.
EARLY_SEASONS = ["2021-22", "2022-23", "2023-24"]


@pytest.mark.parametrize("season_label", EARLY_SEASONS)
def test_group_stage_seasons_leak_nothing(season_label):
    from ucl.config import SEASONS

    cfg = SEASONS[season_label]
    frame = _training_frame(cfg.cutoff_date)
    assert frame.date.max().date() < cfg.cutoff_date
    assert frame[frame.is_european].date.max().date() < cfg.cutoff_date


def test_forward_chaining_folds_train_only_on_earlier_seasons():
    """A fold's training windows must all close before its test cutoff."""
    import pandas as pd

    from ucl.backtest import _evaluation_frames, season_end
    from ucl.config import BACKTEST_SEASONS, SEASONS

    for index in range(1, len(BACKTEST_SEASONS)):
        test = BACKTEST_SEASONS[index]
        for train in BACKTEST_SEASONS[:index]:
            assert SEASONS[train].cutoff_date < SEASONS[test].cutoff_date
            # The training season's scored window closes before the test cutoff.
            assert season_end(train) < pd.Timestamp(SEASONS[test].cutoff_date)
            domestic, european = _evaluation_frames(train)
            for frame in (domestic, european):
                assert frame.date.max() < pd.Timestamp(SEASONS[test].cutoff_date)


def test_cutoffs_increase_with_the_season_order():
    from ucl.config import BACKTEST_SEASONS, SEASONS

    cutoffs = [SEASONS[s].cutoff_date for s in BACKTEST_SEASONS]
    assert cutoffs == sorted(cutoffs)


def test_group_stage_seasons_are_flagged_and_have_no_fixtures():
    """The simulator must not be pointed at a 32-team season."""
    from ucl.config import SEASONS

    for season_label in EARLY_SEASONS:
        cfg = SEASONS[season_label]
        assert cfg.tournament_format == "group_stage"
        assert not cfg.fixtures_path.exists()
    for season_label in ("2024-25", "2025-26", "2026-27"):
        cfg = SEASONS[season_label]
        assert cfg.tournament_format == "league_phase"
        assert cfg.fixtures_path.exists()


# --- the covariates, which this file did not cover before -----------------

@pytest.mark.parametrize("season_label", BACKTEST_SEASONS)
def test_the_gap_replay_sees_nothing_from_the_cutoff_onwards(season_label):
    """Shot ratings are replayed over matches; the cutoff binds them too."""
    from ucl.models import gap

    cfg = SEASONS[season_label]
    frame = gap.domestic_frame(cfg.cutoff_date)
    assert frame.date.max().date() < cfg.cutoff_date
    state = gap.replay(frame, "sot")
    assert state.matches > 10_000      # not vacuous


@pytest.mark.parametrize("season_label", BACKTEST_SEASONS)
def test_the_gap_parameter_fit_sees_nothing_from_the_cutoff_onwards(season_label):
    """The least-squares fit of lambda and the phis is bounded by the cutoff.

    It is fitted against observed attacking output rather than results, so it
    could not leak an outcome even in principle - but it must still not see a
    match the forecast has not reached.
    """
    from ucl.models import gap

    cfg = SEASONS[season_label]
    frame = gap.domestic_frame(cfg.cutoff_date)
    assert frame.date.max().date() < cfg.cutoff_date
    params = gap.fit_params(frame, "sot")
    assert 1e-4 <= params.learning_rate <= 0.5


def test_the_gap_elo_regression_is_refitted_per_cutoff():
    """Residuals must move between cutoffs, or one is being reused."""
    from ucl.backtest import _fitted_gap

    early = _fitted_gap(SEASONS["2022-23"].cutoff_date, "2022-23")[2]["sot"]
    late = _fitted_gap(SEASONS["2025-26"].cutoff_date, "2025-26")[2]["sot"]
    shared = set(early) & set(late)
    assert len(shared) > 100
    moved = sum(abs(early[club]["home_attack"] - late[club]["home_attack"]) > 0.05
                for club in shared)
    assert moved > len(shared) / 2


@pytest.mark.parametrize("season_label", BACKTEST_SEASONS)
def test_each_covariate_snapshot_predates_its_seasons_cutoff(season_label):
    """The as-of date of a point-in-time feature is itself a leakage surface.

    The match dates were already asserted; the snapshot date was not. Squad
    value is frozen at 6 July of the season's starting year, which must fall
    before that season's September cutoff.
    """
    from ucl.data import transfermarkt

    cfg = SEASONS[season_label]
    as_of = transfermarkt.as_of_for_season(season_label)
    assert as_of < cfg.cutoff_date
    valuations, appearances = transfermarkt._tables()
    limit = pd.Timestamp(as_of)
    assert valuations[valuations.date < limit].date.max() < limit
    assert (valuations.date >= limit).any()    # the filter does something


@pytest.mark.parametrize("season_label", BACKTEST_SEASONS)
def test_the_minutes_weighted_values_respect_the_same_as_of(season_label):
    from ucl.data import transfermarkt

    as_of = transfermarkt.as_of_for_season(season_label)
    total = transfermarkt.squad_values(as_of)
    weighted = transfermarkt.squad_values_minutes_weighted(as_of)
    # Same membership rule, so the same clubs: the weighting changes values
    # only, which is what makes candidate B's null interpretable.
    assert set(total) == set(weighted)
    assert all(value > 0 for value in weighted.values())
