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
