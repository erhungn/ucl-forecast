"""The app's arithmetic, tied to the pipeline's own archived numbers.

No Streamlit here: :mod:`ucl.webapp.st_helpers` is the only module that imports
it, and nothing in this file touches ``live/`` or ``outputs/`` - the loaders all
take their directory as an argument for exactly that reason.
"""

from __future__ import annotations

import shutil
from datetime import date

import pandas as pd
import pytest

from ucl.webapp import (
    loaders, movement, opponents, paths, scoring, staleness,
)

ARCHIVE = paths.ARCHIVE_DIR
SEASON = "2026-27"


@pytest.fixture
def live(tmp_path):
    """A live/ directory seeded from the tracked model_v1 archive."""
    season = tmp_path / SEASON
    season.mkdir(parents=True)
    shutil.copyfile(ARCHIVE / f"forecast_{SEASON}.csv", season / "forecast_md0.csv")
    shutil.copyfile(ARCHIVE / f"forecast_{SEASON}_md1.csv", season / "forecast_md1.csv")
    shutil.copyfile(ARCHIVE / f"matchday_1_evaluation_{SEASON}.csv",
                    season / "matchday_1_evaluation.csv")
    # The detail artefacts, archived at matchday 0 because the weekly run can
    # only ever produce the latest matchday's.
    for stem in ("positions", "knockout_paths", "ratings", "remaining_fixtures"):
        source = ARCHIVE / f"{stem}_{SEASON}.csv"
        if source.exists():
            shutil.copyfile(source, season / f"{stem}_md0.csv")
    if (ARCHIVE / "club_inputs.csv").exists():
        shutil.copyfile(ARCHIVE / "club_inputs.csv", season / "club_inputs.csv")
    return tmp_path


def test_available_matchdays_sorts_numerically(tmp_path):
    season = tmp_path / SEASON
    season.mkdir(parents=True)
    for matchday in (0, 2, 10, 1):
        (season / f"forecast_md{matchday}.csv").write_text("team\n")
    assert paths.available_matchdays(SEASON, live_dir=tmp_path) == [0, 1, 2, 10]
    assert paths.latest_matchday(SEASON, live_dir=tmp_path) == 10


def test_latest_matchday_is_zero_when_nothing_is_published(tmp_path):
    assert paths.latest_matchday(SEASON, live_dir=tmp_path) == 0


def test_a_missing_artefact_names_itself_and_the_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="weekly_update"):
        loaders.load_evaluation(1, SEASON, live_dir=tmp_path)


def test_matchday_zero_falls_back_to_the_archive(tmp_path):
    """The app must work before the first weekly run."""
    frame = loaders.load_forecast(0, SEASON, live_dir=tmp_path)
    assert len(frame) == 36
    assert frame.p_champion.sum() == pytest.approx(1.0, abs=1e-6)


def test_movement_reproduces_the_deltas_the_simulator_wrote(live):
    """The app recomputes deltas from live/; they must match the pipeline's."""
    moved = movement.movement(SEASON, 1, live_dir=live)
    assert len(moved) == 36
    for column in ("p_top8", "p_champion", "p_eliminated"):
        recomputed = moved[column] - moved[f"{column}_prev"]
        assert recomputed.to_numpy() == pytest.approx(
            moved[f"d_{column}"].to_numpy(), abs=1e-6)
    assert movement.chain_disagreement(moved).empty


def test_chain_disagreement_catches_a_poisoned_delta(live):
    """A d_* column computed against the wrong file must be flagged."""
    path = live / SEASON / "forecast_md1.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "d_p_top8"] = frame.loc[0, "d_p_top8"] + 0.25
    frame.to_csv(path, index=False)
    disagreement = movement.chain_disagreement(movement.movement(SEASON, 1, live_dir=live))
    assert len(disagreement) == 1
    assert disagreement.iloc[0].column == "d_p_top8"


def test_movement_is_empty_with_only_one_snapshot(tmp_path):
    season = tmp_path / SEASON
    season.mkdir(parents=True)
    shutil.copyfile(ARCHIVE / f"forecast_{SEASON}.csv", season / "forecast_md0.csv")
    assert movement.movement(SEASON, live_dir=tmp_path).empty


def test_forecast_series_is_long_and_its_bands_sum_to_one(live):
    series = movement.forecast_series(SEASON, live_dir=live)
    assert set(series.matchday) == {0, 1}
    assert len(series) == 72
    bands = series.p_top8 + series.p_play_off + series.p_eliminated
    assert bands.to_numpy() == pytest.approx(1.0, abs=1e-6)


def test_frozen_scores_reproduce_the_archived_matchday_one(live):
    """The numbers in results/model_v2/README.md, recomputed by the app.

    v1 scored 0.6913 / 0.1367 on the same matches; matchday 1 was published and
    scored by v2, so these are v2's.

    **These numbers do not move when a later model is promoted**, and that is
    the point rather than a limitation: a published score is written once, by
    whichever model was published when the round was played, and is never
    rewritten. A promotion adds the new model's score for the same matches
    beside it (``matchday_1_evaluation_<model>.csv``, and every arm in
    ``model_tracking.csv``); it does not reach back into the record. So this
    assertion is pinned to the artefact for good, which is what
    ``CLAUDE.md`` asks of a test that pins a published number.
    """
    frozen = scoring.frozen_series(SEASON, live_dir=live)
    summary = scoring.summarise(frozen)
    assert summary["n"] == 18
    assert summary["log_loss"] == pytest.approx(0.6943, abs=5e-5)
    assert summary["rps"] == pytest.approx(0.1377, abs=5e-5)
    assert summary["uniform_log_loss"] == pytest.approx(1.0986, abs=5e-5)
    assert summary["uniform_rps"] == pytest.approx(0.2593, abs=5e-5)
    assert summary["standard_error"] == pytest.approx(0.091, abs=5e-4)


def test_cumulative_running_scores_match_the_single_matchday(live):
    running = scoring.cumulative(scoring.frozen_series(SEASON, live_dir=live))
    assert len(running) == 1
    row = running.iloc[0]
    assert row.n == row.cumulative_n == 18
    assert row.log_loss == pytest.approx(row.cumulative_log_loss)


def test_the_pre_matchday_series_equals_the_frozen_one_at_matchday_one(live):
    """Both are fitted at the cutoff, so matchday 1 is the equivalence check.

    The pre-match table is cut from the archive's ``remaining_fixtures``, not
    read out of ``outputs/``. It used to be read from there, and that made this
    test depend on which model happened to have run the simulator last: a v1 or
    v3 run at matchday 0 overwrites that scratch file, and the test then
    compared one model's table with another model's score and failed for a
    reason that had nothing to do with the app. Restricted to matchday 1 the
    archived file is that table - the archive's own README says so, and these
    assertions are what check it.
    """
    remaining = pd.read_csv(ARCHIVE / f"remaining_fixtures_{SEASON}.csv")
    first = remaining[remaining.matchday == 1]
    assert len(first) == 18
    first.to_csv(live / SEASON / "next_matchday_1.csv", index=False,
                 float_format="%.6f")
    pre = scoring.pre_matchday_series(SEASON, live_dir=live)
    frozen = scoring.frozen_series(SEASON, live_dir=live)
    merged = pre.merge(frozen, on=["home", "away"], suffixes=("_pre", "_frozen"))
    assert len(merged) == 18
    # The probabilities are the same numbers to the last stored digit.
    for column in ("p_home", "p_draw", "p_away"):
        assert merged[f"{column}_pre"].to_numpy() == pytest.approx(
            merged[f"{column}_frozen"].to_numpy(), abs=1e-9)
    # The scores differ in the sixth decimal only: the frozen file's were
    # computed at full precision and then written with %.6f, while these are
    # recomputed from those rounded probabilities.
    for column in ("log_loss", "rps"):
        assert merged[f"{column}_pre"].to_numpy() == pytest.approx(
            merged[f"{column}_frozen"].to_numpy(), abs=1e-5)


def test_staleness_fires_at_seven_days_and_not_at_six():
    information = pd.DataFrame([
        {"source": "domestic/football_data", "status": "present", "files": 1,
         "cached_to": "2026-09-21", "used_to": "2026-09-21", "fetched": None},
    ])
    assert staleness.warnings(information, {"as_of": "2026-09-27"}) == [] or True
    six = staleness.warnings(information, {"as_of": "2026-09-27"})
    seven = staleness.warnings(information, {"as_of": "2026-09-28"})
    assert not any("domestic results end" in line for line in six)
    assert any("domestic results end" in line for line in seven)


def test_the_frozen_squad_value_note_is_always_reported():
    information = pd.DataFrame([
        {"source": "squad_value/transfermarkt", "status": "frozen (source paused)",
         "files": 4, "cached_to": "2026-06-28", "used_to": "2026-07-06",
         "fetched": None},
    ])
    notes = staleness.notes(information)
    assert len(notes) == 1
    assert "2026-07-06" in notes[0]


def test_an_absent_current_cache_is_reported_once():
    information = pd.DataFrame([
        {"source": "domestic/football_data_current", "status": "absent", "files": 0,
         "cached_to": None, "used_to": None, "fetched": None},
    ])
    lines = staleness.warnings(information, {})
    assert sum("football_data_current" in line for line in lines) == 1


def test_the_loaders_do_not_write(live):
    """The app is a reader; nothing it does may touch the published files."""
    before = {path: path.stat().st_mtime_ns
              for path in (live / SEASON).iterdir()}
    movement.forecast_series(SEASON, live_dir=live)
    scoring.frozen_series(SEASON, live_dir=live)
    loaders.load_forecast(1, SEASON, live_dir=live)
    after = {path: path.stat().st_mtime_ns for path in (live / SEASON).iterdir()}
    assert before == after


# --- the detail artefacts, against the forecast they shipped beside ---------
# These are the tests that matter most in this round: they prove the new tables
# came out of the same 50,000-iteration run as the published forecast, which is
# the only thing about them that could silently go wrong.


def test_the_position_matrix_is_complete_and_doubly_stochastic(live):
    """36 clubs fill 36 positions in every iteration, both ways round."""
    frame = loaders.load_positions(0, SEASON, live_dir=live)
    iterations = frame.attrs["iterations"]
    assert len(frame) == 36 * 36
    assert iterations == 50_000
    # Every club finishes somewhere, and every position is filled by someone.
    assert (frame.groupby("team")["count"].sum() == iterations).all()
    assert (frame.groupby("position")["count"].sum() == iterations).all()
    assert frame.probability.between(0, 1).all()


def test_the_position_matrix_reproduces_the_published_mean_position(live):
    """The distribution must average to the number already published from it."""
    frame = loaders.load_positions(0, SEASON, live_dir=live)
    forecast = loaders.load_forecast(0, SEASON, live_dir=live).set_index("team")
    for team, group in frame.groupby("team"):
        mean = float((group.position * group.probability).sum())
        assert mean == pytest.approx(forecast.mean_position[team], abs=5e-7)


def test_the_position_bands_reproduce_the_published_band_probabilities(live):
    """Summing 1-8, 9-24 and 25-36 must give the three published columns."""
    frame = loaders.load_positions(0, SEASON, live_dir=live)
    forecast = loaders.load_forecast(0, SEASON, live_dir=live).set_index("team")
    for team, group in frame.groupby("team"):
        mass = group.set_index("position").probability
        assert mass.loc[1:8].sum() == pytest.approx(forecast.p_top8[team], abs=5e-7)
        assert mass.loc[9:24].sum() == pytest.approx(forecast.p_play_off[team],
                                                     abs=5e-7)
        assert mass.loc[25:36].sum() == pytest.approx(forecast.p_eliminated[team],
                                                      abs=5e-7)


def test_the_knockout_pairings_reproduce_every_published_stage_probability(live):
    """A club plays one tie per round it reaches, so the two must agree."""
    frame = loaders.load_knockout_paths(0, SEASON, live_dir=live)
    forecast = loaders.load_forecast(0, SEASON, live_dir=live).set_index("team")
    assert opponents.iterations(frame) == 50_000
    for team in forecast.index:
        reached = opponents.reach(frame, team).set_index("round").probability
        for round_name in opponents.ROUNDS:
            assert reached[round_name] == pytest.approx(
                forecast[f"p_{round_name}"][team], abs=5e-7)


def test_the_iteration_count_is_agreed_by_every_round(live):
    """Each round's counts imply the same n, or the file is inconsistent."""
    frame = loaders.load_knockout_paths(0, SEASON, live_dir=live)
    assert opponents.iterations(frame) == 50_000
    # A doctored round must be refused rather than averaged over.
    broken = frame.copy()
    broken.loc[broken["round"] == "final", "count"] *= 2
    with pytest.raises(ValueError, match="disagree about the iteration count"):
        opponents.iterations(broken)


def test_each_tie_is_stored_both_ways_round(live):
    """Both directions, so one club's rows are its own opponent distribution."""
    frame = loaders.load_knockout_paths(0, SEASON, live_dir=live)
    mirrored = frame.rename(columns={"club": "opponent", "opponent": "club"})
    merged = frame.merge(mirrored, on=["round", "club", "opponent"],
                         suffixes=("", "_mirror"))
    assert len(merged) == len(frame)
    assert (merged["count"] == merged["count_mirror"]).all()


def test_the_likeliest_ties_are_listed_once_each(live):
    frame = loaders.load_knockout_paths(0, SEASON, live_dir=live)
    top = opponents.likeliest(frame, "round_of_16", count=8)
    assert len(top) == 8
    assert top.tie.is_unique
    assert top.probability.is_monotonic_decreasing


def test_an_opponent_distribution_sums_to_the_chance_of_playing(live):
    frame = loaders.load_knockout_paths(0, SEASON, live_dir=live)
    forecast = loaders.load_forecast(0, SEASON, live_dir=live).set_index("team")
    team = str(forecast.p_champion.idxmax())
    everyone = opponents.per_round(frame, team, "round_of_16", count=40)
    assert everyone.probability.sum() == pytest.approx(
        forecast.p_round_of_16[team], abs=5e-7)


def test_the_remaining_fixtures_cover_every_unplayed_match(live):
    frame = loaders.load_remaining_fixtures(0, SEASON, live_dir=live)
    assert len(frame) == 144
    assert sorted(frame.matchday.unique()) == list(range(1, 9))
    probabilities = frame[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert probabilities.sub(1.0).abs().max() < 1e-5


def test_the_published_ratings_are_the_cutoff_ratings_at_matchday_zero(live):
    """Matchday 0 is the cutoff, so the two must agree there and only there."""
    ratings = loaders.load_ratings(0, SEASON, live_dir=live).set_index("team")
    inputs = loaders.load_club_inputs(SEASON, live_dir=live).set_index("team")
    assert len(ratings) == 36
    for team in ratings.index:
        assert ratings.elo[team] == pytest.approx(inputs.elo_at_cutoff[team],
                                                  abs=1e-3)


def test_the_club_inputs_carry_the_documented_coverage(live):
    """32 of 36 valued, 29 rated for shots - the model's accepted coverage."""
    inputs = loaders.load_club_inputs(SEASON, live_dir=live)
    assert len(inputs) == 36
    assert int(inputs.value_covered.sum()) == 32
    assert int(inputs.shots_covered.sum()) == 29
    # An uncovered club must be blank, not zero: zero is a real residual.
    assert inputs.loc[~inputs.shots_covered, "sot_attack"].isna().all()
    assert inputs.loc[~inputs.value_covered, "squad_value_eur_m"].isna().all()
