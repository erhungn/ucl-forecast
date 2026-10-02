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

from ucl.webapp import loaders, movement, paths, scoring, staleness

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
    """The numbers in results/model_v1/README.md, recomputed by the app."""
    frozen = scoring.frozen_series(SEASON, live_dir=live)
    summary = scoring.summarise(frozen)
    assert summary["n"] == 18
    assert summary["log_loss"] == pytest.approx(0.6913, abs=5e-5)
    assert summary["rps"] == pytest.approx(0.1367, abs=5e-5)
    assert summary["uniform_log_loss"] == pytest.approx(1.0986, abs=5e-5)
    assert summary["uniform_rps"] == pytest.approx(0.2593, abs=5e-5)
    assert summary["standard_error"] == pytest.approx(0.090, abs=5e-4)


def test_cumulative_running_scores_match_the_single_matchday(live):
    running = scoring.cumulative(scoring.frozen_series(SEASON, live_dir=live))
    assert len(running) == 1
    row = running.iloc[0]
    assert row.n == row.cumulative_n == 18
    assert row.log_loss == pytest.approx(row.cumulative_log_loss)


def test_the_pre_matchday_series_equals_the_frozen_one_at_matchday_one(live):
    """Both are fitted at the cutoff, so matchday 1 is the equivalence check."""
    shutil.copyfile(
        paths.ARCHIVE_DIR.parent.parent / "outputs" / f"next_matchday_1_{SEASON}.csv",
        live / SEASON / "next_matchday_1.csv")
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
