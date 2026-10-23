"""The weekly update's refusals and bookkeeping, without running a forecast.

The logic lives in :mod:`ucl.webapp.update` precisely so it can be tested like
this: synthetic frames and ``tmp_path`` caches, no five-minute simulation and no
writes to the real ``data/raw/``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date

import pandas as pd
import pytest

from ucl.data.cutoff import as_of_for_matchday
from ucl.data.manifest import ManifestEntry, read_manifest, write_manifest
from ucl.webapp import update


def fixtures_frame(matchdays=(1, 2, 3), per_matchday=18) -> pd.DataFrame:
    rows = []
    for matchday in matchdays:
        for index in range(per_matchday):
            rows.append({
                "matchday": matchday,
                "date": pd.Timestamp("2026-09-08") + pd.Timedelta(30 * matchday, "D"),
                "home": f"H{matchday}-{index}", "away": f"A{matchday}-{index}",
            })
    return pd.DataFrame(rows)


def played_frame(counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    for matchday, count in counts.items():
        for index in range(count):
            rows.append({
                "matchday": matchday,
                "date": pd.Timestamp("2026-09-08") + pd.Timedelta(30 * matchday, "D"),
                "home_id": f"H{matchday}-{index}", "away_id": f"A{matchday}-{index}",
                "hg": 1, "ag": 0, "status": "played",
            })
    return pd.DataFrame(rows)


def test_the_prefix_stops_at_the_first_incomplete_matchday():
    fixtures = fixtures_frame()
    assert update.latest_complete_prefix(played_frame({1: 18, 2: 18}), fixtures) == 2
    # Matchday 2 postponed, 3 finished: the maximum completed is 3, but the
    # forecast can only run through 1.
    assert update.latest_complete_prefix(played_frame({1: 18, 3: 18}), fixtures) == 1
    assert update.latest_complete_prefix(played_frame({2: 18}), fixtures) == 0


def test_a_skipped_round_is_detected_separately():
    fixtures = fixtures_frame()
    played = played_frame({1: 18, 3: 18})
    matchday = update.latest_complete_prefix(played, fixtures)
    assert update.missing_before(played, fixtures, matchday) == [3]


def test_partial_matchdays_report_both_counts():
    fixtures = fixtures_frame()
    partial = update.partial_matchdays(played_frame({1: 18, 2: 9}), fixtures)
    assert partial == {2: (9, 18)}
    # A matchday with no results at all is not partial, it is simply unplayed.
    assert update.partial_matchdays(played_frame({1: 18}), fixtures) == {}


def _cache(directory, names, *, payload="x"):
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in names:
        path = directory / name
        path.write_text(payload)
        entries.append(ManifestEntry(
            file=name, source_url=f"https://example.invalid/{name}",
            downloaded_at_utc="2026-10-02T00:00:00+00:00",
            sha256="0" * 64, bytes=len(payload)))
    write_manifest(directory, entries)
    return directory


def test_a_collision_names_the_superseded_copy(tmp_path):
    current = _cache(tmp_path / "football_data_current", ["E0_2627.csv"])
    main = _cache(tmp_path / "football_data", ["E0_2627.csv", "E0_2526.csv"])
    found = update.collisions((current, main))
    assert len(found) == 1
    # The weekly refresh is the newer copy, so the one in the main cache loses.
    assert found[0].superseded == main / "E0_2627.csv"
    assert found[0].winner == current / "E0_2627.csv"


def test_a_dry_run_prune_touches_nothing(tmp_path):
    current = _cache(tmp_path / "football_data_current", ["E0_2627.csv"])
    main = _cache(tmp_path / "football_data", ["E0_2627.csv"])
    found = update.collisions((current, main))
    update.prune_superseded(found, dry_run=True)
    assert (main / "E0_2627.csv").exists()
    assert "E0_2627.csv" in read_manifest(main)


def test_pruning_removes_the_file_and_its_manifest_entry(tmp_path):
    current = _cache(tmp_path / "football_data_current", ["E0_2627.csv"])
    main = _cache(tmp_path / "football_data", ["E0_2627.csv", "E0_2526.csv"])
    removed = update.prune_superseded(update.collisions((current, main)))
    assert removed == [main / "E0_2627.csv"]
    assert not (main / "E0_2627.csv").exists()
    manifest = read_manifest(main)
    assert "E0_2627.csv" not in manifest and "E0_2526.csv" in manifest
    # The winning copy is untouched.
    assert (current / "E0_2627.csv").exists()


def test_the_inputs_hash_moves_only_when_an_input_does(tmp_path):
    cache = (_cache(tmp_path / "football_data", ["E0_2627.csv"]),)
    played = played_frame({1: 18})
    first = update.inputs_hash(played, cache)
    assert first == update.inputs_hash(played.copy(), cache)

    corrected = played.copy()
    corrected.loc[0, "hg"] = 3
    assert update.inputs_hash(corrected, cache) != first


def test_the_as_of_date_matches_the_archived_forecast():
    """results/model_v1/README.md records 2026-10-12 for matchday 1."""
    from ucl.config import SEASONS
    from ucl.data import fixtures as fx

    cfg = SEASONS["2026-27"]
    fixtures = pd.read_csv(fx.fixtures_path(cfg.label), parse_dates=["date"])
    assert as_of_for_matchday(fixtures, 0, cfg.cutoff_date) == cfg.cutoff_date
    assert as_of_for_matchday(fixtures, 1, cfg.cutoff_date) == date(2026, 10, 12)


def test_publishing_is_atomic_and_leaves_no_temporary(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("team\nArsenal\n")
    destination = tmp_path / "live" / "2026-27" / "forecast_md1.csv"
    update.publish(source, destination)
    assert destination.read_text() == "team\nArsenal\n"
    assert not list(destination.parent.glob("*.tmp"))


def test_status_round_trips(tmp_path):
    path = tmp_path / "status.json"
    assert update.read_status(path) == {}
    update.write_status(path, {"state": "ok", "through_matchday": 2})
    assert update.read_status(path)["through_matchday"] == 2
    assert json.loads(path.read_text())["state"] == "ok"


def test_the_next_matchday_table_matches_the_frozen_evaluation():
    """Both are the model fitted at the cutoff, so they must agree exactly.

    This is the check on the table scripts/simulate.py now persists: if the
    writer ever drifts from the probabilities the evaluation computes
    independently, one of the two is wrong.
    """
    from ucl.config import OUTPUTS_DIR
    from ucl.webapp import paths

    saved = OUTPUTS_DIR / "next_matchday_1_2026-27.csv"
    if not saved.exists():
        pytest.skip("run scripts/simulate.py once to produce the table")
    evaluation = pd.read_csv(
        paths.ARCHIVE_DIR / "matchday_1_evaluation_2026-27.csv")
    merged = pd.read_csv(saved).merge(
        evaluation, on=["home", "away"], suffixes=("_next", "_eval"))
    assert len(merged) == 18
    for column in ("p_home", "p_draw", "p_away"):
        assert merged[f"{column}_next"].to_numpy() == pytest.approx(
            merged[f"{column}_eval"].to_numpy(), abs=1e-9)


def test_the_script_runs():
    result = subprocess.run(
        [sys.executable, "scripts/weekly_update.py", "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "--prune-stale" in result.stdout


def test_the_matchday_zero_archive_points_at_the_current_model():
    """The fallback snapshot must be the published model's, not a superseded one.

    It anchors every later matchday's d_p_* columns, so pointing it at an old
    archive computes a one-matchday change across two different models - which
    is not a change in anything. This caught exactly that on the v1-to-v2
    switch.

    ``ARCHIVE_DIR`` is now derived from ``config.PUBLISHED_MODEL``, so asserting
    its name against a literal would assert a constant against itself. What is
    still worth checking is the step the derivation cannot make: that the
    archive on disk holds the model the declaration claims, and that the
    earlier ones are still there to compare with. ``tests/test_promotion.py``
    pins the rest of the chain.
    """
    import json

    from ucl import config
    from ucl.webapp import paths

    published = json.load(
        open(paths.ARCHIVE_DIR / f"model_{paths.SEASON}.json"))
    assert published["model"] == config.PUBLISHED_MODEL
    for older in paths.PREVIOUS_ARCHIVES:
        assert older.exists() and older != paths.ARCHIVE_DIR


def test_every_published_snapshot_came_from_the_published_model():
    """live/ must not mix models across a delta."""
    import json

    from ucl.webapp import loaders, paths

    status = loaders.load_status(paths.SEASON)
    if not status:
        pytest.skip("no update has run")
    archived = pd.read_csv(paths.ARCHIVE_DIR / f"forecast_{paths.SEASON}.csv")
    live = pd.read_csv(paths.forecast_path(0, paths.SEASON))
    merged = archived[["team", "p_champion", "mean_position"]].merge(
        live[["team", "p_champion", "mean_position"]], on="team",
        suffixes=("_archive", "_live"))
    assert len(merged) == 36
    for column in ("p_champion", "mean_position"):
        assert (merged[f"{column}_archive"] - merged[f"{column}_live"]
                ).abs().max() == pytest.approx(0.0, abs=1e-9)


# --- the two guards the 2026-10-20 runs needed ------------------------------

def test_the_sweep_covers_every_cache_directory_a_loader_defines():
    """A new source must be swept for near-miss names without being added here.

    Any module-level path under ``data/raw/`` with no file extension is a cache
    directory, and every one of them has to be on the list the name check
    sweeps - the whole failure was a directory nothing was looking for.
    """
    import importlib
    import pkgutil
    from pathlib import Path

    import ucl.data
    from ucl.config import DATA_RAW

    swept = set(update.cache_directories())
    defined = {}
    for found in pkgutil.iter_modules(ucl.data.__path__):
        module = importlib.import_module(f"ucl.data.{found.name}")
        for name, value in vars(module).items():
            if (isinstance(value, Path) and value.parent == DATA_RAW
                    and not value.suffix and not name.startswith("_")):
                defined[value] = f"ucl.data.{found.name}.{name}"
    assert defined, "no cache directories found, so this test proves nothing"
    missing = {path: where for path, where in defined.items() if path not in swept}
    assert not missing, f"not swept for near-miss names: {missing}"


def test_the_sweep_names_a_near_miss_and_passes_on_the_real_caches(tmp_path):
    from ucl.data.manifest import CacheNameError

    (tmp_path / "football_data_current ").mkdir()
    with pytest.raises(CacheNameError, match="football_data_current "):
        update.check_cache_names((tmp_path / "football_data_current",))
    # And the repository's own data/raw, swept exactly as a run sweeps it.
    update.check_cache_names()


def test_the_script_offers_the_stale_override():
    """The refusal has a documented way past it, or a legitimate stale run
    (an off-season re-forecast, say) could not be published at all."""
    result = subprocess.run(
        [sys.executable, "scripts/weekly_update.py", "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0
    # argparse wraps the help text at whatever width it likes, so match on it
    # with the whitespace taken out.
    flat = "".join(result.stdout.split())
    assert "--allow-stale" in flat
    assert "nocurrent-seasonresults" in flat and "10daysbeforetheas-ofdate" in flat


def test_the_published_status_records_how_the_guards_went():
    """A run published with a known-thin cache must be identifiable afterwards."""
    from ucl.webapp import loaders, paths

    status = loaders.load_status(paths.SEASON)
    if not status or status.get("state") != "ok":
        pytest.skip("no successful update has run")
    if "domestic_gaps" not in status:
        pytest.skip("published before the domestic-coverage gate existed")
    assert status["domestic_gaps"] == [] or status["allow_stale"]
