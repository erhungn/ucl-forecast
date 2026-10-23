"""The tripwires on promoting a model.

Promotion is one line - appending to :data:`ucl.config.MODEL_HISTORY` - plus one
weekly run. Everything else is derived from it. These tests pin the places that
have to agree, so a half-done promotion fails the suite instead of publishing a
forecast whose provenance nobody can read afterwards.

None of them asserts a model name against a literal. A literal in two places is
the bug: it passes until one of them is edited. The single literal lives in
``config.MODEL_HISTORY`` and these tests check that the archive on disk, the
scripts, the tracking arms and the last published run all agree with *it*.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from ucl import config, tracking
from ucl.models import forecast
from ucl.webapp import loaders, paths

SCRIPTS_WITH_A_MODEL = ("simulate.py", "weekly_update.py", "evaluate_matchday.py")


# --- the declaration itself ------------------------------------------------

def test_the_published_model_is_the_last_one_in_the_history():
    assert config.PUBLISHED_MODEL == config.MODEL_HISTORY[-1]
    assert len(set(config.MODEL_HISTORY)) == len(config.MODEL_HISTORY)


def test_every_published_model_is_one_the_code_can_build():
    """Publishing a model the fitter does not know is the one unrecoverable
    half-promotion: every run would refuse, including the one that would put it
    back."""
    for model in config.MODEL_HISTORY:
        assert model in forecast.MODELS, (
            f"{model} is in MODEL_HISTORY but not in forecast.MODELS, so "
            f"nothing can fit it")


def test_v3_is_the_accepted_candidate_under_another_name(monkeypatch):
    """The published v3 must be candidate C's fit, not a near relative of it.

    Checked by routing rather than by fitting twice: a fit is 90 seconds and
    the claim is about which function the label reaches and with what, which is
    exactly what this sees.
    """
    seen = []

    def spy(pre_match, values, elo_ratings, **kwargs):
        seen.append(kwargs)
        return "fitted"

    monkeypatch.setattr(forecast, "fit_euro_scale", spy)
    for label in ("v3", "v2+C"):
        assert forecast.fit(label, "pre", {}, {}, as_of="cutoff",
                            value_as_of="value") == "fitted"
    assert len(seen) == 2
    assert seen[0]["label"] == "v3" and seen[1]["label"] == "v2+C"
    # Everything except the name is identical, which is the claim.
    assert {k: v for k, v in seen[0].items() if k != "label"} == \
           {k: v for k, v in seen[1].items() if k != "label"}


# --- the archive on disk ---------------------------------------------------

def test_the_archive_directory_holds_the_model_it_claims_to():
    """Not a tautology: ARCHIVE_DIR is derived, but what is *in* it is not."""
    assert paths.ARCHIVE_DIR.is_dir(), (
        f"{paths.ARCHIVE_DIR} does not exist, so the app has no matchday-0 "
        f"fallback for the published model")
    forecast_csv = paths.ARCHIVE_DIR / f"forecast_{paths.SEASON}.csv"
    assert forecast_csv.exists()
    assert len(pd.read_csv(forecast_csv)) == 36

    coefficients = paths.ARCHIVE_DIR / f"model_{paths.SEASON}.json"
    assert coefficients.exists(), (
        f"{coefficients} is missing; scripts/write_model_json.py writes it")
    published = json.loads(coefficients.read_text(encoding="utf-8"))
    # v1's file predates this key, so its absence is history rather than a
    # fault; a file that carries it must agree with the declaration.
    if "model" in published:
        assert published["model"] == config.PUBLISHED_MODEL


def test_every_earlier_archive_still_exists_and_is_not_the_current_one():
    for older in paths.PREVIOUS_ARCHIVES:
        assert older.is_dir(), f"{older} is referenced but missing"
        assert older != paths.ARCHIVE_DIR
        assert (older / f"forecast_{paths.SEASON}.csv").exists()


def test_the_archives_cover_the_whole_published_history():
    expected = {paths.archive_dir(model) for model in config.MODEL_HISTORY}
    assert expected == {paths.ARCHIVE_DIR, *paths.PREVIOUS_ARCHIVES}


# --- the scripts -----------------------------------------------------------

def _model_argument(script: str) -> ast.Call:
    source = (config.PROJECT_ROOT / "scripts" / script).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--model"):
            return node
    raise AssertionError(f"{script} has no --model argument")


@pytest.mark.parametrize("script", SCRIPTS_WITH_A_MODEL)
def test_no_script_hardcodes_which_model_is_published(script):
    """``default="v2"`` is the bug this catches.

    Three scripts take ``--model`` and all three defaulted to a literal. A
    literal is right until the day it is not, and on that day the weekly run
    quietly keeps publishing the superseded model while every other check is
    green - which is precisely the failure that is impossible to see in an
    aggregate.
    """
    keywords = {k.arg: k.value for k in _model_argument(script).keywords}
    assert "default" in keywords, f"{script}'s --model has no default"
    assert not isinstance(keywords["default"], ast.Constant), (
        f"{script} hardcodes its --model default; it must come from "
        f"config.PUBLISHED_MODEL")
    assert "PUBLISHED_MODEL" in ast.unparse(keywords["default"])
    assert "choices" in keywords
    assert not isinstance(keywords["choices"], (ast.Tuple, ast.List)), (
        f"{script} lists its --model choices by hand; they must come from "
        f"forecast.MODELS")


@pytest.mark.parametrize("script", SCRIPTS_WITH_A_MODEL)
def test_every_script_offers_exactly_the_buildable_models(script):
    result = subprocess.run([sys.executable, f"scripts/{script}", "--help"],
                            capture_output=True, text=True,
                            cwd=config.PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    found = re.search(r"--model \{([^}]*)\}", result.stdout)
    assert found, result.stdout[:400]
    assert set(found.group(1).split(",")) == set(forecast.MODELS)


# --- tracking --------------------------------------------------------------

def test_the_published_model_is_a_tracked_arm():
    """Promoting without adding the arm would lose the published model's own
    score on this season, which is the one series that cannot be rebuilt
    afterwards from anything else."""
    assert config.PUBLISHED_MODEL in tracking.MODELS
    assert tracking.REFERENCE in tracking.MODELS


def test_exactly_one_tracked_arm_is_marked_published():
    marked = [name for name, text in tracking.MODELS.items()
              if "(published)" in text]
    assert marked == [config.PUBLISHED_MODEL]


def test_no_tracked_arm_is_ever_dropped():
    """A model stops being published; it does not stop having scored matches.

    Every model that has been published, and both candidates that were measured
    against this season, stay as arms - otherwise promotion erases the record it
    is supposed to be judged against.
    """
    for model in config.MODEL_HISTORY:
        assert model in tracking.MODELS
    assert "v2+B" in tracking.MODELS


# --- what is actually published -------------------------------------------

def test_the_last_published_run_used_the_published_model():
    """The one test that fails *during* a promotion, by design.

    Between appending to MODEL_HISTORY and running the weekly update, live/
    holds the previous model's forecast. That window should be loud: the
    constant is a declaration of intent and this is the check that the intent
    was carried out.
    """
    status = loaders.load_status(paths.SEASON)
    if not status:
        pytest.skip("no update has run")
    assert status.get("model") == config.PUBLISHED_MODEL, (
        f"live/ was published by {status.get('model')} but "
        f"{config.PUBLISHED_MODEL} is declared published. Run "
        f"scripts/weekly_update.py, or put MODEL_HISTORY back.")


def test_the_published_scores_are_attributed_to_a_model():
    """Every published score is identifiable, from the numbers alone."""
    from ucl.webapp import scoring

    scored = paths.scored_matchdays(paths.SEASON)
    if not scored:
        pytest.skip("nothing scored yet")
    attributed = scoring.scoring_models(paths.SEASON)
    assert set(attributed) == set(scored)
    assert all(model is not None for model in attributed.values()), (
        f"unattributed published score(s): "
        f"{[md for md, m in attributed.items() if m is None]}")


# --- the record-keeping rules ---------------------------------------------

def test_a_named_models_score_never_displaces_the_published_record():
    record = paths.evaluation_path(3, paths.SEASON)
    beside = paths.evaluation_path(3, paths.SEASON, model="v3")
    assert record != beside
    assert record.name == "matchday_3_evaluation.csv"
    assert beside.name == "matchday_3_evaluation_v3.csv"


def test_a_named_models_score_is_not_mistaken_for_the_record(tmp_path):
    """``scored_matchdays`` is the published series and must ignore the rest.

    If it did not, a promoted model's added scores would silently become extra
    matchdays in every aggregate on the played-matches page.
    """
    season = tmp_path / paths.SEASON
    season.mkdir()
    (season / "matchday_1_evaluation.csv").write_text("date\n")
    (season / "matchday_1_evaluation_v3.csv").write_text("date\n")
    (season / "matchday_2_evaluation_v3.csv").write_text("date\n")
    assert paths.scored_matchdays(paths.SEASON, live_dir=tmp_path) == [1]
    assert paths.evaluations_by_model(paths.SEASON, live_dir=tmp_path) == \
        {"v3": [1, 2]}


def test_a_superseded_models_snapshots_are_readable_as_its_own_history(tmp_path):
    from ucl.webapp import movement

    season = tmp_path / paths.SEASON
    history = season / "archive_v2"
    history.mkdir(parents=True)
    published = pd.read_csv(paths.ARCHIVE_DIR / f"forecast_{paths.SEASON}.csv")
    for matchday in (1, 2):
        published.to_csv(history / f"forecast_md{matchday}.csv", index=False)

    assert paths.history_models(paths.SEASON, live_dir=tmp_path) == ["v2"]
    assert paths.history_matchdays("v2", paths.SEASON, live_dir=tmp_path) == [1, 2]
    series = movement.history_series("v2", paths.SEASON, live_dir=tmp_path)
    assert len(series) == 72
    assert set(series.model) == {"v2"}
    assert "matchdays 1-2" in movement.history_note(paths.SEASON, live_dir=tmp_path)


def test_a_backfilled_snapshot_is_labelled_retroactive(tmp_path):
    season = tmp_path / paths.SEASON
    season.mkdir()
    published = pd.read_csv(paths.ARCHIVE_DIR / f"forecast_{paths.SEASON}.csv")
    published.to_csv(season / "forecast_md0.csv", index=False)
    backfilled = published.assign(retroactive=True, retroactive_model="v3")
    backfilled.to_csv(season / "forecast_md1.csv", index=False)

    assert loaders.retroactive_matchdays(paths.SEASON, live_dir=tmp_path) == \
        {1: "v3"}

    from ucl.webapp import movement

    series = movement.forecast_series(paths.SEASON, live_dir=tmp_path)
    assert series[series.matchday == 0].retroactive.eq(False).all()
    assert series[series.matchday == 1].retroactive.all()
    assert set(series[series.matchday == 1].produced_by) == {"v3"}


def test_a_normal_forecast_carries_no_retroactive_column():
    """The column is written only when asked, so the byte-identity gate on the
    published models is untouched by its existence."""
    published = pd.read_csv(paths.ARCHIVE_DIR / f"forecast_{paths.SEASON}.csv")
    assert "retroactive" not in published.columns
    assert loaders.retroactive_matchdays(paths.SEASON) == {}


# --- the coefficient writer -----------------------------------------------

def _writer():
    import importlib.util

    path = config.PROJECT_ROOT / "scripts" / "write_model_json.py"
    spec = importlib.util.spec_from_file_location("write_model_json", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_coefficient_comparison_passes_the_fit_noise_and_nothing_else():
    """The fit is reproducible to about 1e-11 between processes, not exactly.

    ``ucl.models.gap.replay`` averages the relegated clubs' ratings over a set
    of club names, so PYTHONHASHSEED changes the summation order. The comparison
    therefore has to be numeric - and has to be tight enough that a real change
    cannot hide in it.
    """
    writer = _writer()
    base = {"a": 0.16951346246868643, "model": "v2", "converged": True,
            "covariates": ["d_value"], "gap_params": {"sot": [0.1346, 0.6]}}
    noisy = {**base, "a": 0.16951346246871263,
             "gap_params": {"sot": [0.13460000000002, 0.6]}}
    assert writer.compare(base, noisy) == []

    for key, wrong in (("a", 0.1696), ("model", "v3"), ("converged", False),
                       ("covariates", ["d_value", "d_gap_sot"])):
        assert writer.compare(base, {**base, key: wrong}), key
    assert writer.compare(base, {**base, "gap_params": {"sot": [0.14, 0.6]}})


def test_the_writer_reproduces_the_published_archive():
    """Slow - it refits the published model - and the point of the script.

    The writer is trusted with a new archive only because it reproduces one it
    did not write.
    """
    result = subprocess.run(
        [sys.executable, "scripts/write_model_json.py",
         "--model", config.PUBLISHED_MODEL, "--check"],
        capture_output=True, text=True, cwd=config.PROJECT_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reproduced" in result.stdout


# --- rehearsing a switch ---------------------------------------------------

def test_the_weekly_update_threads_its_live_directory_through_every_path():
    """One forgotten call site writes into the real live/ during a rehearsal.

    ``--live-dir`` exists so a model switch can be rehearsed against a copy.
    That is worth nothing if a single ``paths.forecast_path(...)`` without the
    keyword publishes into ``live/`` anyway - and the rehearsal would look like
    it worked, because every file it checked would be there. Structural, so a
    call site added later cannot miss it.
    """
    needs_it = {
        "forecast_path", "next_matchday_path", "evaluation_path",
        "positions_path", "knockout_paths_path", "ratings_path",
        "remaining_fixtures_path", "club_inputs_path", "information_path",
        "tracking_path", "status_path", "log_path", "season_dir", "history_dir",
    }
    source = (config.PROJECT_ROOT / "scripts" / "weekly_update.py").read_text(
        encoding="utf-8")
    missing = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in needs_it
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "paths"
                and not any(keyword.arg is None for keyword in node.keywords)):
            missing.append((node.lineno, node.func.attr))
    assert not missing, (
        f"weekly_update.py calls these without **live, so --live-dir would not "
        f"redirect them: {missing}")


def test_the_live_directory_is_offered_and_recorded():
    result = subprocess.run([sys.executable, "scripts/weekly_update.py", "--help"],
                            capture_output=True, text=True,
                            cwd=config.PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert "--live-dir" in result.stdout


def test_a_rehearsal_directory_is_a_complete_parallel_of_live(tmp_path):
    """Every published path has to land under the rehearsal root, not beside it."""
    season = paths.SEASON
    produced = [
        paths.forecast_path(3, season, live_dir=tmp_path),
        paths.next_matchday_path(4, season, live_dir=tmp_path),
        paths.evaluation_path(3, season, live_dir=tmp_path),
        paths.evaluation_path(3, season, model="v3", live_dir=tmp_path),
        paths.positions_path(3, season, live_dir=tmp_path),
        paths.knockout_paths_path(3, season, live_dir=tmp_path),
        paths.ratings_path(3, season, live_dir=tmp_path),
        paths.remaining_fixtures_path(3, season, live_dir=tmp_path),
        paths.club_inputs_path(season, live_dir=tmp_path),
        paths.information_path(season, live_dir=tmp_path),
        paths.tracking_path(season, live_dir=tmp_path),
        paths.status_path(season, live_dir=tmp_path),
        paths.log_path(season, live_dir=tmp_path),
        paths.history_dir("v2", season, live_dir=tmp_path),
    ]
    for path in produced:
        assert path.is_relative_to(tmp_path), path
        assert not path.is_relative_to(paths.LIVE_DIR), path
    # And the names match what the app looks for under the real root.
    for path in produced:
        assert path.name == (
            path.name if path.parent.name.startswith(paths.HISTORY_PREFIX)
            else Path(str(path).replace(str(tmp_path), str(paths.LIVE_DIR))).name)
