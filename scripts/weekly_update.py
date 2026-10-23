"""Refresh the results, re-forecast, score what was played, publish it all.

One command for the weekly cycle. It refuses rather than guesses, and the order
is deliberate: everything that can refuse does so before the five-minute
simulation, not after it.

    1. cache names           - a folder a loader would read as absent
    2. cache collisions      - a file in two caches, which the loader refuses
    3. cache integrity       - every manifest hash re-checked
    4. seed live/            - matchday 0 from the frozen archive
    5. refresh Wikipedia     - exactly once, here and nowhere else
    6. detect the matchday   - the contiguous complete prefix
    7. domestic coverage     - every main league, close enough to the as-of date
    8. idempotence gate      - unchanged inputs do no work
    9. restore the anchor    - the delta chain reads live/, not scratch
   10. forecast              - scripts/simulate.py
   11. score                 - scripts/evaluate_matchday.py per played matchday
   12. information set       - what each source knew
   13. model tracking        - every candidate scored on the played matches
   14. publish               - atomically, into live/<season>/
   15. status.json           - provenance for the run

Artefacts go to ``live/<season>/``, which only this script writes. ``outputs/``
stays scratch: the test suite runs the simulator into it with a handful of
iterations, so a forecast there may be a smoke test.

**The published record is never overwritten.** Two rules follow from that, and
both are exercised by a model switch:

* A matchday's score - ``matchday_k_evaluation.csv`` - is written once, by
  whichever model was published when that matchday was played, and after that
  it is left alone. A later model's score for the same matches is written
  beside it as ``matchday_k_evaluation_<model>.csv``. Whether the published
  file came from this model is decided by **comparing the numbers**, not by
  trusting a provenance field: identical means nothing happened, different
  means another model wrote it.
* On a model switch the previous model's per-matchday snapshots are **moved**
  into ``live/<season>/archive_<model>/``, not deleted. They are what was
  published at the time; the app reads them back as history, labelled with the
  model that produced them.

``--live-dir`` runs the whole cycle against a copy of ``live/`` instead, which
is how a model switch is rehearsed before it is done for real.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ucl import config
from ucl.config import OUTPUTS_DIR, season as season_config
from ucl.data import fixtures as fx
from ucl.data import football_data, league_results
from ucl.data.cutoff import as_of_for_matchday
from ucl.data.manifest import CacheIntegrityError, CacheNameError, verify_cache
from ucl.models.forecast import MODELS as forecast_models
from ucl.webapp import paths, update

#: The forecast is always produced by the pinned environment that
#: requirements.lock.txt documents, whichever interpreter serves the app.
MODEL_PYTHON = config.PROJECT_ROOT / ".venv" / "bin" / "python"
SCRIPTS = config.PROJECT_ROOT / "scripts"


class Log:
    """Print to the terminal and to ``live/<season>/update.log`` at once.

    The script owns the log, not the app: a browser reload or a dropped phone
    connection must not be able to lose the record of a run.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            os.replace(path, path.with_suffix(".log.1"))
        self.handle = open(path, "w", encoding="utf-8", buffering=1)

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self) -> None:
        self.handle.close()


def interpreter(requested: str | None) -> str:
    if requested:
        return requested
    return str(MODEL_PYTHON) if MODEL_PYTHON.exists() else sys.executable


def run_step(log: Log, argv: list[str]) -> int:
    """Run a pipeline script, streaming its output into the log."""
    log(f"  $ {' '.join(argv)}")
    process = subprocess.Popen(
        argv, cwd=config.PROJECT_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        log("  | " + line.rstrip("\n"))
    return process.wait()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT,
            capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=paths.SEASON)
    parser.add_argument("-n", "--iterations", type=int, default=config.N_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--model", choices=forecast_models,
                        default=config.PUBLISHED_MODEL,
                        help="which model to publish; the default is whichever "
                             "model is published (config.PUBLISHED_MODEL)")
    parser.add_argument("--live-dir", default=None,
                        help="publish into this directory instead of live/. "
                             "For rehearsing a run - a model switch above all - "
                             "against a copy, without touching what the app "
                             "serves.")
    parser.add_argument("--prune-stale", action="store_true",
                        help="delete cache files superseded by a newer copy")
    parser.add_argument("--no-refresh", action="store_true",
                        help="use the cached Wikipedia page instead of fetching")
    parser.add_argument("--allow-partial", action="store_true",
                        help="forecast through the last complete matchday even "
                             "while the next one is half played")
    parser.add_argument("--allow-stale", action="store_true",
                        help="publish even though a main league has no "
                             "current-season results, or the domestic cache "
                             "ends more than "
                             f"{league_results.DOMESTIC_PUBLISH_DAYS} days "
                             "before the as-of date")
    parser.add_argument("--force", action="store_true",
                        help="re-run even when the inputs are unchanged")
    parser.add_argument("--rescore", action="store_true",
                        help="score every played matchday again")
    parser.add_argument("--python", default=None,
                        help="interpreter for the pipeline steps "
                             "(default: .venv/bin/python)")
    args = parser.parse_args(argv)
    cfg = season_config(args.season)
    live = {"live_dir": Path(args.live_dir)} if args.live_dir else {}

    # Read the last run's record before this run's status overwrites it; the
    # idempotence gate in step 6 compares against it.
    previous = update.read_status(paths.status_path(cfg.label, **live))
    log = Log(paths.log_path(cfg.label, **live))
    started = time.perf_counter()
    status = {
        "season": cfg.label, "state": "running", "message": "",
        "pid": os.getpid(), "model": args.model, "iterations": args.iterations,
        "seed": args.seed, "git_commit": git_commit(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if live:
        status["live_dir"] = str(live["live_dir"])
        status["rehearsal"] = True
    update.write_status(paths.status_path(cfg.label, **live), status)

    def finish(state: str, message: str = "", **extra) -> None:
        status.update(
            state=state, message=message,
            finished_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_seconds=round(time.perf_counter() - started, 1), **extra)
        update.write_status(paths.status_path(cfg.label, **live), status)

    try:
        return _run(args, cfg, log, status, finish, previous, live)
    except SystemExit as refusal:
        log(f"\nREFUSED: {refusal}")
        finish("refused", str(refusal))
        raise
    except Exception as error:                      # noqa: BLE001 - recorded
        log(f"\nFAILED: {type(error).__name__}: {error}")
        finish("failed", f"{type(error).__name__}: {error}")
        raise
    finally:
        log.close()


def _run(args, cfg, log: Log, status: dict, finish, previous: dict,
         live: dict) -> int:
    python = interpreter(args.python)
    log(f"=== weekly update: {cfg.label}, model {args.model}, "
        f"{args.iterations} iterations, seed {args.seed} ===")
    log(f"  interpreter {python}")
    season_directory = paths.season_dir(cfg.label, **live)
    if live:
        log(f"  REHEARSAL: publishing into {season_directory}, not live/. "
            f"Nothing the app serves is touched.")
    log(f"  publishing into {season_directory}")
    if args.model not in config.MODEL_HISTORY:
        # A model that is not declared published must not reach the real live/.
        # The app, the matchday-0 anchor and the tracking reference all read
        # config.MODEL_HISTORY, so publishing past it leaves live/ holding a
        # forecast the app does not consider published - visible only in
        # status.json, which is the one place nobody looks first. Promotion is
        # the declaration and then the run, in that order.
        if not live:
            raise SystemExit(
                f"--model {args.model} is not in config.MODEL_HISTORY "
                f"{config.MODEL_HISTORY}, so it is not a published model and "
                f"must not be written into live/. Rehearse it with --live-dir "
                f"<a copy of live/> first; to promote it, append "
                f"{args.model!r} to MODEL_HISTORY and run this again."
            )
        log(f"  {args.model} is not yet a published model "
            f"(config.MODEL_HISTORY = {config.MODEL_HISTORY}); this is a "
            f"rehearsal of promoting it.")

    # --- 1. cache names, before anything reads a cache ----------------------
    # A folder whose name is a near miss reads as absent, and absent is a
    # legitimate state for football_data_current, so nothing downstream can
    # tell the two apart. This can.
    log("\n=== cache names ===")
    try:
        update.check_cache_names()
    except CacheNameError as error:
        raise SystemExit(f"{error}")
    log(f"  {len(update.cache_directories())} expected cache director(ies), "
        f"no near-miss names")

    # --- 2. cache collisions, before anything expensive ---------------------
    found = update.collisions()
    status["collisions"] = [str(c) for c in found]
    status["pruned"] = []
    if found:
        log(f"\n=== cache collisions ({len(found)}) ===")
        for collision in found:
            log(f"  {collision}")
        if not args.prune_stale:
            raise SystemExit(
                f"{len(found)} file(s) are in two caches, which the loader "
                f"refuses (one season must live in exactly one cache). Re-run "
                f"with --prune-stale to delete the superseded copies and their "
                f"manifest entries, or delete them by hand."
            )
        removed = update.prune_superseded(found)
        status["pruned"] = [str(p) for p in removed]
        log(f"  pruned {len(removed)} superseded file(s) and their manifest entries")

    # --- 3. cache integrity --------------------------------------------------
    log("\n=== cache integrity ===")
    for directory in football_data.CACHE_DIRS:
        if not Path(directory).is_dir():
            log(f"  {directory.name}: absent")
            continue
        try:
            verified = verify_cache(directory)
        except CacheIntegrityError as error:
            raise SystemExit(f"{error}")
        log(f"  {directory.name}: {len(verified)} file(s) verified")

    # --- 4. seed live/ from the frozen archive ------------------------------
    # The matchday-0 snapshot anchors every later matchday's d_p_* columns, so
    # it has to come from the model being published. After a model switch the
    # old snapshot is still on disk, and leaving it there would compute a
    # one-matchday change across two different models - which is not a change
    # in anything.
    seed = paths.forecast_path(0, cfg.label, **live)
    previous_model = previous.get("model")
    switched = previous_model not in (None, args.model)
    archive = paths.archive_dir(args.model)
    if not seed.exists() or switched:
        archived = archive / f"forecast_{cfg.label}.csv"
        if switched and not archived.exists():
            # Seeding matchday 0 from the previous model's archive is exactly
            # the cross-model delta this step exists to prevent, so a switch
            # with no archive for the new model refuses instead.
            raise SystemExit(
                f"{archived.relative_to(config.PROJECT_ROOT)} does not exist, "
                f"so matchday 0 cannot be seeded from the model being "
                f"published. Produce the frozen archive for {args.model} first "
                f"(scripts/simulate.py --model {args.model} -n "
                f"{config.N_SIMULATIONS}, then scripts/write_model_json.py), "
                f"or every d_p_* column would be a change across two models."
            )
        if switched:
            # Every snapshot in live/ was produced by the old model, matchday 0
            # included. They are the record of what was published at the time,
            # so the whole series is moved out of the way rather than deleted,
            # and the app reads it back as that model's own history. Moving
            # before seeding, so the record is complete and matchday 0 is not
            # the one snapshot overwritten in place.
            history = paths.history_dir(previous_model, cfg.label, **live)
            history.mkdir(parents=True, exist_ok=True)
            moved = []
            for pattern in ("forecast_md*.csv", "positions_md*.csv",
                            "knockout_paths_md*.csv", "ratings_md*.csv",
                            "remaining_fixtures_md*.csv"):
                for stale in sorted(paths.season_dir(cfg.label, **live)
                                    .glob(pattern)):
                    os.replace(stale, history / stale.name)
                    moved.append(stale.name)
                    log(f"  moved {stale.name} -> {history.name}/: "
                        f"published by {previous_model}")
            if moved:
                (history / "about.json").write_text(json.dumps({
                    "model": previous_model,
                    "superseded_by": args.model,
                    "superseded_at_utc": datetime.now(timezone.utc)
                        .isoformat(timespec="seconds"),
                    "snapshots": sorted(moved),
                    "note": "What this model published, kept as the record. "
                            "The scores for these matchdays stay in "
                            "live/<season>/ as matchday_k_evaluation.csv, and "
                            "every model's score on every played match is in "
                            "model_tracking.csv.",
                }, indent=2) + "\n", encoding="utf-8")
            status["history_dir"] = history.name
            status["history_moved"] = sorted(moved)
        if archived.exists():
            update.publish(archived, seed)
            why = (f"model changed from {previous_model} to {args.model}"
                   if switched else "no published matchday 0 yet")
            log(f"\nseeded {seed.name} from "
                f"{archived.relative_to(config.PROJECT_ROOT)} ({why})")
        # The weekly run forecasts through the latest complete matchday, so it
        # can never produce a matchday-0 artefact again. These are seeded from
        # the archive alongside the forecast, or the detail views would have
        # nothing for matchday 0.
        for archived_name, destination in (
                (f"positions_{cfg.label}.csv",
                 paths.positions_path(0, cfg.label, **live)),
                (f"knockout_paths_{cfg.label}.csv",
                 paths.knockout_paths_path(0, cfg.label, **live)),
                (f"ratings_{cfg.label}.csv",
                 paths.ratings_path(0, cfg.label, **live)),
                (f"remaining_fixtures_{cfg.label}.csv",
                 paths.remaining_fixtures_path(0, cfg.label, **live)),
                ("club_inputs.csv", paths.club_inputs_path(cfg.label, **live))):
            archived_detail = archive / archived_name
            if archived_detail.exists() and (not destination.exists() or switched):
                update.publish(archived_detail, destination)
                log(f"  seeded {destination.name}")

    # --- 5. refresh the played matches, exactly once ------------------------
    log("\n=== results ===")
    try:
        played = league_results.load_played(cfg.label, refresh=not args.no_refresh)
    except league_results.MappingError as error:
        raise SystemExit(
            f"could not read the played matches: {error}. The cached copy is "
            f"left intact, so --no-refresh will run from it."
        )
    status["wikipedia_refreshed"] = not args.no_refresh
    fixtures = pd.read_csv(fx.fixtures_path(cfg.label), parse_dates=["date"])
    log(f"  {len(played)} played match(es) across "
        f"matchday(s) {sorted(set(played.matchday.tolist()))}")

    # --- 6. which matchday the forecast can run through ---------------------
    matchday = update.latest_complete_prefix(played, fixtures)
    partial = update.partial_matchdays(played, fixtures)
    beyond = update.missing_before(played, fixtures, matchday)
    if beyond:
        raise SystemExit(
            f"matchday(s) {beyond} are complete but matchday {matchday + 1} is "
            f"not, so a round was skipped and the forecast cannot advance past "
            f"{matchday}. Check for a postponed fixture."
        )
    if partial and not args.allow_partial:
        first = min(partial)
        have, want = partial[first]
        raise SystemExit(
            f"matchday {first} is partially played ({have} of {want} results); "
            f"mid-round the next-matchday table would be half stale. Wait for "
            f"the round to finish, or pass --allow-partial to forecast through "
            f"matchday {matchday} anyway."
        )
    if partial:
        log(f"  NOTE: matchday {min(partial)} is partially played "
            f"({partial[min(partial)][0]} of {partial[min(partial)][1]}); "
            f"forecasting through matchday {matchday} at your request")
    log(f"  forecasting through matchday {matchday}")

    as_of = as_of_for_matchday(fixtures, matchday, cfg.cutoff_date)
    later = fixtures[fixtures.matchday == matchday + 1]
    status.update(
        through_matchday=matchday, as_of=str(as_of),
        played_matches=int(len(played)),
        completed_matchdays=league_results.completed_matchdays(played, fixtures),
        partial_matchdays={str(k): list(v) for k, v in partial.items()},
        next_matchday=matchday + 1 if len(later) else None,
        next_matchday_first_date=(str(later.date.min().date()) if len(later) else None),
    )

    # --- 7. domestic coverage, before the five-minute simulation ------------
    # The ratings are replayed from domestic results, so a hole here is a hole
    # in every club's rating, and a forecast published from one looks entirely
    # normal: 36 clubs, 144 fixtures, every check green. The two faults below
    # both happened on 2026-10-20 and neither refused anything.
    log("\n=== domestic coverage ===")
    domestic = football_data.load_results(as_of)
    coverage = league_results.domestic_coverage(domestic, cfg.label)
    log(coverage.to_string(index=False))
    behind = [r for r in coverage.itertuples()
              if r.last_result is not None
              and (as_of - r.last_result).days > league_results.DOMESTIC_PUBLISH_DAYS]
    for row in behind:
        # Not a refusal: the bar is on the cache as a whole, as declared. One
        # league alone this far back is still worth seeing in the log.
        log(f"  NOTE: {row.code} ({row.league}) ends {row.last_result}, "
            f"{(as_of - row.last_result).days} days before the as-of date")
    gaps = league_results.domestic_publication_gaps(
        domestic, season_label=cfg.label, as_of=as_of)
    status["domestic_gaps"] = gaps
    status["allow_stale"] = bool(args.allow_stale)
    status["domestic_last_result"] = (
        str(domestic.date.max().date()) if len(domestic) else None)
    if gaps:
        for reason in gaps:
            log(f"  {reason}")
        if not args.allow_stale:
            raise SystemExit(
                "the domestic cache is not fit to publish from: "
                + " ".join(gaps)
                + " Fetch a current copy into "
                  f"{football_data.CURRENT_CACHE_DIR.relative_to(config.PROJECT_ROOT)}/"
                  " and re-run, or pass --allow-stale to publish anyway."
            )
        log("  publishing anyway: --allow-stale")
    else:
        log(f"  every main league has {cfg.label} results, and domestic "
            f"results end {domestic.date.max().date()}, "
            f"{(as_of - domestic.date.max().date()).days} day(s) before the "
            f"as-of date")

    # --- 8. idempotence: unchanged inputs do no work ------------------------
    digest = update.inputs_hash(played)
    status["inputs_hash"] = digest
    unchanged = (previous.get("state") == "ok"
                 and previous.get("through_matchday") == matchday
                 and previous.get("inputs_hash") == digest
                 and previous.get("iterations") == args.iterations
                 and previous.get("model") == args.model)
    if unchanged and not args.force:
        log(f"\nalready up to date through matchday {matchday} "
            f"(run {previous.get('finished_at_utc')}, inputs unchanged); "
            f"nothing to do. Pass --force to re-run anyway.")
        finish("ok", "no change", artefacts=previous.get("artefacts", []),
               verification_passed=previous.get("verification_passed"))
        return 0

    # --- 9. restore the delta anchor ----------------------------------------
    # simulate.py computes d_p_* against the previous matchday's file in
    # outputs/, which the test suite overwrites with smoke runs. Copying the
    # published snapshot back makes live/ the anchor of the chain.
    if matchday > 0:
        anchor_name = (f"forecast_{cfg.label}_md{matchday - 1}.csv" if matchday > 1
                       else f"forecast_{cfg.label}.csv")
        anchor = paths.forecast_path(matchday - 1, cfg.label, **live)
        if not anchor.exists() and switched:
            # A switch has just moved the previous matchday's snapshot into the
            # old model's history, where it belongs: it is not this model's and
            # a delta against it would be a change of model, not of chances.
            # Matchday 0 is this model's own, and is what the app will compare
            # against too, so the stored columns and the recomputed ones agree.
            anchor = paths.forecast_path(0, cfg.label, **live)
            log(f"  delta anchor: matchday {matchday - 1} belongs to "
                f"{previous_model}, so the d_p_* columns are against this "
                f"model's own matchday 0 - a {matchday}-matchday change, which "
                f"is what the app shows after a switch")
        if anchor.exists():
            update.publish(anchor, OUTPUTS_DIR / anchor_name)
            log(f"  delta anchor: {anchor.name} -> outputs/{anchor_name}")
        else:
            log(f"  NOTE: {anchor.name} is not published, so the d_p_* columns "
                f"will be computed against whatever outputs/{anchor_name} holds")

    # --- 10. the forecast -----------------------------------------------------
    log(f"\n=== forecast through matchday {matchday} ===")
    code = run_step(log, [
        python, str(SCRIPTS / "simulate.py"), "--season", cfg.label,
        "-n", str(args.iterations), "--seed", str(args.seed),
        "--model", args.model, "--through-matchday", str(matchday),
    ])
    status["simulate_returncode"] = code
    status["verification_passed"] = code == 0
    if code != 0:
        # simulate.py returns 1 when a verification check fails, having already
        # written its CSV. Nothing is published from a run that failed one.
        raise SystemExit(
            f"scripts/simulate.py exited {code}: a verification check failed, "
            f"so nothing was published. See the log above."
        )

    # --- 11. score the played matchdays --------------------------------------
    log("\n=== scoring ===")
    scored, produced = [], []
    for played_matchday in range(1, matchday + 1):
        target = paths.evaluation_path(played_matchday, cfg.label, **live)
        # A model switch scores every played matchday again, because this
        # model has no scores for the earlier ones yet. The published record is
        # not touched by that: step 14 writes a differing score beside it.
        if (target.exists() and not args.rescore and not switched
                and played_matchday != matchday):
            log(f"  matchday {played_matchday}: already scored")
            scored.append(played_matchday)
            continue
        code = run_step(log, [
            python, str(SCRIPTS / "evaluate_matchday.py"), "--season", cfg.label,
            "--matchday", str(played_matchday), "--model", args.model,
        ])
        if code != 0:
            raise SystemExit(
                f"scripts/evaluate_matchday.py exited {code} on matchday "
                f"{played_matchday}; nothing was published")
        scored.append(played_matchday)
        # Only a matchday scored in *this* run has a scratch file this run
        # wrote. A skipped one is already published and correct, and reading
        # outputs/ for it would read whatever was left there.
        produced.append(played_matchday)

    # --- 12. the information set --------------------------------------------
    log("\n=== information set ===")
    information = league_results.information(cfg.label, as_of=as_of)
    log(information.to_string(index=False))
    stale = league_results.staleness(information, as_of)
    if stale:
        log(f"  NOTE: {stale}")
    status["sources"] = {
        row.source: {"cached_to": str(row.cached_to), "used_to": str(row.used_to)}
        for row in information.itertuples()
    }

    # --- 13. prospective model tracking --------------------------------------
    # Every candidate's probabilities for the played matches, all from the same
    # frozen cutoff information set. Nothing here changes the forecast; it
    # accumulates evidence on this season, which 144 matches cannot settle.
    log("\n=== model tracking ===")
    from ucl import tracking

    tracked = tracking.track(cfg.label, played)
    if len(tracked):
        tracked.to_csv(paths.tracking_path(cfg.label, **live), index=False,
                       float_format="%.6f")
        summary = tracking.summary(tracked)
        for row in summary.itertuples():
            value = getattr(row, "vs_reference", None)
            gap = (f"  {value:+.4f} vs {tracking.REFERENCE} "
                   f"(SE {row.vs_reference_standard_error:.4f})"
                   if isinstance(value, float) and value == value else "")
            log(f"  {row.model:8} log-loss {row.log_loss:.4f} "
                f"RPS {row.rps:.4f} over {row.n} matches{gap}")
        log("  monitoring, not a gate: over all 144 league-phase matches the "
            "paired standard error will still be about three times the "
            "difference the backtest resolved.")
        status["tracked_models"] = list(summary.model)
    else:
        log("  no played matches yet")

    # --- 14. publish ---------------------------------------------------------
    log("\n=== published ===")
    artefacts = []
    suffix = f"_md{matchday}" if matchday > 0 else ""
    forecast = OUTPUTS_DIR / f"forecast_{cfg.label}{suffix}.csv"
    artefacts.append(update.publish(
        forecast, paths.forecast_path(matchday, cfg.label, **live)).name)
    preview = OUTPUTS_DIR / f"next_matchday_{matchday + 1}_{cfg.label}.csv"
    if preview.exists():
        # The pre-match table for the round about to be played. Published once
        # and then left alone like a score: it is the genuine prediction, made
        # before the matches, and a later model cannot have made it.
        destination = paths.next_matchday_path(matchday + 1, cfg.label, **live)
        if destination.exists() and switched:
            log(f"  NOTE: {destination.name} already exists and was published "
                f"by {previous_model} before matchday {matchday + 1}; left as "
                f"it is, because it is a pre-match prediction and this model "
                f"did not make it")
        else:
            artefacts.append(update.publish(preview, destination).name)
    elif len(later):
        log(f"  NOTE: {preview.name} was not written")
    # The four artefacts the detail views read. Each is guarded: a view that
    # cannot find its file says so and the rest of the app is unaffected.
    for source_name, destination in (
            (f"positions_{cfg.label}{suffix}.csv",
             paths.positions_path(matchday, cfg.label, **live)),
            (f"knockout_paths_{cfg.label}{suffix}.csv",
             paths.knockout_paths_path(matchday, cfg.label, **live)),
            (f"ratings_{cfg.label}{suffix}.csv",
             paths.ratings_path(matchday, cfg.label, **live)),
            (f"remaining_fixtures_{cfg.label}{suffix}.csv",
             paths.remaining_fixtures_path(matchday, cfg.label, **live))):
        source = OUTPUTS_DIR / source_name
        if source.exists():
            artefacts.append(update.publish(source, destination).name)
        else:
            log(f"  NOTE: {source_name} was not written")
    # The published record of each matchday's score, and this model's score
    # beside it when they are not the same thing. Which case it is comes from
    # comparing the files, so it holds across any number of model switches and
    # needs no provenance field to be right.
    added = []
    for played_matchday in produced:
        source = OUTPUTS_DIR / f"matchday_{played_matchday}_evaluation_{cfg.label}.csv"
        if not source.exists():
            continue
        record = paths.evaluation_path(played_matchday, cfg.label, **live)
        if record.exists() and not filecmp.cmp(source, record, shallow=False):
            destination = paths.evaluation_path(
                played_matchday, cfg.label, model=args.model, **live)
            log(f"  matchday {played_matchday}: {record.name} was published by "
                f"another model and is left as it is; {args.model}'s score for "
                f"the same matches goes to {destination.name}")
            added.append(destination.name)
        else:
            destination = record
        artefacts.append(update.publish(source, destination).name)
    status["evaluations_added"] = added
    information.to_csv(paths.information_path(cfg.label, **live), index=False)
    artefacts.append(paths.information_path(cfg.label, **live).name)
    if paths.tracking_path(cfg.label, **live).exists():
        artefacts.append(paths.tracking_path(cfg.label, **live).name)
    published_root = paths.season_dir(cfg.label, **live)
    for name in artefacts:
        log(f"  {published_root}/{name}")

    # --- 15. status ----------------------------------------------------------
    finish("ok", artefacts=artefacts)
    log(f"\ndone in {status['duration_seconds']}s, "
        f"through matchday {matchday}, as of {as_of}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
