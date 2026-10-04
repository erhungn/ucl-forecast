"""Refresh the results, re-forecast, score what was played, publish it all.

One command for the weekly cycle. It refuses rather than guesses, and the order
is deliberate: everything that can refuse does so before the five-minute
simulation, not after it.

    1. cache collisions      - a file in two caches, which the loader refuses
    2. cache integrity       - every manifest hash re-checked
    3. seed live/            - matchday 0 from the frozen archive
    4. refresh Wikipedia     - exactly once, here and nowhere else
    5. detect the matchday   - the contiguous complete prefix
    6. idempotence gate      - unchanged inputs do no work
    7. restore the anchor    - the delta chain reads live/, not scratch
    8. forecast              - scripts/simulate.py
    9. score                 - scripts/evaluate_matchday.py per played matchday
   10. information set       - what each source knew
   11. model tracking        - every candidate scored on the played matches
   12. publish               - atomically, into live/<season>/
   13. status.json           - provenance for the run

Artefacts go to ``live/<season>/``, which only this script writes. ``outputs/``
stays scratch: the test suite runs the simulator into it with a handful of
iterations, so a forecast there may be a smoke test.
"""

from __future__ import annotations

import argparse
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
from ucl.data.manifest import CacheIntegrityError, verify_cache
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
    parser.add_argument("--model", choices=("v2", "v1", "elo"), default="v2")
    parser.add_argument("--prune-stale", action="store_true",
                        help="delete cache files superseded by a newer copy")
    parser.add_argument("--no-refresh", action="store_true",
                        help="use the cached Wikipedia page instead of fetching")
    parser.add_argument("--allow-partial", action="store_true",
                        help="forecast through the last complete matchday even "
                             "while the next one is half played")
    parser.add_argument("--force", action="store_true",
                        help="re-run even when the inputs are unchanged")
    parser.add_argument("--rescore", action="store_true",
                        help="score every played matchday again")
    parser.add_argument("--python", default=None,
                        help="interpreter for the pipeline steps "
                             "(default: .venv/bin/python)")
    args = parser.parse_args(argv)
    cfg = season_config(args.season)

    # Read the last run's record before this run's status overwrites it; the
    # idempotence gate in step 6 compares against it.
    previous = update.read_status(paths.status_path(cfg.label))
    log = Log(paths.log_path(cfg.label))
    started = time.perf_counter()
    status = {
        "season": cfg.label, "state": "running", "message": "",
        "pid": os.getpid(), "model": args.model, "iterations": args.iterations,
        "seed": args.seed, "git_commit": git_commit(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    update.write_status(paths.status_path(cfg.label), status)

    def finish(state: str, message: str = "", **extra) -> None:
        status.update(
            state=state, message=message,
            finished_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_seconds=round(time.perf_counter() - started, 1), **extra)
        update.write_status(paths.status_path(cfg.label), status)

    try:
        return _run(args, cfg, log, status, finish, previous)
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


def _run(args, cfg, log: Log, status: dict, finish, previous: dict) -> int:
    python = interpreter(args.python)
    log(f"=== weekly update: {cfg.label}, model {args.model}, "
        f"{args.iterations} iterations, seed {args.seed} ===")
    log(f"  interpreter {python}")

    # --- 1. cache collisions, before anything expensive ---------------------
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

    # --- 2. cache integrity --------------------------------------------------
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

    # --- 3. seed live/ from the frozen archive ------------------------------
    # The matchday-0 snapshot anchors every later matchday's d_p_* columns, so
    # it has to come from the model being published. After a model switch the
    # old snapshot is still on disk, and leaving it there would compute a
    # one-matchday change across two different models - which is not a change
    # in anything.
    seed = paths.forecast_path(0, cfg.label)
    switched = previous.get("model") not in (None, args.model)
    if not seed.exists() or switched:
        archived = paths.ARCHIVE_DIR / f"forecast_{cfg.label}.csv"
        if archived.exists():
            update.publish(archived, seed)
            why = (f"model changed from {previous.get('model')} to {args.model}"
                   if switched else "no published matchday 0 yet")
            log(f"\nseeded {seed.name} from "
                f"{archived.relative_to(config.PROJECT_ROOT)} ({why})")
            if switched:
                # Every later snapshot was produced by the old model.
                for stale in sorted(paths.season_dir(cfg.label).glob(
                        "forecast_md*.csv")):
                    if stale != seed:
                        stale.unlink()
                        log(f"  removed {stale.name}: produced by "
                            f"{previous.get('model')}")

    # --- 4. refresh the played matches, exactly once ------------------------
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

    # --- 5. which matchday the forecast can run through ---------------------
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

    # --- 6. idempotence: unchanged inputs do no work ------------------------
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

    # --- 7. restore the delta anchor ----------------------------------------
    # simulate.py computes d_p_* against the previous matchday's file in
    # outputs/, which the test suite overwrites with smoke runs. Copying the
    # published snapshot back makes live/ the anchor of the chain.
    if matchday > 0:
        anchor_name = (f"forecast_{cfg.label}_md{matchday - 1}.csv" if matchday > 1
                       else f"forecast_{cfg.label}.csv")
        anchor = paths.forecast_path(matchday - 1, cfg.label)
        if anchor.exists():
            update.publish(anchor, OUTPUTS_DIR / anchor_name)
            log(f"  delta anchor: {anchor.name} -> outputs/{anchor_name}")
        else:
            log(f"  NOTE: {anchor.name} is not published, so the d_p_* columns "
                f"will be computed against whatever outputs/{anchor_name} holds")

    # --- 8. the forecast -----------------------------------------------------
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

    # --- 9. score the played matchdays --------------------------------------
    log("\n=== scoring ===")
    scored = []
    for played_matchday in range(1, matchday + 1):
        target = paths.evaluation_path(played_matchday, cfg.label)
        if target.exists() and not args.rescore and played_matchday != matchday:
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

    # --- 10. the information set --------------------------------------------
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

    # --- 11. prospective model tracking --------------------------------------
    # Every candidate's probabilities for the played matches, all from the same
    # frozen cutoff information set. Nothing here changes the forecast; it
    # accumulates evidence on this season, which 144 matches cannot settle.
    log("\n=== model tracking ===")
    from ucl import tracking

    tracked = tracking.track(cfg.label, played)
    if len(tracked):
        tracked.to_csv(paths.tracking_path(cfg.label), index=False,
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

    # --- 12. publish ---------------------------------------------------------
    log("\n=== published ===")
    artefacts = []
    suffix = f"_md{matchday}" if matchday > 0 else ""
    forecast = OUTPUTS_DIR / f"forecast_{cfg.label}{suffix}.csv"
    artefacts.append(update.publish(
        forecast, paths.forecast_path(matchday, cfg.label)).name)
    preview = OUTPUTS_DIR / f"next_matchday_{matchday + 1}_{cfg.label}.csv"
    if preview.exists():
        artefacts.append(update.publish(
            preview, paths.next_matchday_path(matchday + 1, cfg.label)).name)
    elif len(later):
        log(f"  NOTE: {preview.name} was not written")
    for played_matchday in scored:
        source = OUTPUTS_DIR / f"matchday_{played_matchday}_evaluation_{cfg.label}.csv"
        if source.exists():
            artefacts.append(update.publish(
                source, paths.evaluation_path(played_matchday, cfg.label)).name)
    information.to_csv(paths.information_path(cfg.label), index=False)
    artefacts.append(paths.information_path(cfg.label).name)
    if paths.tracking_path(cfg.label).exists():
        artefacts.append(paths.tracking_path(cfg.label).name)
    for name in artefacts:
        log(f"  live/{cfg.label}/{name}")

    # --- 13. status ----------------------------------------------------------
    finish("ok", artefacts=artefacts)
    log(f"\ndone in {status['duration_seconds']}s, "
        f"through matchday {matchday}, as of {as_of}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
