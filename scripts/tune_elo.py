"""Forward-chained tuning of the Elo parameters across five backtest seasons.

For each test season from 2022-23 on, the parameters are tuned on the seasons
before it and the test season is scored exactly once. The goal-difference arm -
the classic World Football Elo weighting against a tunable exponent - is chosen
inside each fold on the training seasons alone; the test season never decides
between arms.

The pre-declared primary test is the x3-weighted pooled log-loss in frozen mode,
and only that decides acceptance, on a cluster bootstrap over (season, home
club) clusters. Pure-European log-loss, walk-forward scoring and the arm
comparison are reported as sensitivity analyses.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ucl.backtest import bootstrap_difference, paired_losses, pooled_objective
from ucl.config import BACKTEST_SEASONS, OUTPUTS_DIR
from ucl.models.elo import EloParams

BOUNDS = {
    "k_domestic": (2.0, 45.0),
    "k_european": (1.0, 55.0),
    "season_carryover": (0.60, 1.00),
    "gd_exponent": (0.0, 0.9),
}
START = {"k_domestic": 20.0, "k_european": 25.0, "season_carryover": 0.90,
         "gd_exponent": 0.4}
ARMS = ("classic", "exponent")
DEFAULTS = EloParams()


def folds() -> list[tuple[str, tuple[str, ...]]]:
    """(test season, training seasons) for every fold after the first season."""
    return [
        (BACKTEST_SEASONS[i], BACKTEST_SEASONS[:i])
        for i in range(1, len(BACKTEST_SEASONS))
    ]


def build(values: dict, arm: str) -> EloParams:
    return EloParams(
        k_domestic=values["k_domestic"],
        k_european=values["k_european"],
        season_carryover=values["season_carryover"],
        gd_exponent=None if arm == "classic" else values["gd_exponent"],
    )


def tune(seasons: tuple[str, ...], arm: str, *, european_only: bool,
         maxiter: int) -> tuple[EloParams, float, int]:
    keys = ["k_domestic", "k_european", "season_carryover"]
    if arm == "exponent":
        keys.append("gd_exponent")
    evaluations = 0

    def wrapped(x: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        values = {}
        for key, value in zip(keys, x):
            low, high = BOUNDS[key]
            if not (low <= value <= high):
                return 50.0
            values[key] = float(value)
        values.setdefault("gd_exponent", 0.0)
        return pooled_objective(seasons, build(values, arm), european_only=european_only)

    result = minimize(wrapped, np.array([START[k] for k in keys]), method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 0.2, "fatol": 1e-5})
    values = {k: float(v) for k, v in zip(keys, result.x)}
    values.setdefault("gd_exponent", 0.0)
    return build(values, arm), float(result.fun), evaluations


def run(objective: str, maxiter: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    european_only = objective == "european"
    rows, paired = [], []
    for test, train in folds():
        picks = {}
        for arm in ARMS:
            started = time.perf_counter()
            params, value, evaluations = tune(
                train, arm, european_only=european_only, maxiter=maxiter)
            picks[arm] = (params, value)
            print(f"  [{objective:8s}] fold {test}: train {'+'.join(train):29s} "
                  f"arm {arm:8s} training objective {value:.5f} "
                  f"({evaluations} evals, {time.perf_counter() - started:.0f}s)")
        # The arm is chosen on the training seasons only.
        arm = min(picks, key=lambda a: picks[a][1])
        params = picks[arm][0]
        print(f"             -> chose arm {arm!r} on training data; scoring {test} once")

        frame = paired_losses(test, params, DEFAULTS)
        frame["fold"] = test
        paired.append(frame)
        walk = paired_losses(test, params, DEFAULTS, mode="walk_forward")
        rows.append({
            "objective": objective, "fold": test, "train": "+".join(train), "arm": arm,
            "training_objective": picks[arm][1],
            "delta_frozen": float((frame.difference * frame.weight).sum() / frame.weight.sum()),
            "delta_walk_forward": float((walk.difference * walk.weight).sum() / walk.weight.sum()),
            "classic_training": picks["classic"][1], "exponent_training": picks["exponent"][1],
            **params.as_dict(),
        })
    return pd.DataFrame(rows), pd.concat(paired, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries, pooled = {}, {}
    for objective in ("weighted", "european"):
        print(f"\n=== forward chaining, {objective} objective ===")
        summary, paired = run(objective, args.maxiter)
        summaries[objective] = summary
        pooled[objective] = paired
        summary.to_csv(OUTPUTS_DIR / f"forward_chain_{objective}.csv", index=False)
        paired.to_csv(OUTPUTS_DIR / f"forward_chain_paired_{objective}.csv", index=False)

    print("\n=== per-fold results (primary objective: weighted, frozen) ===")
    primary = summaries["weighted"]
    for row in primary.itertuples():
        frame = pooled["weighted"]
        fold = frame[frame.fold == row.fold]
        cluster = bootstrap_difference(fold, cluster=True, draws=args.draws)
        print(f"  {row.fold}: arm {row.arm:8s} kd={row.k_domestic:5.1f} ke={row.k_european:5.2f} "
              f"carry={row.season_carryover:.3f} gd={row.gd_exponent} | "
              f"delta {row.delta_frozen:+.5f} "
              f"[{cluster['low']:+.5f}, {cluster['high']:+.5f}]")

    print("\n=== POOLED, primary test ===")
    frame = pooled["weighted"]
    cluster = bootstrap_difference(frame, cluster=True, draws=args.draws)
    per_match = bootstrap_difference(frame, cluster=False, draws=args.draws)
    print(f"  matches {cluster['n']}, clusters {cluster['clusters']}")
    print(f"  weighted mean delta log-loss: {cluster['difference']:+.5f} "
          f"(negative = tuned is better)")
    print(f"  cluster bootstrap 95% CI : [{cluster['low']:+.5f}, {cluster['high']:+.5f}]"
          f"  excludes zero: {cluster['excludes_zero']}")
    print(f"  per-match bootstrap 95% CI: [{per_match['low']:+.5f}, {per_match['high']:+.5f}]"
          f"  excludes zero: {per_match['excludes_zero']}")

    accepted = bool(cluster["excludes_zero"] and cluster["difference"] < 0)
    print(f"\n  DECISION: {'ACCEPT the tuning' if accepted else 'KEEP the defaults'}")

    euro = pooled["european"]
    euro_cluster = bootstrap_difference(euro, cluster=True, draws=args.draws)
    print(f"\n=== sensitivities (not the decision) ===")
    print(f"  pure-European objective, pooled delta {euro_cluster['difference']:+.5f} "
          f"[{euro_cluster['low']:+.5f}, {euro_cluster['high']:+.5f}]")
    print(f"  walk-forward, per fold: "
          + ", ".join(f"{r.fold} {r.delta_walk_forward:+.5f}" for r in primary.itertuples()))
    print(f"  arm chosen per fold: "
          + ", ".join(f"{r.fold} {r.arm}" for r in primary.itertuples()))

    with open(OUTPUTS_DIR / "forward_chain_decision.json", "w", encoding="utf-8") as fh:
        json.dump({"primary": cluster, "per_match": per_match,
                   "european_sensitivity": euro_cluster, "accepted": accepted}, fh, indent=2)
        fh.write("\n")
    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
