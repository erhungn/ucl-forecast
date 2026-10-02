"""Tune the Elo parameters on one season, holding another out.

The search is a coarse grid over the two categorical choices - what a club
regresses toward between seasons, and whether the goal-difference weighting is
the classic rule or a tunable exponent - with Nelder-Mead on the continuous
parameters inside each combination.

Two objectives are reported: the pooled log-loss with European matches weighted
up, which is what the tuning optimises, and pure European log-loss alongside it,
so the sensitivity of the choice is visible.

The market never enters the objective.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ucl.backtest import score_season
from ucl.config import OUTPUTS_DIR
from ucl.models.elo import CARRYOVER_TARGETS, EloParams

BOUNDS = {
    "k_domestic": (2.0, 45.0),
    "k_european": (1.0, 55.0),
    "home_advantage": (30.0, 95.0),
    "season_carryover": (0.60, 1.00),
    "gd_exponent": (0.0, 0.9),
}
CONTINUOUS = ["k_domestic", "k_european", "home_advantage", "season_carryover"]


def build(values: dict, target: str, gd_rule: str) -> EloParams:
    return EloParams(
        k_domestic=values["k_domestic"],
        k_european=values["k_european"],
        home_advantage=values["home_advantage"],
        season_carryover=values["season_carryover"],
        gd_exponent=None if gd_rule == "classic" else values["gd_exponent"],
        carryover_target=target,
    )


def objective_value(params: EloParams, season: str, objective: str,
                    weight: float) -> float:
    score = score_season(season, params, modes=("frozen",))
    return score.european_log_loss() if objective == "european" else score.objective(weight)


def search(season: str, target: str, gd_rule: str, objective: str, weight: float,
           maxiter: int) -> tuple[EloParams, float, int]:
    keys = CONTINUOUS + (["gd_exponent"] if gd_rule == "exponent" else [])
    start = np.array([
        {"k_domestic": 20.0, "k_european": 25.0, "home_advantage": 65.0,
         "season_carryover": 0.90, "gd_exponent": 0.4}[k] for k in keys
    ])
    evaluations = 0

    def wrapped(x: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        values = {}
        for key, value in zip(keys, x):
            low, high = BOUNDS[key]
            if not (low <= value <= high):
                return 50.0           # Nelder-Mead has no bounds; wall it off.
            values[key] = float(value)
        values.setdefault("gd_exponent", 0.0)
        return objective_value(build(values, target, gd_rule), season, objective, weight)

    result = minimize(wrapped, start, method="Nelder-Mead",
                      options={"maxiter": maxiter, "xatol": 0.25, "fatol": 1e-5})
    values = {k: float(v) for k, v in zip(keys, result.x)}
    values.setdefault("gd_exponent", 0.0)
    return build(values, target, gd_rule), float(result.fun), evaluations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--european-weight", type=float, default=3.0)
    parser.add_argument("--maxiter", type=int, default=160)
    args = parser.parse_args(argv)

    baseline = score_season(args.season, EloParams(), modes=("frozen",))
    print(f"tuning on {args.season}")
    print(f"baseline (current defaults): weighted {baseline.objective(args.european_weight):.5f}, "
          f"european {baseline.european_log_loss():.5f}")

    rows = []
    started = time.perf_counter()
    for objective in ("weighted", "european"):
        for target, gd_rule in itertools.product(CARRYOVER_TARGETS, ("classic", "exponent")):
            tick = time.perf_counter()
            params, value, evaluations = search(
                args.season, target, gd_rule, objective, args.european_weight, args.maxiter
            )
            score = score_season(args.season, params, modes=("frozen",))
            rows.append({
                "objective": objective, "carryover_target": target, "gd_rule": gd_rule,
                "value": value, "weighted": score.objective(args.european_weight),
                "european": score.european_log_loss(), "evaluations": evaluations,
                "seconds": time.perf_counter() - tick, **params.as_dict(),
            })
            print(f"  [{objective:8s}] {target:14s} {gd_rule:8s} -> "
                  f"weighted {rows[-1]['weighted']:.5f}  european {rows[-1]['european']:.5f}  "
                  f"({evaluations} evals, {rows[-1]['seconds']:.0f}s)")

    frame = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUTS_DIR / f"tuning_{args.season}.csv", index=False)

    best_weighted = frame[frame.objective == "weighted"].nsmallest(1, "weighted").iloc[0]
    best_european = frame[frame.objective == "european"].nsmallest(1, "european").iloc[0]
    print(f"\nbest under the weighted objective: {best_weighted.carryover_target} / "
          f"{best_weighted.gd_rule}, weighted {best_weighted.weighted:.5f}")
    print(f"best under pure European log-loss : {best_european.carryover_target} / "
          f"{best_european.gd_rule}, european {best_european.european:.5f}")
    chosen = {k: best_weighted[k] for k in
              ("k_domestic", "k_european", "home_advantage", "season_carryover",
               "gd_exponent", "carryover_target")}
    with open(OUTPUTS_DIR / f"tuned_params_{args.season}.json", "w", encoding="utf-8") as fh:
        json.dump({k: (None if pd.isna(v) else v) for k, v in chosen.items()}, fh, indent=2)
        fh.write("\n")
    print(f"total {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
