"""Forward-chained tuning of Dixon-Coles against the frozen Elo baseline.

Only two things are tuned - the time-decay half-life and the ridge strength -
so the search is an honest grid rather than an optimiser whose path has to be
explained. mu, home, beta and rho are fitted by maximum likelihood inside each
fit.

Protocol is identical to the Elo tuning so the two are comparable: each fold
tunes on the seasons before its test season and scores that season once, and the
pre-declared primary test is x3-weighted pooled log-loss, frozen mode, with a
cluster bootstrap over (season, home club).

If a fold selects a point on the edge of the grid, the grid is extended in that
direction and the fold is re-run, because an edge selection means the optimum
may lie outside what was searched.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ucl.backtest import bootstrap_difference, paired_losses_dc, pooled_objective_dc
from ucl.config import BACKTEST_SEASONS, OUTPUTS_DIR

HALF_LIVES = [91.0, 183.0, 274.0, 365.0, 548.0, 730.0, 1096.0]
RIDGES = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
#: Extensions used when a fold lands on an edge, in each direction.
HALF_LIFE_LOW, HALF_LIFE_HIGH = [45.0, 61.0], [1460.0, 2190.0]
RIDGE_LOW, RIDGE_HIGH = [0.5, 1.0], [400.0, 800.0]


def folds() -> list[tuple[str, tuple[str, ...]]]:
    return [(BACKTEST_SEASONS[i], BACKTEST_SEASONS[:i])
            for i in range(1, len(BACKTEST_SEASONS))]


def search(seasons: tuple[str, ...], half_lives: list[float], ridges: list[float],
           *, european_only: bool) -> tuple[float, float, float, pd.DataFrame]:
    rows = []
    for half_life in half_lives:
        for ridge in ridges:
            value = pooled_objective_dc(
                seasons, half_life, ridge, european_only=european_only)
            rows.append({"half_life": half_life, "ridge": ridge, "value": value})
    frame = pd.DataFrame(rows)
    best = frame.loc[frame.value.idxmin()]
    return float(best.half_life), float(best.ridge), float(best.value), frame


def run(objective: str, draws: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    european_only = objective == "european"
    rows, paired, extensions = [], [], []
    for test, train in folds():
        half_lives, ridges = list(HALF_LIVES), list(RIDGES)
        started = time.perf_counter()
        half_life, ridge, value, grid = search(
            train, half_lives, ridges, european_only=european_only)

        # Extend the grid if the pick sits on an edge, then search again.
        for _ in range(3):
            extended = False
            if half_life == min(half_lives):
                half_lives = sorted(set(half_lives + HALF_LIFE_LOW)); extended = True
            if half_life == max(half_lives):
                half_lives = sorted(set(half_lives + HALF_LIFE_HIGH)); extended = True
            if ridge == min(ridges):
                ridges = sorted(set(ridges + RIDGE_LOW)); extended = True
            if ridge == max(ridges):
                ridges = sorted(set(ridges + RIDGE_HIGH)); extended = True
            if not extended:
                break
            note = (f"{objective}/{test}: selection on a grid edge "
                    f"(half-life {half_life:.0f}d, ridge {ridge:g}); extended and re-ran")
            extensions.append(note)
            print(f"    {note}")
            half_life, ridge, value, grid = search(
                train, half_lives, ridges, european_only=european_only)

        print(f"  [{objective:8s}] fold {test}: train {'+'.join(train):29s} "
              f"half-life {half_life:6.0f}d ridge {ridge:6g} "
              f"training {value:.5f} ({len(grid)} grid points, "
              f"{time.perf_counter() - started:.0f}s)")

        frame = paired_losses_dc(test, half_life, ridge)
        frame["fold"] = test
        paired.append(frame)
        walk = paired_losses_dc(test, half_life, ridge, mode="walk_forward")
        rows.append({
            "objective": objective, "fold": test, "train": "+".join(train),
            "half_life": half_life, "ridge": ridge, "training_objective": value,
            "grid_points": len(grid),
            "delta_frozen": float((frame.difference * frame.weight).sum() / frame.weight.sum()),
            "delta_walk_forward": float((walk.difference * walk.weight).sum() / walk.weight.sum()),
        })
    return pd.DataFrame(rows), pd.concat(paired, ignore_index=True), extensions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries, pooled, all_extensions = {}, {}, []
    for objective in ("weighted", "european"):
        print(f"\n=== Dixon-Coles forward chaining, {objective} objective ===")
        summary, paired, extensions = run(objective, args.draws)
        summaries[objective], pooled[objective] = summary, paired
        all_extensions.extend(extensions)
        summary.to_csv(OUTPUTS_DIR / f"dc_forward_chain_{objective}.csv", index=False)
        paired.to_csv(OUTPUTS_DIR / f"dc_forward_chain_paired_{objective}.csv", index=False)

    print("\n=== per fold (primary: weighted, frozen) ===")
    primary = summaries["weighted"]
    for row in primary.itertuples():
        fold = pooled["weighted"].query("fold == @row.fold")
        cluster = bootstrap_difference(fold, cluster=True, draws=args.draws)
        per_match = bootstrap_difference(fold, cluster=False, draws=args.draws)
        print(f"  {row.fold}: half-life {row.half_life:6.0f}d ridge {row.ridge:6g} | "
              f"delta {row.delta_frozen:+.5f} cluster [{cluster['low']:+.5f}, "
              f"{cluster['high']:+.5f}] per-match [{per_match['low']:+.5f}, "
              f"{per_match['high']:+.5f}]")

    print("\n=== POOLED, primary test (Dixon-Coles vs the Elo baseline) ===")
    frame = pooled["weighted"]
    cluster = bootstrap_difference(frame, cluster=True, draws=args.draws)
    per_match = bootstrap_difference(frame, cluster=False, draws=args.draws)
    print(f"  matches {cluster['n']}, clusters {cluster['clusters']}")
    print(f"  weighted mean delta log-loss: {cluster['difference']:+.5f} "
          f"(negative = Dixon-Coles better)")
    print(f"  cluster   95% CI [{cluster['low']:+.5f}, {cluster['high']:+.5f}]  "
          f"excludes zero: {cluster['excludes_zero']}")
    print(f"  per-match 95% CI [{per_match['low']:+.5f}, {per_match['high']:+.5f}]  "
          f"excludes zero: {per_match['excludes_zero']}")
    accepted = bool(cluster["excludes_zero"] and cluster["difference"] < 0)
    print(f"\n  DECISION: {'ACCEPT Dixon-Coles' if accepted else 'KEEP the Elo baseline'}")

    euro = bootstrap_difference(pooled["european"], cluster=True, draws=args.draws)
    print("\n=== sensitivities (not the decision) ===")
    print(f"  pure-European objective: pooled delta {euro['difference']:+.5f} "
          f"[{euro['low']:+.5f}, {euro['high']:+.5f}]")
    print("  walk-forward per fold: "
          + ", ".join(f"{r.fold} {r.delta_walk_forward:+.5f}" for r in primary.itertuples()))
    domestic = frame[frame.competition == "domestic"]
    dom = bootstrap_difference(domestic, cluster=True, draws=args.draws)
    print(f"  domestic only ({dom['n']} matches): {dom['difference']:+.5f} "
          f"[{dom['low']:+.5f}, {dom['high']:+.5f}]")
    european_rows = frame[frame.competition != "domestic"]
    eur = bootstrap_difference(european_rows, cluster=True, draws=args.draws)
    print(f"  European only ({eur['n']} matches): {eur['difference']:+.5f} "
          f"[{eur['low']:+.5f}, {eur['high']:+.5f}]")
    print(f"  grid extensions: {len(all_extensions)}")
    for note in all_extensions:
        print(f"    {note}")

    with open(OUTPUTS_DIR / "dc_decision.json", "w", encoding="utf-8") as fh:
        json.dump({"primary": cluster, "per_match": per_match, "european": euro,
                   "domestic_only": dom, "european_only": eur,
                   "extensions": all_extensions, "accepted": accepted}, fh, indent=2)
        fh.write("\n")
    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
