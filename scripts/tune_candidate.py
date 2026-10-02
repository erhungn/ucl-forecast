"""Forward-chained evaluation of a candidate information source.

Same protocol as every previous test so the results are comparable: each fold
tunes on the seasons before its test season, scores that season once, and the
pooled difference against the accepted baseline is judged by a cluster
bootstrap over (season, home club).

The primary test differs by candidate, and is pre-declared here:

* ``market`` - **European matches only**. The covariate is built from domestic
  closing odds, so scoring it on domestic outcomes would be circular. That
  leaves a far smaller sample, and the bar is correspondingly harder.
* ``congestion`` - the usual x3-weighted pooled log-loss.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from ucl.backtest import (
    bootstrap_difference, paired_losses_covariate, pooled_objective_covariate,
)
from ucl.config import BACKTEST_SEASONS, OUTPUTS_DIR

# Below ~61 days no league clears MIN_MATCHES, so the covariate is empty and the
# model silently collapses to the baseline. The grid starts where the covariate
# actually exists; the degenerate region is reported, not searched.
HALF_LIVES = [61.0, 91.0, 183.0, 365.0, 548.0, 730.0, 1096.0]
PRIMARY = {"market": "european", "market_elo": "european", "congestion": "pooled",
           "squad_value": "pooled"}
#: market_elo is candidate 1b, declared after candidate 1 failed: the same
#: market strengths mapped into Elo units per league instead of standardised.
POST_HOC = {"market_elo"}


def folds():
    return [(BACKTEST_SEASONS[i], BACKTEST_SEASONS[:i])
            for i in range(1, len(BACKTEST_SEASONS))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=sorted(PRIMARY))
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args(argv)
    candidate = args.candidate
    primary = PRIMARY[candidate]
    european_only = primary == "european"

    started = time.perf_counter()
    label = f"{candidate!r}" + (" [POST HOC - declared after candidate 1 failed]"
                                if candidate in POST_HOC else "")
    print(f"=== candidate {label}: forward chaining, primary test = {primary} ===")
    rows, paired = [], []
    for test, train in folds():
        tick = time.perf_counter()
        if candidate in ("market", "market_elo"):
            scores = {hl: pooled_objective_covariate(
                train, candidate, half_life_days=hl, european_only=european_only)
                for hl in HALF_LIVES}
            half_life = min(scores, key=scores.get)
            edge = half_life in (min(HALF_LIVES), max(HALF_LIVES))
            training_value = scores[half_life]
        else:
            half_life, edge, training_value = 365.0, False, pooled_objective_covariate(
                train, candidate, european_only=european_only)

        frame = paired_losses_covariate(test, candidate, half_life_days=half_life)
        frame["fold"] = test
        paired.append(frame)
        walk = paired_losses_covariate(
            test, candidate, half_life_days=half_life, mode="walk_forward") \
            if False else frame  # walk-forward is not defined for frozen covariates
        subset = frame[frame.competition != "domestic"] if european_only else frame
        rows.append({
            "candidate": candidate, "fold": test, "train": "+".join(train),
            "half_life": half_life, "grid_edge": edge, "training_objective": training_value,
            "delta_primary": float(
                (subset.difference * subset.weight).sum() / subset.weight.sum()),
            "delta_all": float((frame.difference * frame.weight).sum() / frame.weight.sum()),
        })
        print(f"  fold {test}: train {'+'.join(train):29s} half-life {half_life:6.0f}d"
              f"{' (EDGE)' if edge else '       '} training {training_value:.5f} "
              f"delta {rows[-1]['delta_primary']:+.5f} ({time.perf_counter()-tick:.0f}s)")

    summary = pd.DataFrame(rows)
    pooled = pd.concat(paired, ignore_index=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUTS_DIR / f"candidate_{candidate}_folds.csv", index=False)
    pooled.to_csv(OUTPUTS_DIR / f"candidate_{candidate}_paired.csv", index=False)

    scope = pooled[pooled.competition != "domestic"] if european_only else pooled
    print(f"\n=== per fold, {primary} scope ===")
    for row in summary.itertuples():
        fold = scope[scope.fold == row.fold]
        cluster = bootstrap_difference(fold, cluster=True, draws=args.draws)
        per_match = bootstrap_difference(fold, cluster=False, draws=args.draws)
        print(f"  {row.fold}: n={cluster['n']:5d} delta {row.delta_primary:+.5f} "
              f"cluster [{cluster['low']:+.5f}, {cluster['high']:+.5f}] "
              f"per-match [{per_match['low']:+.5f}, {per_match['high']:+.5f}]")

    cluster = bootstrap_difference(scope, cluster=True, draws=args.draws)
    per_match = bootstrap_difference(scope, cluster=False, draws=args.draws)
    print(f"\n=== POOLED, primary test ({primary}) ===")
    print(f"  matches {cluster['n']}, clusters {cluster['clusters']}")
    print(f"  weighted mean delta log-loss: {cluster['difference']:+.5f} "
          f"(negative = the candidate is better)")
    print(f"  cluster   95% CI [{cluster['low']:+.5f}, {cluster['high']:+.5f}]  "
          f"excludes zero: {cluster['excludes_zero']}")
    print(f"  per-match 95% CI [{per_match['low']:+.5f}, {per_match['high']:+.5f}]  "
          f"excludes zero: {per_match['excludes_zero']}")
    accepted = bool(cluster["excludes_zero"] and cluster["difference"] < 0)
    print(f"\n  DECISION: {'ACCEPT ' + candidate if accepted else 'KEEP the baseline'}")

    print("\n=== sensitivities (not the decision) ===")
    other = pooled[pooled.competition == "domestic"] if european_only else \
        pooled[pooled.competition != "domestic"]
    label = "domestic (circular for this candidate)" if european_only else "European only"
    if len(other):
        side = bootstrap_difference(other, cluster=True, draws=args.draws)
        print(f"  {label}: {side['difference']:+.5f} [{side['low']:+.5f}, {side['high']:+.5f}]"
              f" over {side['n']} matches")
    full = bootstrap_difference(pooled, cluster=True, draws=args.draws)
    print(f"  all matches pooled: {full['difference']:+.5f} "
          f"[{full['low']:+.5f}, {full['high']:+.5f}]")
    if summary.grid_edge.any():
        print(f"  NOTE: {int(summary.grid_edge.sum())} fold(s) selected a half-life on the "
              f"grid edge ({min(HALF_LIVES):.0f}-{max(HALF_LIVES):.0f} days)")

    with open(OUTPUTS_DIR / f"candidate_{candidate}_decision.json", "w", encoding="utf-8") as fh:
        json.dump({"candidate": candidate, "post_hoc": candidate in POST_HOC,
                   "primary": primary, "cluster": cluster,
                   "per_match": per_match, "all_matches": full, "accepted": accepted}, fh, indent=2)
        fh.write("\n")
    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
