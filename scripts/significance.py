"""Is v2's improvement over v1 distinguishable from noise?

Analysis only: it re-scores the models the backtest already produced and writes
``results/significance/``. It fits nothing new, promotes nothing, and touches
neither ``live/`` nor the published coefficients.

The gate comes first. If the per-match losses do not reproduce the archived
deltas to the last digit, the export is wrong and the script stops rather than
reporting statistics about the wrong numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ucl import significance as sig
from ucl.config import BACKTEST_SEASONS, PROJECT_ROOT

OUT = PROJECT_ROOT / "results" / "significance"

#: Each comparison, and the archived delta it has to reproduce.
GATES = {
    ("v1", "elo"): "results/model_v1/candidate_squad_value_decision.json",
    ("v2", "v1"): "results/candidate_gap/candidate_gap_decision.json",
    ("v2+B", "v2"): "results/candidate_gap_minutes/candidate_gap_minutes_decision.json",
}
#: What the report compares, and against what.
COMPARISONS = [("v1", "elo"), ("v2", "v1"), ("v2+B", "v2"), ("v2+B", "v1"),
               ("v2_sot_only", "v1"), ("v2_sot_only", "v2")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args(argv)

    seasons = tuple(BACKTEST_SEASONS[1:])      # the four scored folds
    models = ("elo", "v1", "v2", "v2+B", "v2_sot_only")
    print(f"=== per-match out-of-sample losses: {len(models)} models, "
          f"{len(seasons)} folds ===")
    losses = sig.per_match_losses(seasons, models)
    matches = len(losses) // len(models)
    print(f"  {matches:,} matches x {len(models)} models = {len(losses):,} rows")

    print("\n=== gate: the archived deltas must reproduce exactly ===")
    failures = []
    for (model, reference), path in GATES.items():
        archived = json.load(open(PROJECT_ROOT / path))["cluster"]["difference"]
        differences = sig.paired_differences(losses, model, reference)
        got = sig.weighted_mean(differences.d, differences.weight)
        ok = abs(got - archived) < 1e-12
        print(f"  {model:12} vs {reference:4} {got:+.9f} | archived "
              f"{archived:+.9f} | {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{model} vs {reference}")
    if failures:
        raise SystemExit(
            f"the exported losses do not reproduce {', '.join(failures)}; "
            f"refusing to report statistics about numbers that do not match "
            f"the archive")

    OUT.mkdir(parents=True, exist_ok=True)
    losses.to_csv(OUT / "per_match_losses.csv", index=False,
                  float_format="%.9f")
    print(f"\nwritten to {(OUT / 'per_match_losses.csv').relative_to(PROJECT_ROOT)}")

    rows = []
    for model, reference in COMPARISONS:
        differences = sig.paired_differences(losses, model, reference)
        block = sig.block_bootstrap(differences, draws=args.draws, seed=args.seed)
        cluster = sig.cluster_bootstrap(differences, draws=args.draws,
                                        seed=args.seed)
        two_way = sig.two_way_bootstrap(differences, draws=args.draws,
                                        seed=args.seed)
        unweighted = sig.block_bootstrap(differences, draws=args.draws,
                                         seed=args.seed, weighted=False)
        daily = sig.diebold_mariano(differences, daily=True)
        raw = sig.diebold_mariano(differences, daily=False)
        rows.append({
            "model": model, "reference": reference, "n": block["n"],
            "blocks": block["blocks"], "clusters": cluster["blocks"],
            "mean_d": block["mean"], "low": block["low"], "high": block["high"],
            "standard_error": block["standard_error"],
            "share_below_zero": block["share_below_zero"],
            "excludes_zero": block["excludes_zero"],
            "cluster_low": cluster["low"], "cluster_high": cluster["high"],
            "cluster_excludes_zero": cluster["excludes_zero"],
            "two_way_low": two_way["low"], "two_way_high": two_way["high"],
            "two_way_excludes_zero": two_way["excludes_zero"],
            "two_way_standard_error": two_way["standard_error"],
            "two_way_share_below_zero": two_way["share_below_zero"],
            "home_clusters": two_way["home_clusters"],
            "away_clusters": two_way["away_clusters"],
            "unweighted_mean_d": unweighted["mean"],
            "unweighted_low": unweighted["low"],
            "unweighted_high": unweighted["high"],
            "dm_observations": daily["observations"], "dm_lags": daily["lags"],
            "dm_statistic": daily["statistic"], "dm_p_value": daily["p_value"],
            "dm_raw_statistic": raw["statistic"], "dm_raw_lags": raw["lags"],
            "dm_raw_p_value": raw["p_value"],
        })
        print(f"\n=== {model} vs {reference} ===")
        print(f"  mean d {block['mean']:+.6f}  95% CI [{block['low']:+.6f}, "
              f"{block['high']:+.6f}]  SE {block['standard_error']:.6f}")
        print(f"  blocks {block['blocks']} weeks, P(mean d < 0) = "
              f"{block['share_below_zero']:.3f}, excludes zero "
              f"{block['excludes_zero']}")
        print(f"  cluster bootstrap [{cluster['low']:+.6f}, {cluster['high']:+.6f}]"
              f"  excludes zero {cluster['excludes_zero']} "
              f"({cluster['blocks']} clusters)")
        print(f"  two-way pigeonhole [{two_way['low']:+.6f}, {two_way['high']:+.6f}]"
              f"  excludes zero {two_way['excludes_zero']} "
              f"({two_way['home_clusters']} home x {two_way['away_clusters']} away)")
        print(f"  unweighted mean d {unweighted['mean']:+.6f} "
              f"[{unweighted['low']:+.6f}, {unweighted['high']:+.6f}]")
        print(f"  Diebold-Mariano, daily means: t = {daily['statistic']:+.3f}, "
              f"p = {daily['p_value']:.4f} ({daily['observations']} days, "
              f"{daily['lags']} lags)")
        print(f"  Diebold-Mariano, per match:  t = {raw['statistic']:+.3f}, "
              f"p = {raw['p_value']:.4f} ({raw['observations']} matches, "
              f"{raw['lags']} lags)")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "comparisons.csv", index=False, float_format="%.9f")
    print(f"\nwritten to {(OUT / 'comparisons.csv').relative_to(PROJECT_ROOT)}")

    # --- per season, for the two comparisons that matter ---------------------
    seasonal = []
    for model, reference in (("v2", "v1"), ("v2_sot_only", "v2")):
        frame = sig.per_season(sig.paired_differences(losses, model, reference))
        frame.insert(0, "comparison", f"{model} vs {reference}")
        seasonal.append(frame)
        print(f"\n=== {model} vs {reference}, per season ===")
        print(frame.to_string(index=False, float_format=lambda v: f"{v:+.6f}"))
    seasonal = pd.concat(seasonal, ignore_index=True)
    seasonal.to_csv(OUT / "per_season.csv", index=False, float_format="%.9f")

    # --- standardised coefficients ------------------------------------------
    print("\n=== standardised coefficients (coefficient x covariate SD) ===")
    standardised = sig.standardised_coefficients(seasons + ("2026-27",))
    print(standardised.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    standardised.to_csv(OUT / "standardised_coefficients.csv", index=False,
                        float_format="%.9f")
    print(f"\nwritten to {(OUT / 'per_season.csv').name}, "
          f"{(OUT / 'standardised_coefficients.csv').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
