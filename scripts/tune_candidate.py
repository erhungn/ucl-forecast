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
* ``squad_value`` - the usual x3-weighted pooled log-loss.
* ``gap`` - the usual x3-weighted pooled log-loss, measured against **model
  v1** rather than against Elo, because v1 is what is accepted. Pairing a
  candidate that contains the squad-value term against Elo alone would credit
  it with that term's gain as well as its own.
* ``gap_minutes`` - the same, measured against **v1 + A**: it replaces the
  squad-value total with a minutes-weighted mean and changes nothing else, so
  the comparison isolates the weighting.
* ``euro_scale`` - **candidate C**, declared in full before it ran. Model v2
  plus a per-competition goal scale and extra home advantage for the UCL main
  phase and for everything else European; domestic is the baseline. Measured
  against **v2**.

  Primary: **European matches only, on the exact Poisson scoreline loss**
  -log P(hg, ag), judged by the **two-way** pigeonhole bootstrap. The scoreline
  rather than 1X2 because the candidate acts on the goal distribution and a 1X2
  triple is a coarse functional of it: at the effect size the raw rates imply,
  1X2 gives |z| under 0.9 however right the term is, while the scoreline carries
  19 times the KL divergence. Choosing 1X2 would have been choosing a test known
  in advance to be inconclusive.

  Guard, declared with the primary: the European 1X2 delta must **not** be
  positive at the point estimate. A term that fixes the goal scale while making
  the match-outcome forecast worse is rejected even if the scoreline clears.

  Two weaknesses, recorded in the archive rather than discovered later: the
  motivating evidence is the 288 league-phase matches of 2024-25 and 2025-26,
  which are two of the four seasons this chain scores, so a pass is a weaker
  confirmation than an independent holdout; and `a`, `h`, `b` and every beta are
  refitted alongside the new terms, so the domestic deltas are reported as a
  non-worsening check.

  Note on the objective: ``european_only`` also narrows
  ``pooled_objective_covariate``, and CLAUDE.md warns against *tuning* on
  European log-loss. It does not bind here - this candidate has no tuned
  hyperparameter, the half-life is fixed at 365 days, so the training objective
  is reported and never optimised.

Every run reports three intervals - two-way pigeonhole, one-way cluster and
per-match - and ``RULE`` below says which one decides. The six archived
candidates keep the one-way rule they were declared under; a new candidate
declares ``"two_way"``, which has been the project's standing acceptance rule
since milestone 8 even though, until this change, no code path here computed it.

``--label`` marks a re-run whose only change is an implementation fix, so the
output says which run it is. For ``squad_value`` the covariate's coverage is
printed per fold, because a null at partial coverage is a weaker statement than
a null at full coverage and should read as such.
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
from ucl.significance import two_way_bootstrap

# Below ~61 days no league clears MIN_MATCHES, so the covariate is empty and the
# model silently collapses to the baseline. The grid starts where the covariate
# actually exists; the degenerate region is reported, not searched.
HALF_LIVES = [61.0, 91.0, 183.0, 365.0, 548.0, 730.0, 1096.0]
PRIMARY = {"market": "european", "market_elo": "european", "congestion": "pooled",
           "squad_value": "pooled", "gap": "pooled", "gap_minutes": "pooled",
           "euro_scale": "european"}
#: What each candidate is measured against. Absent means the Elo baseline,
#: which is what was accepted when the earlier candidates were tested.
BASELINE = {"gap": "squad_value", "gap_minutes": "gap", "euro_scale": "gap"}
#: market_elo is candidate 1b, declared after candidate 1 failed: the same
#: market strengths mapped into Elo units per league instead of standardised.
POST_HOC = {"market_elo"}
#: Which interval decides. Absent means the one-way cluster bootstrap over
#: (season, home club) - what the six archived candidates were judged by, and
#: what their decision files must keep reproducing to the last digit. Two-way
#: pigeonhole clustering is the standing rule from milestone 8 onward, so any
#: NEW candidate declares "two_way" here before it runs. The two-way interval
#: is now computed and reported for every candidate either way: it was declared
#: standing in milestone 9 but no code path produced it, so until now a
#: candidate could not actually be judged by the rule the project had adopted.
RULE: dict[str, str] = {"euro_scale": "two_way"}
#: Which loss the primary interval is computed on. Absent means the 1X2 log-loss
#: every earlier candidate used. "scoreline" is the exact Poisson scoreline loss
#: -log P(hg, ag): candidate C acts on the goal distribution, and a 1X2 triple is
#: a coarse functional of it - the KL divergence the candidate can deliver is 19
#: times larger in the scoreline likelihood, and on 1X2 the test would be
#: inconclusive at any plausible effect size. Declared before the run; the 1X2
#: delta is kept as a guard so a term that fixes goals while degrading the
#: match-outcome forecast is still rejected.
METRIC: dict[str, str] = {"euro_scale": "scoreline"}


def _two_way(frame, draws):
    """The standing acceptance interval from a paired-loss frame.

    ``two_way_bootstrap`` reads ``d``, ``weight``, ``home_cluster`` and
    ``away_cluster``; the paired frame carries the last three already and calls
    the difference ``difference``, so renaming it here is the whole adapter.
    """
    return two_way_bootstrap(frame.assign(d=frame.difference), draws=draws)


def folds():
    return [(BACKTEST_SEASONS[i], BACKTEST_SEASONS[:i])
            for i in range(1, len(BACKTEST_SEASONS))]


def _coverage_by_season(candidate: str, seasons: list[str]) -> dict:
    """Share of scored matches where the covariate is defined for both clubs."""
    from ucl.backtest import _evaluation_frames
    from ucl.config import SEASONS

    if candidate == "squad_value":
        from ucl.data import transfermarkt

        out = {}
        for season in seasons:
            values = transfermarkt.squad_values(
                transfermarkt.as_of_for_season(season))
            domestic, euro = _evaluation_frames(season)
            out[season] = {
                "european": transfermarkt.coverage(
                    values, euro.home_id.to_numpy(), euro.away_id.to_numpy()),
                "domestic": transfermarkt.coverage(
                    values, domestic.home_id.to_numpy(), domestic.away_id.to_numpy()),
                "clubs_valued": len(values),
            }
        return out

    if candidate == "euro_scale":
        from ucl.data import european as european_data

        out = {}
        for season in seasons:
            domestic, euro = _evaluation_frames(season)
            main = european_data.is_main_phase(euro)
            ucl_main = int((main & euro.competition.eq("UCL")).sum())
            out[season] = {
                # Every match carries a competition, so "coverage" here is not
                # about missing data: it is the share of the scored European
                # sample the UCL term can act on at all, which is what dilutes
                # the measured effect.
                "european": ucl_main / max(len(euro), 1),
                "domestic": 0.0,
                "clubs_rated": len(euro),
            }
        return out

    if candidate in ("gap", "gap_minutes"):
        import numpy as np

        from ucl.backtest import _fitted_gap
        from ucl.models import gap as gap_model

        weighting = "minutes" if candidate == "gap_minutes" else "total"
        out = {}
        for season in seasons:
            *_, residuals, _, _ = (None, *_fitted_gap(
                SEASONS[season].cutoff_date, season, weighting))
            domestic, euro = _evaluation_frames(season)
            rated = residuals["sot"]
            shares = {}
            for name, frame in (("european", euro), ("domestic", domestic)):
                difference = gap_model.match_difference(
                    rated, frame.home_id.to_numpy(), frame.away_id.to_numpy())
                shares[name] = float(np.mean(difference != 0.0))
            shares["clubs_rated"] = len(rated)
            out[season] = shares
        return out

    return {}


def _fitted_terms(candidate: str, season_label: str) -> dict:
    """The candidate's own new coefficients, per fold.

    Printed and archived because a delta without the coefficient behind it says
    nothing about why: a null from a term fitted at zero is a different finding
    from a null from a term fitted large.
    """
    if candidate != "euro_scale":
        return {}
    from ucl.backtest import _fitted_euro_scale
    from ucl.config import SEASONS

    model = _fitted_euro_scale(SEASONS[season_label].cutoff_date, season_label)[-1]
    out = {f"c_{k}": v for k, v in model.scale.items()}
    out.update({f"g_{k}": v for k, v in model.home_extra.items()})
    return out


def _rates_summary(frame, home, away, draw_column) -> dict:
    """Predicted draw rate, goals per match and P(over 2.5) for one model."""
    from ucl.models import scorelines

    return {
        "draw rate": float(frame[draw_column].mean()),
        "goals per match": float((frame[home] + frame[away]).mean()),
        "over 2.5": float(scorelines.probability_over(
            2.5, frame[home].to_numpy(), frame[away].to_numpy()).mean()),
    }


def _diagnostics(scope) -> dict:
    """The diagnosis restated as a measurement, before and after.

    These decide nothing. If they improve while the primary test fails, the
    candidate is still rejected and this table is what says so.
    """
    def block(frame):
        candidate = _rates_summary(frame, "rate_home", "rate_away", "p_draw")
        baseline = _rates_summary(frame, "base_rate_home", "base_rate_away",
                                  "base_p_draw")
        goals = frame.hg + frame.ag
        observed = {
            "draw rate": float((frame.outcome == 1).mean()),
            "goals per match": float(goals.mean()),
            "over 2.5": float((goals > 2.5).mean()),
        }
        return {name: {"baseline": baseline[name], "candidate": candidate[name],
                       "observed": observed[name]} for name in observed}

    return {
        "n": int(len(scope)),
        "pooled": block(scope),
        "folds": {fold: block(part) for fold, part in scope.groupby("fold")},
        "ucl_main_share": (float((scope.level == "ucl_main").mean())
                           if "level" in scope else None),
    }


def _domestic_check(pooled, draws, scoreline) -> dict:
    """Domestic deltas, because the shared terms are refitted alongside the new ones.

    `a`, `h`, `b` and every beta move when the competition intercepts are added,
    so domestic predictions change too. Binding on nothing, but a domestic
    degradation would mean the new terms are absorbing variance that belongs
    elsewhere.
    """
    domestic = pooled[pooled.competition == "domestic"]
    if not len(domestic):
        return {}
    out = {"scoreline" if scoreline else "outcome":
           bootstrap_difference(domestic, cluster=True, draws=draws)}
    if scoreline:
        out["outcome"] = bootstrap_difference(
            domestic.assign(difference=domestic.outcome_difference),
            cluster=True, draws=draws)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=sorted(PRIMARY))
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--label", default="",
                        help="free-text run label, printed and archived")
    args = parser.parse_args(argv)
    candidate = args.candidate
    primary = PRIMARY[candidate]
    european_only = primary == "european"
    metric = METRIC.get(candidate, "outcome")
    scoreline = metric == "scoreline"

    started = time.perf_counter()
    label = f"{candidate!r}" + (" [POST HOC - declared after candidate 1 failed]"
                                if candidate in POST_HOC else "")
    if args.label:
        label += f" [{args.label}]"
    baseline = BASELINE.get(candidate)
    print(f"=== candidate {label}: forward chaining, primary test = {primary} ===")
    print(f"  measured against: {baseline or 'the Elo baseline'}")
    print(f"  primary metric: {metric}"
          + ("  (-log P(exact score); the 1X2 delta is a non-worsening guard)"
             if scoreline else "  (1X2 log-loss)"))
    print(f"  decided by: {RULE.get(candidate, 'one_way_cluster')}")
    coverage = _coverage_by_season(candidate, [test for test, _ in folds()])
    if coverage:
        print("  covariate coverage (both clubs known), by test season:")
        for season, share in coverage.items():
            extra = share.get("clubs_valued") or share.get("clubs_rated")
            print(f"    {season}: European {share['european']:.1%}, "
                  f"domestic {share['domestic']:.1%}  ({extra} clubs)")
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

        frame = paired_losses_covariate(test, candidate, half_life_days=half_life,
                                        baseline=baseline, scoreline=scoreline)
        frame["fold"] = test
        if scoreline:
            # The primary statistic travels in `difference` so every bootstrap
            # and every print below is metric-agnostic; the 1X2 delta is kept
            # under its own name as the declared guard.
            frame["outcome_difference"] = frame.difference
            frame["difference"] = frame.scoreline_difference
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
        terms = _fitted_terms(candidate, test)
        if terms:
            rows[-1].update(terms)
            print("    fitted terms: " + "  ".join(
                f"{k} {v:+.5f}" for k, v in terms.items()))

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
        two_way = _two_way(fold, args.draws)
        print(f"  {row.fold}: n={cluster['n']:5d} delta {row.delta_primary:+.5f} "
              f"two-way [{two_way['low']:+.5f}, {two_way['high']:+.5f}] "
              f"cluster [{cluster['low']:+.5f}, {cluster['high']:+.5f}] "
              f"per-match [{per_match['low']:+.5f}, {per_match['high']:+.5f}]")

    cluster = bootstrap_difference(scope, cluster=True, draws=args.draws)
    per_match = bootstrap_difference(scope, cluster=False, draws=args.draws)
    two_way = _two_way(scope, args.draws)
    rule = RULE.get(candidate, "one_way_cluster")
    print(f"\n=== POOLED, primary test ({primary}), decided by {rule} ===")
    print(f"  matches {cluster['n']}, clusters {cluster['clusters']}, "
          f"home clusters {two_way['home_clusters']}, "
          f"away clusters {two_way['away_clusters']}")
    print(f"  weighted mean delta log-loss: {cluster['difference']:+.5f} "
          f"(negative = the candidate is better)")
    print(f"  two-way   95% CI [{two_way['low']:+.5f}, {two_way['high']:+.5f}]  "
          f"excludes zero: {two_way['excludes_zero']}")
    print(f"  cluster   95% CI [{cluster['low']:+.5f}, {cluster['high']:+.5f}]  "
          f"excludes zero: {cluster['excludes_zero']}")
    print(f"  per-match 95% CI [{per_match['low']:+.5f}, {per_match['high']:+.5f}]  "
          f"excludes zero: {per_match['excludes_zero']}")
    guard = None
    if scoreline:
        outcome_scope = scope.assign(difference=scope.outcome_difference)
        guard = bootstrap_difference(outcome_scope, cluster=True, draws=args.draws)
        guard_two_way = _two_way(outcome_scope, args.draws)
        print(f"\n  GUARD, 1X2 log-loss on the same matches: "
              f"{guard['difference']:+.5f}")
        print(f"    two-way 95% CI [{guard_two_way['low']:+.5f}, "
              f"{guard_two_way['high']:+.5f}]")
        print(f"    declared condition: the point estimate must not be positive "
              f"-> {'PASS' if guard['difference'] <= 0 else 'FAIL'}")
    if rule == "two_way":
        accepted = bool(two_way["excludes_zero"] and two_way["mean"] < 0)
    else:
        accepted = bool(cluster["excludes_zero"] and cluster["difference"] < 0)
    if guard is not None:
        accepted = bool(accepted and guard["difference"] <= 0)
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

    diagnostics = _diagnostics(scope) if scoreline else {}
    if diagnostics:
        print("\n=== calibration diagnostics, European matches, before and after ===")
        print(f"  {'':22} {'baseline':>10} {'candidate':>10} {'observed':>10}")
        for name, row in diagnostics["pooled"].items():
            print(f"  {name:22} {row['baseline']:10.4f} {row['candidate']:10.4f} "
                  f"{row['observed']:10.4f}")
        print("\n  per fold (candidate vs observed):")
        for fold, rows in diagnostics["folds"].items():
            print(f"    {fold}: draws {rows['draw rate']['candidate']:.3f}/"
                  f"{rows['draw rate']['observed']:.3f}  "
                  f"goals {rows['goals per match']['candidate']:.2f}/"
                  f"{rows['goals per match']['observed']:.2f}  "
                  f"over 2.5 {rows['over 2.5']['candidate']:.3f}/"
                  f"{rows['over 2.5']['observed']:.3f}")

    with open(OUTPUTS_DIR / f"candidate_{candidate}_decision.json", "w", encoding="utf-8") as fh:
        json.dump({"candidate": candidate, "post_hoc": candidate in POST_HOC,
                   "run_label": args.label, "baseline": baseline or "elo",
                   "coverage": coverage,
                   "primary": primary, "metric": metric, "rule": rule,
                   "two_way": two_way, "cluster": cluster,
                   "per_match": per_match, "all_matches": full,
                   "guard_1x2": guard,
                   "guard_1x2_two_way": guard_two_way if scoreline else None,
                   "diagnostics": diagnostics,
                   "domestic_check": _domestic_check(pooled, args.draws, scoreline),
                   "motivating_evidence_overlaps_test_seasons":
                       candidate == "euro_scale",
                   "accepted": accepted}, fh, indent=2)
        fh.write("\n")
    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
