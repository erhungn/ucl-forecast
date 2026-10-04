# Candidate B: minutes-weighted squad value - REJECTED (null)

Milestone 8, declared and run once as binding. Tested against **v1 + A** at
commit `3e4e0d0`: it replaces the squad-value total with a minutes-weighted
mean and changes nothing else, so the comparison isolates the weighting.

## The change

For each club, with `minutes_p` the minutes a player actually played inside the
same 180-day membership window and `value_p` his latest valuation strictly
before the as-of date:

    v_club = sum_p (minutes_p / sum_q minutes_q) * value_p

A minutes-weighted **mean**, not a per-90 total: a total would scale with how
many matches a club happened to play in the window, which says nothing about
its quality. The covariate stays `log v_home - log v_away`, so the scaling
cancels anyway. The frozen as-of rule is unchanged - 6 July of the season's
starting year.

The idea it tests is a real weakness of v1: a squad total counts a bench of
expensive reserves exactly like a first eleven, and a minutes-weighted mean
describes the side that actually takes the field.

## Primary test: x3-weighted pooled log-loss, frozen, cluster bootstrap

    19,188 matches in 1,309 clusters, delta -0.00001
      cluster   95% CI [-0.00047, +0.00044]  includes zero
      per-match 95% CI [-0.00039, +0.00037]  includes zero

    DECISION: KEEP the baseline (v1 + A)

As clean a null as the project has produced - the point estimate is one
hundred-thousandth of a nat, and the interval is all but symmetric about zero.
Stable across 4,000 / 20,000 / 50,000 draws.

The folds disagree in sign, which is what noise looks like: 2022-23 -0.00007,
2023-24 -0.00034, **2024-25 +0.00122** (significantly worse on its own),
**2025-26 -0.00091** (significantly better on its own). Two folds individually
clearing zero in opposite directions is a reminder of why the pre-declared test
is the pooled one.

## Sensitivities, which do not decide

| scope | delta | cluster 95% CI | n |
|---|---|---|---|
| **pooled (primary)** | **-0.00001** | **[-0.00047, +0.00044]** | 19,188 |
| domestic only | -0.00036 | [-0.00087, +0.00015] | 17,356 |
| European only | **+0.00107** | **[+0.00002, +0.00214]** | 1,832 |

Worth noting plainly: on European matches the minutes-weighted version is
*significantly worse*. That is a sensitivity and not the decision, but it
points the same way as the verdict rather than against it.

## Why it changes so little

The two measures rank the clubs almost identically - **Spearman 0.9894** over
the 2026-27 league phase. A club that can afford an expensive squad also gives
those players the minutes, so weighting by minutes mostly reproduces the total.
The largest rank movements in the whole league phase are three places:

| club | total | minutes-weighted |
|---|---|---|
| Atletico Madrid | #10 (578.1m) | #13 (25.8m/player) |
| Borussia Dortmund | #13 (490.0m) | #10 (28.2m/player) |
| Fenerbahce | #24 (257.2m) | #27 (12.0m/player) |

Coverage is **identical** to v1's - the same 291 to 318 clubs per fold, 43-47%
of European and ~70% of domestic matches - because membership is already
derived from an in-window appearance and every appearance carries minutes. So
this test changed the covariate's values and nothing else, which is what makes
the null interpretable: the weighting genuinely adds nothing, rather than
having been diluted by a coverage change.

## Fitted coefficients

| cutoff | | `b_d_value` | sot | shots off | corners |
|---|---|---|---|---|---|
| 2022-23 | total | +0.0484 | +0.0287 | -0.0045 | +0.0004 |
| 2022-23 | minutes | +0.0625 | +0.0312 | -0.0056 | -0.0003 |
| 2024-25 | total | +0.0403 | +0.0305 | -0.0068 | -0.0043 |
| 2024-25 | minutes | +0.0564 | +0.0305 | -0.0062 | -0.0057 |
| 2026-27 | total | +0.0400 | +0.0233 | -0.0034 | -0.0032 |
| 2026-27 | minutes | +0.0487 | +0.0232 | -0.0034 | -0.0022 |

The minutes-weighted coefficient is uniformly larger, which is arithmetic
rather than evidence: a per-player mean has a narrower spread than a squad
total, so the same effect needs a bigger coefficient. The GAP coefficients are
untouched, as they should be.

Files: `candidate_gap_minutes_folds.csv`, `candidate_gap_minutes_paired.csv`,
`candidate_gap_minutes_decision.json`, `run.log`.
