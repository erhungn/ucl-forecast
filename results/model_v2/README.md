# Model v2

The current forecasting model: **Elo with association offsets, the pre-season
squad-value prior, and the three GAP shot-rating residuals.** It supersedes
`model_v1`, which stays as the reference v2 was measured against.

- Tag: `model-v2`
- Archived: 2026-10-04
- Simulations: 50,000 iterations, seed 20260907

## The model

    d       = (elo_home - elo_away) / 400
    d_value = log(value_home) - log(value_away)
    d_gap_m = (S_h_hat - S_a_hat) from Elo-residual GAP ratings, per measure m

    log(lambda_home) = a + h + b*d + b_value*d_value + sum_m b_m * d_gap_m

| coefficient | value |
|---|---|
| `a` | 0.16951 |
| `h` | 0.23373 |
| `b` | 0.69818 |
| `b_d_value` | +0.04004 |
| `b_d_gap_sot` | **+0.02331** |
| `b_d_gap_shots_off` | -0.00342 |
| `b_d_gap_corners` | -0.00319 |

Fitted on 65,389 pre-cutoff matches at pre-match ratings. Elo is unchanged from
`elo-baseline-v0`; the Elo update inside the simulation still uses Elo alone, so
every covariate shifts the goal rates and none of them touches the rating
dynamics.

## Why it was promoted

The forward chain accepted the shot ratings as candidate A, measured against v1
rather than against Elo: pooled delta **-0.00042**, cluster CI
**[-0.00076, -0.00008]**, per-match [-0.00074, -0.00010], **all four folds
negative**, stable across bootstrap draws and seeds. See
`candidate_gap_decision.json` and `../candidate_gap/README.md`.

Candidate B (minutes-weighted squad value, on top of A) was rejected: pooled
-0.00001, CI [-0.00047, +0.00044]. See `../candidate_gap_minutes/`.

**This season cannot confirm or overturn that.** The paired standard error
between v1 and v2 after matchday 1 is 0.004, and over all 144 league-phase
matches it will be roughly 0.0014 - still about three times the effect the
backtest resolved. `live/<season>/model_tracking.csv` follows both models from
the same frozen information set as monitoring, not as a gate.

## The gate the promotion had to pass

The covariate model was generalised from one covariate to any number
(`ucl.models.covariates.ForecastModel`, assembled in `ucl.models.forecast`).
Configured as v1, it must reproduce the frozen v1 forecast exactly, or a v1-v2
comparison would be measuring the refactor:

    diff outputs/gate_v1_50k.csv results/model_v1/forecast_2026-27.csv
    -> bit-identical across all 14 columns, 50,000 iterations, same seed

`gate_v1_reproduction.log` is that run. The three archived candidate results
(squad value, A, B) also reproduce to the last digit through the refactored
path.

## v2 against v1 and against both markets

`v1_vs_v2_market.csv` holds every club. The ten largest P(champion) moves:

| club | v1 | v2 | move | exchange | bookmaker de-vigged | v1/exch | v2/exch |
|---|---|---|---|---|---|---|---|
| Bayern Munich | 0.180 | 0.200 | **+0.020** | 0.125 | 0.109 | 1.44 | **1.60** |
| Arsenal | 0.229 | 0.220 | -0.010 | 0.125 | 0.109 | 1.83 | **1.76** |
| Real Madrid | 0.065 | 0.073 | +0.008 | 0.125 | 0.096 | 0.52 | **0.58** |
| Aston Villa | 0.027 | 0.022 | -0.006 | 0.015 | 0.022 | 1.82 | 1.45 |
| Barcelona | 0.109 | 0.104 | -0.005 | 0.155 | 0.102 | 0.70 | 0.67 |
| Paris Saint-Germain | 0.066 | 0.061 | -0.005 | 0.165 | 0.127 | 0.40 | 0.37 |
| Borussia Dortmund | 0.021 | 0.017 | -0.004 | 0.015 | 0.019 | 1.39 | 1.11 |
| Manchester United | 0.027 | 0.030 | +0.004 | 0.045 | 0.045 | 0.59 | **0.68** |
| Sporting CP | 0.019 | 0.021 | +0.003 | - | 0.008 | - | - |
| PSV Eindhoven | 0.011 | 0.008 | -0.002 | - | 0.006 | - | - |

The four clubs that prompted this, with the shot residual that moved them:

| club | P(champion) | P(top 8) | vs exchange | vs bookmaker de-vigged | shots on target vs Elo |
|---|---|---|---|---|---|
| **Arsenal** | 0.229 -> 0.220 | 0.790 -> 0.786 | 1.83x -> **1.76x** | 2.10x -> **2.01x** | **-1.52** |
| **Porto** | 0.015 -> 0.013 | 0.291 -> 0.286 | not priced | 1.96x -> **1.70x** | **-1.18** |
| **Real Madrid** | 0.065 -> 0.073 | 0.465 -> 0.483 | 0.52x -> **0.58x** | 0.67x -> **0.76x** | **+1.18** |
| **Manchester United** | 0.027 -> 0.030 | 0.297 -> 0.302 | 0.59x -> **0.68x** | 0.59x -> **0.68x** | **+1.58** |

All four move **towards** the market, each in the direction its shot residual
implies - the two the model over-rated come down, the two it under-rated go up.
That is the mechanism doing what the diagnostic said it would.

**And the largest move of all goes the other way.** Bayern Munich out-shoots
its Elo by +1.09 shots on target, so v2 raises it from 0.180 to 0.200 and the
gap to the exchange *widens*, 1.44x to 1.60x. The shot ratings genuinely
disagree with the market about Bayern. Reporting only the four agreeable cases
would misrepresent what this model does.

In aggregate the market gap barely moves: mean |model - market| goes from
0.0187 to 0.0183 against the bookmaker's de-vigged prices (27 clubs) and from
0.0402 to 0.0399 against the exchange's raw prices (13 clubs). **v2 is not a
fix for limitation 1.** It is a small, independently-measured improvement in
out-of-sample log-loss that happens to move four named clubs the right way.

## How strong is this result? Restated after the significance analysis

Measured in `results/significance/` after the fact, with the per-match
out-of-sample losses reproducing this delta exactly:

| resampling | 95% CI for v2 - v1 | excludes zero |
|---|---|---|
| weekly blocks (155) | [-0.000736, -0.000123] | yes |
| one-way clusters, (season, home club) (1,309) | [-0.000765, -0.000086] | yes |
| **two-way, (season, home club) x (season, away club)** | **[-0.001006, +0.000163]** | **no** |

**This acceptance was made under the one-way rule, which was pre-declared.
Under two-way clustering the interval includes zero.** The one-way scheme
treats a club's away matches as independent of its home matches, which the
frozen ratings make false: the same rating error travels with the club at both
ends of a fixture. The two-way interval is the honest one, and it is about 40%
wider.

The point estimate does not move. It favours v2 in **all four folds**
(-0.000712, -0.000200, -0.000383, -0.000406), a one-sided sign test p of 1/16 =
0.0625. A Diebold-Mariano test on daily means does not reject (p = 0.50),
though it tests a smaller quantity with much less precision.

**The honest summary: small, consistent in sign, not robustly significant.**
The decision not to revisit it is deliberate - the rule was fixed before the
result - and v2 stays published because the point estimates favour it and the
cost of being wrong either way is small. **Two-way clustering is the acceptance
rule for any future candidate.**

## Coverage

| covariate | clubs | both clubs, of the 144 fixtures |
|---|---|---|
| `d_value` | 32/36 | 79.2% |
| `d_gap_sot` / `shots_off` / `corners` | 29/36 | 64.6% |

Squad value misses LASK, Sabah, Slovan Bratislava and Viking; the shot ratings
also miss Bodo/Glimt, Shakhtar Donetsk and Slavia Praha, whose leagues carry no
shot data at all. An uncovered club falls back on the rest of the model, which
is the convention every candidate was scored under.

## Verification

All ten simulator checks pass; 0 unresolved league-phase ties in 50,000
iterations. `simulate_md0.log` and `simulate_md1.log` are the runs.

## Files

`forecast_2026-27.csv`, `forecast_2026-27_md1.csv`, `forecast_2026-27_v1.csv`
(the v1 forecast beside it), `model_2026-27.json`,
`gap_residuals_2026-27.csv`, `v1_vs_v2_market.csv`,
`candidate_gap_decision.json`, `candidate_gap_folds.csv`,
`gate_v1_reproduction.log`, `simulate_md0.log`, `simulate_md1.log`.
