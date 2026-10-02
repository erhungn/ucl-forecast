# Elo baseline v0

The frozen reference the Dixon-Coles work (milestone 4) is measured against.

- Commit: `c1f1c48956a2eee9f87e931ae5a15ca9fb97b3c8`
- Tag: `elo-baseline-v0`
- Archived: 2026-10-02

## Frozen parameters

| parameter | value | note |
|---|---|---|
| `K_DOMESTIC` | 20.0 | tuning preferred lower, not significantly |
| `K_EUROPEAN` | 25.0 | likewise |
| `HOME_ADVANTAGE` | 65.0 | sweep span 0.0005 across 45-95, i.e. flat |
| `SEASON_CARRYOVER` | 0.90 | target: league mean |
| `GD_EXPONENT` | `None` | classic World Football Elo rule, **fixed by judgement** |

Match model: `log(lambda_home) = a + h + b*d`, `log(lambda_away) = a - b*d`,
with `d = (elo_home - elo_away)/400` and **no** Elo home advantage in `d` -
`h` carries home advantage, and counting both would double it. Fitted values
are in `poisson_2026-27.json`.

## Why these parameters and not tuned ones

Five-season forward chaining (2021-22 to 2025-26, each fold tuned on the seasons
before its test season, test season scored once) returned a null under the
pre-declared primary test: x3-weighted pooled log-loss, frozen mode, cluster
bootstrap over (season, home club).

    pooled: 19,188 matches in 1,309 clusters, delta -0.00040
      cluster   95% CI [-0.00126, +0.00049]  includes zero
      per-match 95% CI [-0.00114, +0.00032]  includes zero

Two findings carried forward:

- The goal-difference exponent is **not identified**. It pinned to 0.0 when
  tuned on 2024-25 alone and to the opposite bound 0.90 when tuned on four
  seasons, with K falling to 7-11 in the same runs: a low K with heavy margin
  weighting and a moderate K with light weighting are close to the same model.
  Hence fixing the classic rule by judgement rather than fitting it.
- Tuning on **pure European log-loss is actively harmful** out of sample: pooled
  delta +0.00285, CI [+0.00150, +0.00430]. The x3-weighted objective is right.

## Baseline performance

The market beats the model on domestic matches in all five seasons, by roughly
0.03 log-loss - see `backtest_summary.csv`. Walk-forward beats frozen by about
0.008 domestically, which is the cost of ratings going stale over a season.

## 2026-27 market gap

Mean absolute gap to de-vigged outright odds 0.0206 over the 27 priced clubs
(total 0.556). Largest disagreements: Arsenal +0.115 and Bayern +0.077 too high,
PSG -0.076, Real Madrid -0.045 and Liverpool -0.042 too low. The market is a
diagnostic and never enters any objective.

## Files

| file | contents |
|---|---|
| `forecast_2026-27.csv` | 2,000-iteration forecast with the market column |
| `backtest_summary.csv` | five seasons, model vs market, frozen vs walk-forward |
| `elo_2026-27.csv` | club ratings at the 2026-09-07 cutoff |
| `association_elo_2026-27.csv` | association means and fitted offsets |
| `poisson_2026-27.json` | fitted a, h, b |
| `forward_chain_*.csv`, `forward_chain_decision.json` | the tuning that was rejected |
