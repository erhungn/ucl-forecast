# Model v1

The accepted forecasting model: **Elo with association offsets, plus a
pre-season squad-value prior.** It supersedes `elo_baseline_v0`, which stays as
the reference the covariate was measured against.

- Tag: `model-v1`
- Archived: 2026-10-02
- Simulations: 50,000 iterations, seed 20260907

## The model

    d       = (elo_home - elo_away) / 400
    d_value = log(value_home) - log(value_away)          # 0 unless both valued

    log(lambda_home) = a + h + b * d + b_value * d_value
    log(lambda_away) = a     - b * d - b_value * d_value

| coefficient | value |
|---|---|
| `a` | 0.16947 |
| `h` | 0.23396 |
| `b` | 0.69698 |
| `b_value` | **0.03987** |

Fitted on 65,389 pre-cutoff matches at pre-match ratings. Elo is unchanged from
`elo-baseline-v0` - same K 20/25, home advantage 65, carryover 0.90 toward the
league mean, classic goal-difference rule - and the Elo update inside the
simulation still uses Elo alone. Squad value shifts the goal rates, never the
rating dynamics.

Squad value is a **pre-season prior**: the snapshot is 6 July of the season's
starting year, the staleness the frozen source imposes. For 2026-27 that is
2026-07-06, and it cannot be refreshed - the source has been paused since then.
Every forecast prints it in its information set, where `used_to` stays
2026-07-06 while the result sources advance.

## Why it was accepted

Five-season forward chaining, pre-declared primary test (x3-weighted pooled
log-loss, frozen, cluster bootstrap over (season, home club)):

    19,188 matches in 1,309 clusters, delta -0.00099
      cluster   95% CI [-0.00187, -0.00008]  excludes zero
      per-match 95% CI [-0.00176, -0.00026]  excludes zero

The fifth covariate tried and the first to clear the bar, after four nulls
(Dixon-Coles, market-implied ratings twice, fixture congestion). Two
implementation defects had to be fixed first, both found by looking at the
feature rather than at the test statistic:

1. **Club-name mapping** - 110 of 796 Transfermarkt clubs matched, so three
   quarters of European matches had no covariate. Rebuilt deterministically to
   439 clubs (`scripts/build_transfermarkt_map.py`); European coverage 22-25%
   -> 48-57%. That run is `../candidate_squad_value_coverage_fix/`.
2. **National-team appearances** - squad membership took a player's last
   appearance in the 180-day window, and `appearances.csv` also holds
   national-team matches. The 2026 World Cup ran 11-28 June 2026, so at the
   2026-27 as-of 16.3% of players were assigned to their country and dropped
   out of their club's value: PSG read 406m instead of 1,362m, Manchester City
   251m instead of 1,307m, Bayern 193m instead of 962m, Porto above Bayern.

Only the World Cup and the 2025-26 Africa Cup of Nations have appearance rows
in this dataset - Euro 2021, Euro 2024 and the Copa América have none - which
is why the earlier as-of dates were almost untouched (0.4-1.1% of players, rank
correlation 1.0000 between the two constructions, largest club value moving
4%). The **membership-fix verification** re-run therefore reproduces the
decision exactly: pooled delta -0.00099, cluster CI [-0.00187, -0.00008],
per-match CI [-0.00176, -0.00026], ACCEPT. Only two per-fold numbers move, in
the fifth decimal. Log: `membership_fix_verification.log`.

## Coverage, stated plainly

**Accepted at about half coverage.** Across the five backtest cutoffs 48-57% of
European matches had both clubs valued; the rest fell back to Elo, and the
measured effect is diluted accordingly. Within the 2026-27 league phase coverage
is higher - **32 of 36 clubs valued, both clubs valued in 79.2% of the 144
fixtures.** Unvalued: **LASK** and **Viking** (in the club table, but with no
appearance inside the membership window - the dataset follows a club only while
it plays a competition it covers), **Sabah** and **Slovan Bratislava** (not in
the Transfermarkt club table at all).

## 2026-27 forecast, v1 beside the baseline

Both at 50,000 iterations and the same seed, so the differences are the model's
and not Monte Carlo noise. `forecast_2026-27.csv` is v1;
`forecast_2026-27_elo_n50000.csv` is `elo-baseline-v0` re-run at the matching
iteration count. The archived `elo_baseline_v0/forecast_2026-27.csv` (2,000
iterations) reproduces bit-identically from this code with `--model elo
-n 2000`, so the refactor changed nothing about the baseline.

| club | P(top 8) v1 | elo | P(champion) v1 | elo | market |
|---|---|---|---|---|---|
| Arsenal | 0.790 | 0.776 | 0.229 | 0.226 | 0.109 |
| Bayern Munich | 0.736 | 0.724 | 0.180 | 0.185 | 0.109 |
| Barcelona | 0.581 | 0.572 | 0.109 | 0.103 | 0.102 |
| Manchester City | 0.497 | 0.460 | 0.089 | 0.075 | 0.090 |
| Paris Saint-Germain | 0.434 | 0.407 | 0.066 | 0.058 | 0.127 |
| Real Madrid | 0.465 | 0.419 | 0.065 | 0.053 | 0.096 |
| Inter | 0.516 | 0.521 | 0.047 | 0.052 | 0.029 |
| Liverpool | 0.380 | 0.357 | 0.028 | 0.027 | 0.076 |

Largest P(champion) moves: Manchester City +0.015, Real Madrid +0.011, PSG
+0.008, Bodo/Glimt -0.005, Barcelona +0.005, Bayern -0.005, Inter -0.005, PSV
-0.004. Largest P(top 8) moves: Bodo/Glimt -0.057, Real Madrid +0.045,
Manchester City +0.037, PSG +0.026, PSV -0.026, Liverpool +0.022, AEK Athens
-0.017, Sporting CP -0.015.

That is the direction the market diagnostic pointed at in milestone 3.5: the
elite clubs the baseline deflated move up, the domestic dominators it inflated
move down. Mean |model - market| over the 27 priced clubs falls from 0.0205 to
0.0187. **The market is a diagnostic and never entered an objective**, so this
is corroboration, not evidence - and the squad-value covariate is itself partly
a market consensus about quality, so some agreement is to be expected.

## Matchday 1

`forecast_2026-27_md1.csv`, information set as of 2026-10-12, matchday 1 fixed
at its actual scores (18 matches), 126 fixtures simulated. Largest moves since
matchday 0: Manchester City P(top 8) 0.497 -> 0.677, Inter 0.516 -> 0.352, Real
Madrid 0.465 -> 0.600, Porto 0.291 -> 0.160, PSV 0.223 -> 0.099.

`matchday_1_evaluation_2026-27.csv` scores the frozen pre-tournament
probabilities on those 18 matches: log-loss **0.6913**, RPS **0.1367**, against
a uniform baseline of 1.0986 / 0.2593. The baseline model gives 0.6992 /
0.1395. The standard error on the model's log-loss is 0.090 over 18 matches,
roughly a hundred times the differences the backtest was built to resolve:
**this is anecdote and distinguishes nothing.**

## The weekly update

| source | refresh | cadence |
|---|---|---|
| UCL results | Wikipedia league-phase page, `--refresh` | after each matchday |
| domestic results | manual Colab fetch into `data/raw/football_data_current/` | weekly |
| **squad value** | **none available - source paused since 2026-07-06** | **frozen** |

The squad-value term is a pre-season prior for the whole season by necessity. If
the Kaggle dataset resumes, the as-of date can advance; until then every
forecast carries the same 2026-07-06 snapshot, and says so in its information
set.

## Files

`forecast_2026-27.csv`, `forecast_2026-27_md1.csv`,
`forecast_2026-27_elo_n50000.csv`, `matchday_1_evaluation_2026-27.csv`,
`model_2026-27.json`, `squad_values_2026-27.csv`,
`candidate_squad_value_{folds,paired}.csv`,
`candidate_squad_value_decision.json`, `membership_fix_verification.log`,
`simulate_md0.log`, `simulate_md1.log`.
