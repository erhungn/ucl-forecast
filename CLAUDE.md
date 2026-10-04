# UCL 2026-27 Forecasting Model

## Goal
Pre-tournament forecast of the 2026-27 UEFA Champions League from a frozen
information set. Cutoff: 2026-09-07 (the day before Matchday 1).

Outputs:
- Match level: home/draw/away probabilities and scoreline distribution for all
  144 league-phase fixtures.
- Team level: P(top 8), P(9-24), P(25-36 eliminated), P(reach R16), P(QF),
  P(SF), P(final), P(champion).

## Hard rules
- Everything in English: code, identifiers, comments, log/print messages,
  file names, output headers.
- No look-ahead: a single `CUTOFF_DATE` in `config.py`; every loader asserts
  that no row used for fitting is dated on or after the cutoff.
- Reproducible: fixed random seeds, pinned dependencies (`pyproject.toml`).
- Python 3.11+, numpy / pandas / scipy. PyMC or cmdstanpy only if the
  hierarchical Bayesian variant is built.
- Tournament rules (tiebreakers, bracket) must have pytest coverage before the
  simulator is trusted.

## Competition format (verify against UEFA regulations before coding)
- 36 teams, 8 league-phase matches each (4 home, 4 away), single table.
- 1-8: direct to Round of 16. 9-24: two-legged knockout play-off.
  25-36: eliminated from all European competition.
- League-phase tiebreakers, per Article 18 of the 2026/27 regulations, in
  order (ten criteria, verified verbatim against the official PDF):
  a. superior goal difference in the league phase
  b. higher number of goals scored
  c. higher number of away goals scored
  d. higher number of wins
  e. higher number of away wins
  f. higher number of points obtained collectively by league phase opponents
  g. superior collective goal difference of those opponents
  h. higher number of goals scored collectively by those opponents
  i. lower disciplinary points total (red card = 3, yellow card = 1,
     expulsion for two yellow cards in one match = 3)
  j. higher club coefficient (Annex D)
  Criteria i and j are not simulatable from scorelines: i needs cards, which we
  do not model, and j needs the Annex D coefficient. A tie surviving h must be
  reported as unresolved, never silently resolved.
- Knockout bracket is fixed by league position with constrained draws.
  Implement it exactly per the official regulations; where UEFA uses a draw,
  randomize inside the simulation within the same constraints.
- Home advantage in two-legged ties, per Article 19.02/19.03/19.04: the side
  with the lower seeding position hosts the second leg, in every round. That
  covers "the seeded clubs play the return leg matches at home" in the play-offs
  and round of 16, teams ranked 1-4 at home in the quarter-final return leg and
  1-2 in the semi-final return leg. If a seeded team is eliminated, the team that
  beat it inherits its seeding position for the rest of the bracket.
- Two-legged ties: aggregate score, then extra time, then penalties
  (no away-goals rule).
- Final: single match, 2027-06-05, Riyadh Air Metropolitano (Madrid), neutral
  venue (flag: Atletico Madrid's home ground).

## Model
1. Ratings: **our own Elo** replayed from all domestic and European results
   since 2012, with association-level strength offsets fitted on pre-cutoff
   European matches for the cross-league scale. (ClubElo was unreachable;
   `data/clubelo.py` remains an optional cross-check.)
2. Goal model, **as built (model v2)**: independent Poisson on the Elo
   difference, a pre-season squad-value term and three shot-rating terms,
   fitted jointly by MLE on pre-match ratings:
   `log(lambda_home) = a + h + b*d + b_value*d_value + sum_m b_m*d_gap_m`,
   `d = (elo_h - elo_a)/400`, `d_value = log(value_h) - log(value_a)`,
   `d_gap_m` the Elo-residual GAP difference per attacking measure. Any
   covariate is zero unless both clubs are covered.
   Dixon-Coles (tau, ridge-shrunk att/def, time decay) was built and tested
   against this and returned a null; the code stays in `models/dixon_coles.py`
   for later. xG is blocked on data: no maintained free source covers the
   backtest window.
3. Simulation: Monte Carlo, default N = 50_000.
   - "Hot" simulation: update Elo inside each iteration after every simulated
     match.
   - Extra time: scale lambdas by 1/3.
   - Penalties: ~50/50 with a small rating tilt (parameterized).

## Validation
- Run the identical pipeline on 2024-25 and 2025-26 with their own cutoffs.
- Match-level metrics: RPS, log-loss, Brier, calibration plot.
- Benchmark: de-vigged bookmaker closing odds (football-data.co.uk).
- Team-level: compare stage probabilities with realized outcomes.

## Data sources
- Match results: football-data.co.uk (domestic leagues), UCL results
- Elo: ClubElo API (http://api.clubelo.com)
- xG (optional): Understat / FBref
- 2026-27 fixtures: official league-phase schedule (store as CSV in data/raw)

## Project layout
```
src/ucl/
  config.py
  data/          # loaders, team-name mapping, cutoff checks
  models/        # elo.py, dixon_coles.py
  sim/           # league_phase.py, tiebreakers.py, knockout.py, bracket.py
  evaluate.py    # RPS, log-loss, calibration
scripts/         # fetch_data.py, fit.py, simulate.py, backtest.py
tests/
data/raw/  data/processed/
outputs/         # csv tables + figures
```

## Milestones
1. Data pipeline + team-name mapping across sources + cutoff assertions - DONE
2. Elo baseline model and simulator end-to-end (no Dixon-Coles yet) - DONE
3. Tiebreaker and bracket tests passing - DONE (done with milestone 2)
3.5 Backtest harness and Elo tuning - DONE. Reordered ahead of Dixon-Coles:
   the simulator was sound and the ratings were the weak part, so tuning them
   first gives milestone 4 a fixed reference. Frozen as `elo-baseline-v0`.
4. Dixon-Coles with Elo covariate, fitted and compared to the Elo baseline -
   DONE, and a null; the Elo baseline stayed the forecast model
5. Backtests: forward chaining over 2021-22 to 2025-26 (was 2024-25 and
   2025-26 only; extended once history reached 2016-17) - DONE
6. Frozen 2026-27 pre-tournament forecast and the "update after matchday k"
   mode, driven by an as-of date - DONE
7. New information sources, one at a time against the accepted model. Four
   nulls, then squad market value accepted - DONE. Frozen as `model-v1`.
8. A second outright benchmark, shot-based GAP ratings and minutes-weighted
   squad value - DONE. The benchmark showed favourite-longshot bias accounts
   for part of the market gap; GAP ratings accepted and promoted to
   `model-v2`; minutes-weighting rejected.

## What the backtests established (do not relitigate without new evidence)
- **Protocol.** Forward chaining: for each test season, tune on the seasons
  before it and score it once. The pre-declared primary test is x3-weighted
  pooled log-loss, frozen mode, with a **cluster bootstrap over (season, home
  club)** - a mis-rated club biases all its matches the same way, so per-match
  resampling understates uncertainty. Accept a change only if the pooled 95% CI
  excludes zero. Everything else is a sensitivity.
- **Elo tuning returned a null** (pooled delta -0.00040, CI [-0.00126,
  +0.00049]), so the defaults stand: K 20/25, home advantage 65, carryover 0.90
  toward the league mean.
- **The goal-difference rule is fixed by judgement at the classic World Football
  Elo weighting.** It is not identified by the data: the exponent pinned to 0.0
  tuned on one season and to 0.90 tuned on four, trading off against K.
- **Never tune on pure European log-loss**: out of sample it is significantly
  worse (pooled delta +0.00285, CI [+0.00150, +0.00430]). ~500 European matches
  a season is too thin; the domestic sample carries the statistical power.
- Home advantage is applied **once**, in the Elo update. The Poisson model uses
  the raw rating difference and carries home advantage in its own intercept.
- The market is a **diagnostic only** and never enters an objective.
- **Four covariate candidates failed** under this protocol: Dixon-Coles,
  market-implied ratings (both the standardised and the Elo-units mappings), and
  fixture congestion. Do not relitigate them without new evidence or a
  materially better data source.
- **Squad market value is accepted, and is model v1**: pooled delta
  **-0.00099, cluster CI [-0.00187, -0.00008]**, excluding zero and stable
  across bootstrap draws and seeds. `b_value` +0.0399 beside `b_elo` 0.697.
  The stopping rule declared before candidate 1b therefore does not fire.
  See `results/model_v1/`.
- It is **accepted at about half coverage** - 48-57% of European matches in the
  backtest had both clubs valued, the rest falling back to Elo - and it is a
  **pre-season prior**, frozen at 6 July, that cannot be refreshed while the
  source is paused. Both belong in any statement of what it delivers.
- **Shot-based GAP ratings are accepted and promoted: the current model is
  `model-v2`** (milestone 8, candidate A): pooled delta **-0.00042, cluster CI
  [-0.00076, -0.00008]**, measured against model v1, all four folds negative
  and stable across bootstrap draws and seeds. Wheatcroft's GAP ratings on
  shots on target, shots off target and corners, implemented from arXiv
  2001.09097. The signal is **shots on target** (`b` +0.023 to +0.031 across
  every cutoff); shots off target is slightly negative and corners carry
  nothing. See `results/model_v2/` and `results/candidate_gap/`.
- **The covariate model carries any number of covariates**
  (`ForecastModel`, assembled in `models/forecast.py`). The gate on every
  generalisation of it: configured as v1 it must reproduce
  `results/model_v1/forecast_2026-27.csv` **bit-identically** at 50,000
  iterations, and the three archived candidate deltas must reproduce to the
  last digit. Otherwise a v1-v2 comparison measures the refactor.
- **v2 moves four named clubs towards the market and one away.** Arsenal
  2.10x -> 2.01x the de-vigged bookmaker, Porto 1.96x -> 1.70x, Real Madrid
  0.67x -> 0.76x, Manchester United 0.59x -> 0.68x - each in the direction its
  shot residual implies. But Bayern out-shoots its Elo by +1.09 and v2 raises
  it from 0.180 to 0.200, widening the gap. In aggregate mean |model - market|
  barely moves (0.0187 -> 0.0183). **v2 is not a fix for limitation 1.**
- **This season cannot choose between v1 and v2.** The paired standard error is
  0.004 after matchday 1 and about 0.0014 over all 144 league-phase matches,
  still roughly three times the effect the backtest resolved.
  `live/<season>/model_tracking.csv` follows both from the same frozen
  information set as **monitoring, not a gate**; it would catch something badly
  wrong and nothing finer.
- **Shot ratings have no cross-league bridge**, so they enter as the *residual*
  of a per-league regression on Elo. No cached match carries both shot data and
  two clubs from different leagues - the extra leagues and every European match
  have no shot columns - and raw ratings put Celtic and Ajax above Manchester
  City. Regressing on Elo leaves league quality where European results have
  already calibrated it. Do not standardise within league: that is candidate
  1's error.
- **The shot ratings independently flag Arsenal and Porto** as over-rated by
  Elo (-1.52 and -1.18 shots on target below prediction), which is the two
  clubs the market diagnostic also disputes - the first corroboration of
  limitation 1 from a source the market had no part in.
- **Minutes-weighted squad value returned a null** (milestone 8, candidate B):
  pooled delta -0.00001, CI [-0.00047, +0.00044], measured against v1 + A. The
  two measures rank the clubs at Spearman 0.9894 - a club that can afford an
  expensive squad also gives those players the minutes - so weighting by
  minutes mostly reproduces the total. Coverage was identical, so the null is
  about the weighting and nothing else. See `results/candidate_gap_minutes/`.
- **A candidate must be paired against the accepted model, not against Elo.**
  `paired_losses_covariate` takes a `baseline`; candidate A passes
  `squad_value`. Measuring a candidate that contains the squad-value term
  against Elo alone would credit it with that term's gain as well as its own.
- **Favourite-longshot bias in the outright benchmark is measured, not
  assumed** (milestone 8, part 1). A second market agrees with the bookmaker
  within a few percent on favourites and differs about twofold on longshots, so
  proportional de-vigging understates the favourites: Arsenal is 1.83x the
  exchange's raw price, not 2.10x the bookmaker's de-vigged one. The exchange
  is too thin to replace the benchmark (13 usable prices against a
  pre-declared minimum of 15), so both are reported.
- **Two implementation defects preceded that acceptance, and both were found by
  inspecting the feature rather than the test statistic.** Incomplete name
  mapping (110 of 796 clubs) and national-team appearances counted as club
  membership (16.3% of players at the 2026-27 as-of, six days after the 2026
  World Cup). A null from a covariate is only evidence once the covariate has
  been looked at: print it, sort it, and see whether it says what it should.
