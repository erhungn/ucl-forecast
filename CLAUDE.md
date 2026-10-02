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
1. Ratings: ClubElo ratings as of the cutoff date (cross-league calibrated).
2. Goal model: Dixon-Coles with an Elo covariate:
   `log(lambda_ij) = mu + home + beta * (elo_i - elo_j) / 400 + att_i - def_j`
   - att/def deviations shrunk toward zero (ridge penalty or Bayesian prior)
   - exponential time-decay weights (tune xi on backtests)
   - low-score correction rho
   - optional: fit on xG instead of goals and compare
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
1. Data pipeline + team-name mapping across sources + cutoff assertions
2. Elo baseline model and simulator end-to-end (no Dixon-Coles yet)
3. Tiebreaker and bracket tests passing
4. Dixon-Coles with Elo covariate, fitted and compared to Elo baseline
5. Backtests on 2024-25 and 2025-26
6. Frozen 2026-27 pre-tournament forecast; later, an "update after
   matchday k" mode
