# Candidate C: a European goal-scale term - ACCEPTED

Declared in full and run once as binding, at commit `d4aa4f4`. Tested against
**model v2**, the accepted model, per the standing rule that a candidate is
paired against what is accepted and not against Elo.

Model development had been **closed** at milestone 9. It was reopened for this
one candidate, on historical evidence, and closed again afterwards.

## Why it was declared

The Poisson model is fitted on one pooled sample. At the 2026-27 cutoff that is
**65,389 matches, of which 6,405 (9.8%) are European**, so the intercept is set
by domestic football at **2.775** goals a match. The competition being forecast
does not score at that rate: across the cache the **UCL main phase runs 3.13**,
and over the **288 league-phase matches of 2024-25 and 2025-26** - the same
36-team format as 2026-27 - it was **3.32 goals a match, 64.6% over 2.5, 14.9%
draws**, against the published model's 2.82, 53.5% and 22.6%.

This reaches the team-level numbers, not just the scoreline grid: the simulator
draws scorelines from these rates, so an excess draw rate compresses the spread
of points and goal difference, and P(top 8), P(9-24) and P(25-36) are integrals
over that spread.

## The method

Three competition levels, with **domestic as the baseline** carrying neither
term (naming it would make the indicators collinear with `a`):

    log λ_h = a + h + b·d + Σβ_m x_m + c_lvl + (0 if neutral else g_lvl)
    log λ_a = a     - b·d - Σβ_m x_m + c_lvl

`c` is a goal **scale**, added to both log-rates; `g` is **extra home
advantage**, added to the home side only and switched off by `neutral=True`
exactly as `h` is. Every covariate is identical to v2's, so the comparison
isolates these four parameters.

**Per competition rather than one shared European term.** Both are identified
in every fold, so the choice was about what gets pooled, and the three do not
score alike: UCL main phase 3.13 goals a match, UEL 2.81, UECL 2.86, UCL
qualifying 2.80. A single European term would be dominated by the UEL/UECL mass,
which already sits near the domestic rate, and would under-correct exactly where
the error is. A three-way split was ruled out: UECL has no main-phase matches at
all before the first fold's cutoff.

**UCL qualifying rounds go with the other-European term**, not with the draw
they feed. They score 2.80 against the main phase's 3.13; pooling 543 qualifying
matches into the UCL term would have dragged its coefficient down by about a
fifth of the gap being closed. `european.is_main_phase` makes the split, from
`phase` where that is authoritative and from the stage string only for the
2021-22 to 2023-24 "mixed" files, raising on anything it cannot classify.

## Primary test: European scoreline log-loss, two-way clustered

    1,832 European matches, 405 home clusters, 404 away clusters
    weighted mean delta -0.00961   (negative = the candidate is better)
      two-way   95% CI [-0.01892, -0.00087]  excludes zero: True
      cluster   95% CI [-0.01461, -0.00441]  excludes zero: True
      per-match 95% CI [-0.01473, -0.00478]  excludes zero: True

    GUARD, European 1X2 log-loss: -0.00240  (declared condition: not positive)
      two-way   95% CI [-0.00622, +0.00138]  excludes zero: False

    DECISION: ACCEPT

**Why the scoreline rather than 1X2, decided before the run.** The candidate
acts on the goal distribution, and a 1X2 triple is a coarse functional of it. At
the effect size the raw rates implied, the 1X2 metric gives |z| under 0.9
*however right the term is* - the KL divergence the candidate can deliver is
**19 times larger** in the scoreline likelihood, which is also the likelihood
the model is fitted on. Declaring 1X2 would have been declaring a test known in
advance to be inconclusive. The 1X2 delta is kept as a non-worsening guard so
that a term improving goals while degrading match outcomes is still rejected; it
passes on the point estimate but its own interval does **not** clear zero, and
nothing here claims it does.

Three of four folds are clearly negative; the first, trained on a single season,
is flat:

| fold | train | n | delta | two-way 95% CI |
|---|---|---|---|---|
| 2022-23 | 2021-22 | 410 | +0.00012 | [-0.01870, +0.01826] |
| 2023-24 | +2022-23 | 407 | -0.01316 | [-0.03075, +0.00339] |
| 2024-25 | +2023-24 | 523 | -0.01131 | [-0.02675, +0.00314] |
| 2025-26 | +2024-25 | 492 | -0.01297 | [-0.03343, +0.00529] |

No single fold clears zero under two-way clustering; the acceptance rests on the
pooled sample, as candidate A's did.

## Fitted coefficients

| cutoff | `c_ucl_main` | `c_other_european` | `g_ucl_main` | `g_other_european` |
|---|---|---|---|---|
| 2022-23 | **+0.1046** | -0.0305 | -0.0364 | +0.0818 |
| 2023-24 | **+0.1028** | +0.0018 | -0.0392 | +0.0555 |
| 2024-25 | **+0.0928** | -0.0139 | -0.0240 | +0.0739 |
| 2025-26 | **+0.0964** | -0.0230 | -0.0099 | +0.0863 |
| 2026-27 | **+0.1040** | -0.0348 | +0.0040 | +0.0913 |

**The signal is the UCL goal scale**: `c_ucl_main` is +0.093 to +0.105 at every
cutoff, a factor of about **1.10** on both rates. `c_other_european` is small and
changes sign once, so UEL/UECL score close to what the domestic intercept already
predicts for them - which is what made the per-competition split the right call
rather than a shared term.

The home-advantage terms landed in the opposite place from the expectation that
motivated including them. `g_ucl_main` is **negative in all four backtest
folds** (-0.036 to -0.010): the UCL main phase has slightly *less* home
advantage than domestic football, not more. `g_other_european` is consistently
**+0.055 to +0.086**. The term was declared and kept as declared, including the
UCL half that fits near zero.

At the 2026-27 cutoff the in-sample likelihood-ratio statistic is **75.3 on 4
degrees of freedom**, and v2's own coefficients are unchanged to the last digit
(`a` 0.169513, `h` 0.233734, `b` 0.698176), so the four terms are additions
rather than a re-parameterisation.

## Where the gain comes from

| level | matches | scoreline delta | 1X2 delta | share of the total gain |
|---|---|---|---|---|
| `ucl_main` | 610 | **-0.02283** | -0.00308 | 79% |
| `other_european` | 1,222 | -0.00301 | -0.00206 | 21% |

One third of the scored European sample carries four fifths of the improvement,
which is the competition the term was built for.

**A pre-run power calculation under-predicted this by a factor of 2.3** and is
worth recording as a modelling error rather than quietly dropping. It predicted
about -0.0041 against the -0.0096 measured. Roughly 40% of the shortfall came
from assuming the non-UCL two thirds of the sample were inert, when they supply
21% of the gain; the rest came from using the 144 published 2026-27 league-phase
rates as a stand-in for the backtest's fixture mix, which understates the
per-match effect. The direction of the conclusion it supported - that 1X2 could
not resolve this candidate - is unaffected: the measured 1X2 interval does not
clear zero either.

## Calibration diagnostics, European matches, before and after

| | v2 | candidate C | observed |
|---|---|---|---|
| draw rate | 0.2326 | 0.2251 | **0.1943** |
| goals per match | 2.7798 | 2.9001 | **2.9733** |
| over 2.5 | 0.5245 | 0.5523 | **0.5737** |

**Goals and over-2.5 move most of the way; the draw rate barely moves.** That is
the predicted behaviour of a scale term and not a disappointment: raising both
rates disperses the Skellam difference and so lowers P(0), but only weakly. The
analysis in `docs/METHODOLOGY.md` §3 put roughly a third of the draw gap on the
intercept and the rest on the independent-Poisson form itself; the measured
0.2326 → 0.2251 against a 0.1943 target is consistent with that, closing about
a quarter.

Per fold, candidate against observed:

| fold | draws | goals | over 2.5 |
|---|---|---|---|
| 2022-23 | 0.228 / 0.215 | 2.86 / 2.87 | 0.543 / 0.529 |
| 2023-24 | 0.225 / 0.197 | 2.91 / 3.09 | 0.554 / 0.595 |
| 2024-25 | 0.223 / 0.187 | 2.91 / 3.01 | 0.555 / 0.591 |
| 2025-26 | 0.225 / 0.183 | 2.91 / 2.92 | 0.556 / 0.575 |

## Two weaknesses, declared before the run

**The motivating evidence is not independent of the test.** The 288
league-phase matches that diagnosed the gap are 2024-25 and 2025-26, two of the
four seasons the forward chain scores. The fit itself is clean - each fold is
fitted only on data before its own cutoff, and 2024-25 is scored by a model that
never saw it - but the *decision to test a goal-scale term at all* came from
seasons inside the test set. **A pass here is a weaker confirmation than an
independent holdout would be**, and this is recorded in the decision JSON as
`motivating_evidence_overlaps_test_seasons`. Note that the two folds predating
the evidence point the same way (2022-23 flat, 2023-24 -0.0132), which is mild
reassurance and not a substitute.

**Domestic non-worsening check.** `a`, `h`, `b` and every β are refitted
alongside the new terms, so domestic predictions change too:

| domestic, 17,356 matches | delta | cluster 95% CI |
|---|---|---|
| scoreline | +0.000024 | [-0.000102, +0.000145] |
| 1X2 | -0.000081 | [-0.000146, -0.000018] |

No degradation. The scoreline delta is indistinguishable from zero and the 1X2
delta is marginally favourable, so the new terms are not absorbing variance that
belongs to the shared ones.

## Sensitivities, which do not decide

| scope | delta | cluster 95% CI | n |
|---|---|---|---|
| **European scoreline (primary)** | **-0.00961** | **[-0.01461, -0.00441]** | 1,832 |
| all matches, scoreline | -0.00229 | [-0.00355, -0.00108] | 19,188 |
| domestic scoreline (circular here) | +0.00002 | [-0.00010, +0.00015] | 17,356 |
| European 1X2 (the guard) | -0.00240 | [-0.00469, -0.00025] | 1,832 |

## Coverage

Every match carries a competition, so there is no missing data. What the
`coverage` block records instead is the share of each scored European sample the
UCL term can act on at all - the dilution factor behind the power table:

| test season | UCL main share of the European sample |
|---|---|
| 2022-23 | 30.5% |
| 2023-24 | 30.7% |
| 2024-25 | 35.6% |
| 2025-26 | 35.6% |

Pooled: 33.3%.

## Not in the forecast

**Accepted as a candidate; not promoted.** The published forecast remains model
v2. Promotion to v3 needs its own gates - v2 reproducing bit-identically with
the new terms disabled, a re-simulation at 50,000 iterations, and the app and
weekly update switched over - and is a separate decision.

Note for whoever does it: `src/ucl/tracking.py` calls `CovariateModel.rates`
directly and would need the competition level passed. It raises rather than
mis-scoring, by the guard added with this candidate, but it is the one call site
no byte-identity gate covers.

## An open lead, post hoc and untested

The draw excess and the home-win shortfall both survive this candidate, and
`g_ucl_main` fits near zero. A different explanation would fit all three at
once: that the **Elo difference is too compressed in European matches**, so a
European-specific slope `b` rather than an intercept is what the data wants. A
larger `b` lowers P(draw), raises P(home) for the stronger home side, and leaves
the mean goal rate roughly where it is.

Generated by looking at this candidate's residual behaviour, so **post hoc**;
not quantified, not fitted, not tested. It would need its own pre-declaration
and a holdout that played no part in generating the idea - which excludes
2024-25 and 2025-26.

Files: `candidate_euro_scale_folds.csv`, `candidate_euro_scale_paired.csv`,
`candidate_euro_scale_decision.json`, `run.log`. The paired file reproduces
every interval above to 1e-15.
