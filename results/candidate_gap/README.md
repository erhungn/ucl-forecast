# Candidate A: GAP shot ratings - ACCEPTED

Milestone 8, declared and run once as binding. Tested against **model v1** at
commit `163b1ef` - not against the Elo baseline, which is what the earlier
candidates were measured against. Pairing a candidate that contains the
squad-value term against Elo alone would have credited it with that term's gain
as well as its own; `paired_losses_covariate` gained a `baseline` argument for
this, and the squad-value candidate still reproduces its archived result
through the old path.

## The method

Wheatcroft's Generalised Attacking Performance ratings, from *Forecasting
football matches by predicting match statistics* (arXiv 2001.09097),
implemented from the paper rather than from memory. Four ratings per club per
measure - attacking and defensive, home and away - with expected output

    S_h_hat = (H_i^a + A_j^d) / 2        S_a_hat = (A_j^a + H_i^d) / 2

and the paper's equations 7 and 8 moving all eight ratings towards the error,
each floored at zero. Three measures, as the paper uses: **shots on target,
shots off target, corners.** The learning rate and the two phi parameters are
fitted per cutoff by least squares against observed attacking output - the
paper's equation 2 - an objective that never sees a match result, so fitting it
cannot leak the thing being forecast.

Fitted at the 2026-27 cutoff: shots on target `lambda` 0.135, `phi_home` 0.604,
`phi_away` 0.571; shots off target 0.179 / 0.589 / 0.559; corners 0.095 / 0.622
/ 0.606.

## The cross-league problem, and what was done about it

**No match in the cache carries both shot data and two clubs from different
leagues** - the seven extra leagues and all 6,423 European matches have no shot
columns at all - so GAP ratings have no cross-league bridge, unlike Elo (which
European results calibrate) or squad value (euros are a common unit). Raw
ratings are confounded accordingly: at the 2022-23 cutoff **Celtic and Ajax
out-rate Manchester City on shots on target**, because they dominate weaker
leagues.

So the covariate is **the part of the rating that Elo does not explain**:
within each league the four ratings are regressed on the clubs' Elo and only
the residual is kept. League quality stays with Elo, where it is already
calibrated, and what remains is shot performance relative to what a club's
rating predicts - a quantity that means the same thing in two leagues. The raw
ratings correlate +0.60 with Elo; the residuals correlate +0.000 by
construction, and Celtic leaves the top five.

Standardising within league was **not** done: that is precisely what deleted
league quality and made candidate 1 significantly worse.

## Primary test: x3-weighted pooled log-loss, frozen, cluster bootstrap

    19,188 matches in 1,309 clusters, delta -0.00042
      cluster   95% CI [-0.00076, -0.00008]  excludes zero: True
      per-match 95% CI [-0.00074, -0.00010]  excludes zero: True

    DECISION: ACCEPT

All four folds are negative: 2022-23 -0.00071, 2023-24 -0.00020, 2024-25
-0.00038, 2025-26 -0.00041. Only the first clears zero on its own.

The interval clears zero by 0.00008, so it was checked for bootstrap noise and
is stable: 4,000 / 20,000 / 50,000 draws and five further seeds all give an
upper bound between -0.00008 and -0.00009.

## Fitted coefficients

| cutoff | `b_elo` | `b_d_value` | sot | shots off | corners |
|---|---|---|---|---|---|
| 2022-23 | 0.7047 | +0.0484 | **+0.0287** | -0.0045 | +0.0004 |
| 2023-24 | 0.7040 | +0.0431 | **+0.0253** | -0.0051 | -0.0035 |
| 2024-25 | 0.7100 | +0.0403 | **+0.0305** | -0.0068 | -0.0043 |
| 2025-26 | 0.7069 | +0.0341 | **+0.0244** | -0.0050 | -0.0022 |
| 2026-27 | 0.6982 | +0.0400 | **+0.0233** | -0.0034 | -0.0032 |

**The signal is shots on target.** Its coefficient is positive and stable at
+0.023 to +0.031 across every cutoff. Shots *off* target is consistently
slightly negative, which is coherent - shooting from poor positions is weak
evidence, and it is why the paper prefers shots on target to raw shots -
and corners carry essentially nothing. `b_d_value` falls from +0.048 to
+0.034-0.040 once the shot ratings are present, so the two overlap a little, as
expected of two views of the same quality.

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

## Sensitivities, which do not decide and which qualify the result

| scope | delta | cluster 95% CI | n |
|---|---|---|---|
| **pooled (primary)** | **-0.00042** | **[-0.00076, -0.00008]** | 19,188 |
| domestic only | -0.00031 | [-0.00067, +0.00005] | 17,356 |
| European only | -0.00079 | [-0.00165, +0.00007] | 1,832 |

**Neither half clears zero on its own.** The acceptance rests on the pooled
sample; stated plainly, this is a small effect that needed all 19,188 matches
to resolve. Both halves point the same way, and the European point estimate is
about twice the domestic one - which is the direction the Elo-residual design
was meant to make possible, since a raw cross-league comparison would have been
actively misleading there.

## Coverage

| test season | European | domestic | clubs rated |
|---|---|---|---|
| 2022-23 | 43.2% | 70.4% | 291 |
| 2023-24 | 47.4% | 70.0% | 301 |
| 2024-25 | 43.5% | 69.8% | 310 |
| 2025-26 | 45.8% | 70.2% | 318 |

Shots exist for 33,930 of 59,316 cached matches: 95.9% of main-league matches,
**none** of the seven extra leagues and **none** of the European matches.
**29 of the 36 league-phase clubs** have a shot rating; Bodo/Glimt, LASK,
Sabah, Shakhtar Donetsk, Slavia Praha, Slovan Bratislava and Viking do not, and
fall back on the rest of the model.

## Mechanism: who the shot ratings disagree with Elo about

Shots on target beyond what Elo predicts, at the 2026-27 cutoff:

| above | | below | |
|---|---|---|---|
| Manchester United | +1.58 | Borussia Dortmund | -2.35 |
| VfB Stuttgart | +1.31 | **Arsenal** | **-1.52** |
| RB Leipzig | +1.18 | **Porto** | **-1.18** |
| Real Madrid | +1.18 | Aston Villa | -0.95 |
| Lens | +1.11 | Lille | -0.79 |

Worth noting beside the market diagnostic: **Arsenal and Porto, the two clubs
whose forecast probabilities most exceed the market, are the ones the shot
ratings say Elo over-rates.** That is corroboration from a source the market
had no part in, and it is the first independent support for the README's
limitation 1.

## Not yet in the forecast

Accepted as a candidate; **not promoted**. `SquadValueModel` carries exactly one
covariate in four places, so wiring a second into the simulator and regenerating
`live/` is separate work, declared as follow-on in the milestone 8 plan. The
published forecast remains model v1 until that is done.

Files: `candidate_gap_folds.csv`, `candidate_gap_paired.csv`,
`candidate_gap_decision.json`, `run.log`.
