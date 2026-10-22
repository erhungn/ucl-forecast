# Methodology

The mathematics of the UCL 2026-27 forecast, as implemented. Every formula
names the file and function that computes it, and every number was read from
the code or from a fitted output rather than from a write-up.

Fitted values quoted as "at the 2026-27 cutoff" are at `cutoff_date`
= **2026-09-07**, the day before matchday 1, and come from
`results/model_v2/model_2026-27.json` unless stated otherwise.

**Contents**

1. [Elo ratings](#1-elo-ratings)
2. [Association offsets: a Davidson model](#2-association-offsets-a-davidson-model)
3. [The Poisson match model](#3-the-poisson-match-model)
4. [Scorelines and the closed forms](#4-scorelines-and-the-closed-forms)
5. [GAP shot ratings and the Elo residual](#5-gap-shot-ratings-and-the-elo-residual)
6. [The Monte Carlo simulator](#6-the-monte-carlo-simulator)
7. [Evaluation metrics](#7-evaluation-metrics)
8. [Inference](#8-inference)
9. [De-vigging](#9-de-vigging)
10. [Rejected methods](#10-rejected-methods)

---

## 1. Elo ratings

`src/ucl/models/elo.py` — `replay()`, `goal_difference_multiplier()`,
`elo_as_of()`. Constants in `src/ucl/config.py`.

ClubElo was unreachable, so ratings are replayed from every cached domestic and
European result from `ELO_BURN_IN_START` = **2012-01-01** to the cutoff. At the
2026-27 cutoff that is **65,389 matches** over **966 clubs**
(`EloResult.diagnostics`).

### Expected score

For a home club rated $R_h$ against an away club rated $R_a$, with
$\eta$ = `HOME_ADVANTAGE` = **65** Elo points:

$$E_h = \frac{1}{1 + 10^{-(R_h + \eta - R_a)/400}}$$

Implemented in `replay()` as
`1.0 / (1.0 + 10.0 ** (-(home + params.home_advantage - away) / 400.0))`.

$\eta$ enters **here and nowhere else**. The match model of §3 uses the raw
difference and carries home advantage in its own intercept; adding $\eta$ there
too would count it twice, once in Elo points and once in goals. The constant's
docstring in `config.py` says so, and the separation is the project's single
most repeated invariant.

### The update

With the realised score $S_h \in \\{1, \tfrac12, 0\\}$ for a home win, draw and
loss:

$$\Delta = K \cdot g(|\mathrm{gd}|) \cdot (S_h - E_h), \qquad
R_h \leftarrow R_h + \Delta, \qquad R_a \leftarrow R_a - \Delta$$

The update is zero-sum by construction: the away club loses exactly what the
home club gains.

$$K = \begin{cases}
20 & \text{domestic} \\
25 & \text{European}
\end{cases}$$

`K_DOMESTIC` = 20 and `K_EUROPEAN` = 25.

### Goal-difference multiplier

`goal_difference_multiplier(gd, exponent)`. With `GD_EXPONENT = None`, the
default, this is the classic World Football Elo weighting:

$$g(d) = \begin{cases}
1 & d \le 1 \\
1.5 & d = 2 \\
\dfrac{11 + d}{8} & d \ge 3
\end{cases}$$

If an exponent $\gamma$ is given instead, $g(d) = (1 + d)^{\gamma}$, so
$\gamma = 0$ ignores the margin entirely.

**This rule is a judgement, not a fit.** It is not identified by the data: the
exponent tuned to 0.0 on one season and to 0.90 on four, trading off against
$K$. The default stands because tuning returned a null (pooled
$\Delta$ log-loss $-0.00040$, cluster CI $[-0.00126, +0.00049]$).

### Entering and regressing between seasons

A club seen for the first time (`replay().enter`) starts at its league's mean
plus `NEW_CLUB_OFFSET` = **−50**, since most arrivals are promotions. A club
whose association has no cached domestic league has no league mean, so it
starts at the association baseline

$$B(\text{assoc}) = R_0 + o_{\text{assoc}}, \qquad R_0 = 1500$$

where $R_0$ is `START_RATING`

with $o$ the offset of §2 and no promotion penalty to apply.

At a season boundary — detected **per association**, because leagues label and
schedule seasons differently — every club in that association is regressed
toward a target $T$:

$$R \leftarrow T + \rho\,(R - T), \qquad \rho = 0.90$$

$\rho$ is `SEASON_CARRYOVER` = 0.90.

`CARRYOVER_TARGET = "league_mean"` makes $T$ the mean rating of the
association's clubs. The targets are computed **once, before any club is
moved**, so the result cannot depend on set iteration order. The alternative
target `"club_long_run"` is an exponentially weighted mean of the club's own
season-end ratings with decay $\tfrac12$ per season, most recent first:

$$T_{\text{long run}} = \frac{\sum_{i\ge 0} 2^{-i} R_{n-i}}{\sum_{i\ge 0} 2^{-i}}$$

falling back to the league mean for a club with fewer than two seasons behind
it. It is not the published setting.

---

## 2. Association offsets: a Davidson model

`src/ucl/models/elo.py` — `fit_association_offsets()`.

Starting every club at 1500 would leave the between-league spread to emerge
from a few thousand European matches, far too slowly. Instead association
strength offsets are fitted once, in Elo points, from all pre-cutoff European
matches, and added to the starting ratings.

A **Davidson** model (a Bradley-Terry extension with an explicit draw term).
With association offsets $o$ and $\eta$ = 65 as above, write

$$p = 10^{(o_{\text{home}} + \eta)/400}, \qquad q = 10^{o_{\text{away}}/400}$$

then

$$P(\text{home}) = \frac{p}{p + q + \nu\sqrt{pq}}, \qquad
P(\text{draw}) = \frac{\nu\sqrt{pq}}{p + q + \nu\sqrt{pq}}, \qquad
P(\text{away}) = \frac{q}{p + q + \nu\sqrt{pq}}$$

The draw parameter $\nu \ge 0$ keeps draws from biasing the offsets; the code
optimises $\log\nu$ so positivity is automatic. Working directly in Elo points
avoids a detour through a goal-rate scale.

The objective is the negative log-likelihood of the realised three-way
outcomes,

$$-\log L(o, \nu) = -\sum_{i} \log P_{y_i}\big(o, \nu\big)$$

minimised by L-BFGS-B. Offsets are **mean-centred inside the objective** on
every evaluation, which is what makes them identifiable — adding a constant to
all of them leaves every probability unchanged. Associations with fewer than
`min_matches` = **10** European matches are held at 0 rather than allowed to
chase noise.

Fitted at the 2026-27 cutoff, from **6,405** European matches:

| quantity | value |
|---|---|
| associations fitted | 54 |
| associations held at zero | 0 |
| draw parameter $\nu$ | **0.6202** |
| mean offset | 0 (exactly, by the centring) |
| largest offset | ENG **+412.3** |
| smallest offset | **−450.6** |

Offsets per association are archived in
`results/elo_baseline_v0/association_elo_2026-27.csv`.

> **Honest caveat.** At this cutoff L-BFGS-B returns `success = False`
> (`diagnostics["offset_converged"]`). The likelihood is flat near the optimum
> in 54 parameters and the returned point is used as-is. The offsets are stable
> in the way that matters downstream — they correlate 0.9991 between home
> advantage 40 and 90, mean absolute difference 6.4 Elo, which is why
> `backtest._cached_offsets` caches them in 5-Elo buckets — but the flag is
> real and is recorded here rather than left in a diagnostics dict.

---

## 3. The Poisson match model

`src/ucl/models/poisson.py` — `PoissonModel.rates()`, `fit_poisson()`.
`src/ucl/models/covariates.py` — `CovariateModel.rates()`,
`fit_covariate_model()`, `ForecastModel.rates_for()`.
`src/ucl/models/forecast.py` — `fit_v1()`, `fit_v2()`.

The tiebreakers need goal difference, so the model must produce **scorelines**,
not win probabilities. Goals are independent Poisson with a log link on the
rating difference.

### The linear predictor

$$d = \frac{R_h - R_a}{400} \qquad \text{(no } \eta \text{ here — see §1)}$$

Baseline (`PoissonModel`):

$$\log \lambda_h = a + h + b\,d, \qquad \log \lambda_a = a - b\,d$$

With $M$ **symmetric** covariates $x_m$ — one value per match, acting like the
rating difference (`CovariateModel.rates`):

$$\log \lambda_h = a + h + b\,d + \sum_{m=1}^{M} \beta_m x_m, \qquad
\log \lambda_a = a - b\,d - \sum_{m=1}^{M} \beta_m x_m$$

The sign flip on the away side is what makes them symmetric: a covariate
favouring the home club disfavours the away club by the same amount. The model
also supports **per-side** covariates $u$, added to whichever team they
describe and *not* subtracted from the other:

$$\log \lambda_h \mathrel{+}= c_k u_{k,\text{home}}, \qquad
\log \lambda_a \mathrel{+}= c_k u_{k,\text{away}}$$

Fixture congestion was the one per-side candidate: short rest tires one team,
not both. Setting any covariate column to zero reproduces the baseline
exactly, which is how an unrated club falls back and how the tests pin the
nesting.

`neutral=True` drops $h$ from the home side, which is how the final is played.

### Fitting

Poisson maximum likelihood on the ratings **as they stood before each match**;
fitting on post-match ratings would leak the result into its own covariate.
Dropping terms that do not depend on the parameters, `fit_covariate_model`
minimises

$$-\ell(\theta) = \sum_i \Big(\lambda_{h,i} - y_{h,i}\log\lambda_{h,i}\Big)
+ \sum_i \Big(\lambda_{a,i} - y_{a,i}\log\lambda_{a,i}\Big)$$

with analytic gradients. Writing $r_h = \lambda_h - y_h$ and
$r_a = \lambda_a - y_a$ for the residuals:

$$\frac{\partial(-\ell)}{\partial a} = \sum_i (r_{h,i} + r_{a,i}), \qquad
\frac{\partial(-\ell)}{\partial h} = \sum_i r_{h,i}, \qquad
\frac{\partial(-\ell)}{\partial b} = \sum_i d_i\,(r_{h,i} - r_{a,i})$$

$$\frac{\partial(-\ell)}{\partial \beta_m} = \sum_i x_{m,i}\,(r_{h,i} - r_{a,i}),
\qquad
\frac{\partial(-\ell)}{\partial c_k} = \sum_i \big(u_{k,h,i}\,r_{h,i} + u_{k,a,i}\,r_{a,i}\big)$$

L-BFGS-B, `ftol=1e-12`, `gtol=1e-10`; plain BFGS reports precision loss on a
likelihood this flat. The reported `log_likelihood` is the **full** Poisson
log-likelihood including the dropped constants, recomputed with
`scipy.stats.poisson.logpmf`.

One difference worth knowing: `fit_poisson` multiplies $h$ by an indicator so a
neutral-venue column can be honoured, while `fit_covariate_model` applies $h$
to every row. Every fitting frame used for the published models is
venue-resolved domestic and European football, so the two agree.

### The covariates

| covariate | definition | built by |
|---|---|---|
| `d_value` | $\log V_h - \log V_a$, squad market value in euros; **0** unless both clubs are valued | `covariates.value_covariate()` |
| `d_gap_sot` | residual GAP difference, shots on target | `forecast.gap_covariates()`, §5 |
| `d_gap_shots_off` | residual GAP difference, shots off target | ” |
| `d_gap_corners` | residual GAP difference, corners | ” |

A log ratio makes the squad-value term scale-free: multiplying every valuation
by a constant leaves the covariate unchanged, so the currency unit and any
market-wide inflation cancel.

`ForecastModel.rates()` **raises** rather than silently returning Elo-only
rates, because the covariates are keyed by club id and cannot be evaluated
without them — a guard added after that exact bug cost two re-runs.

### Fitted coefficients, model v2

At the 2026-27 cutoff: **65,389 matches**, converged, full log-likelihood
**−192,131.957**.

| parameter | value | meaning |
|---|---|---|
| $a$ | **0.169513** | log base rate, ≈ 1.185 goals |
| $h$ | **0.233734** | home advantage, a factor of $e^{h}$ ≈ 1.263 on the home rate |
| $b$ | **0.698176** | per 400 Elo points |
| $\beta$ for `d_value` | **+0.040042** | per unit of log value ratio |
| $\beta$ for `d_gap_sot` | **+0.023307** | per shot on target above Elo's prediction |
| $\beta$ for `d_gap_shots_off` | **−0.003423** | |
| $\beta$ for `d_gap_corners` | **−0.003195** | |

Model v1 is the same form with `d_value` alone; its coefficients are in
`results/model_v1/`. Coverage at this cutoff: `d_value` reaches 232 clubs and
**79.2%** of the 144 league-phase fixtures, the three GAP terms 329 clubs and
**64.6%**.

Raw coefficients are not comparable across covariates — one is a log ratio of
euros, the others are differences in shots. `significance.standardised_coefficients()`
multiplies each by its covariate's standard deviation on the fitting sample,
giving the change in $\log\lambda$ per one-standard-deviation move.

### A known limitation: the intercept is domestic

The model is fitted on one pooled sample, so $a$ and $h$ are set by whatever
scores most of the matches. At the 2026-27 cutoff that sample is **65,389
matches, of which only 6,405 (9.8%) are European**:

| sample | matches | goals/match | draws | home wins | over 2.5 |
|---|---|---|---|---|---|
| domestic | 58,984 | 2.775 | 0.253 | 0.440 | 0.527 |
| European, all | 6,405 | 2.897 | 0.215 | 0.477 | 0.548 |
| UCL main phase | 1,372 | 3.132 | 0.196 | — | — |
| other European | 5,033 | 2.833 | — | — | — |

So the intercept lands near 2.78 goals a match, and the competition being
forecast does not score at that rate. Over the **288 league-phase matches of
2024-25 and 2025-26** — the 36-team format, the same tournament as 2026-27 —
the observed rates were **3.32 goals/match, 64.6% over 2.5, 14.9% draws** and
roughly 51% home wins. What the published model says across the 144 league-phase
fixtures at the cutoff:

| | model | 36-team league phase |
|---|---|---|
| goals per match | 2.82 | **3.32** |
| P(over 2.5) | 0.535 | **0.646** |
| P(draw) | 0.226 | **0.149** |
| P(home) | 0.452 | **0.514** |

The goal scale is low by about 15% and the draw rate high by about 7 points,
measured on two completed seasons with no 2026-27 data involved.

**One parameter would fix most, but not all, of it.** Scaling every
$\lambda$ by 1.165 moves the published forecast to 3.29 goals/match, 0.636 over
2.5 and 0.468 home wins — essentially onto the 288-match targets — while P(draw)
only falls 0.226 → 0.203 against a target of 0.149. A multiplicative scale
raises the total rate, which disperses the Skellam difference and so reduces
P(0), but not by enough: **roughly a third of the draw gap is attributable to
the intercept and the rest is not.** The residual is a property of the
independent-Poisson form itself — exactly what the Dixon-Coles $\tau$ of §10
corrects — and that null was measured on the **pooled** sample, never on
European matches alone. It is the open question this limitation leaves behind.

**What it affects.** Directly, every goal-scale output: the displayed scoreline
grid, `p_over_2_5`, `modal_score` and `p_modal`. Indirectly and more
importantly, the **team-level probabilities**: the simulator draws scorelines
from these same rates, so an excess draw rate compresses the spread of points
and goal difference, and P(top 8), P(9-24) and P(25-36) are integrals over that
spread. A compressed goal difference also changes how often Article 18's
criterion (a) resolves a tie. The 1X2 ordering is not in question — this moves
probability between the three outcomes by one or two points, not the ranking —
but it is a model limitation, not a presentation one.

**Candidate C** tests the direct fix — a per-competition intercept shift fitted
on pre-cutoff European matches, declared and scored under the §8 protocol. Its
verdict, accepted or rejected, is in `results/candidate_euro_scale/`.

---

## 4. Scorelines and the closed forms

`src/ucl/models/scorelines.py` — `joint_matrix()`, `folded_matrix()`,
`modal_scoreline()`, `probability_over()`, `probability_margin_at_least()`.
`src/ucl/backtest.py` — `probabilities_from_rates()`.

### The joint matrix

Goals are independent, so the scoreline distribution is an outer product
(`joint_matrix`, `MAX_GOALS = 10`):

$$P(H = i, A = j) = \frac{e^{-\lambda_h}\lambda_h^{\,i}}{i!}\cdot
\frac{e^{-\lambda_a}\lambda_a^{\,j}}{j!}$$

1X2 probabilities are sums over its triangles and diagonal, renormalised over
the truncated grid (`probabilities_from_rates`):

$$P(\text{home}) = \sum_{i>j} P(i,j), \qquad
P(\text{draw}) = \sum_{i=j} P(i,j), \qquad
P(\text{away}) = \sum_{i<j} P(i,j)$$

There is **no low-score correction**: the published model is independent
Poisson. Dixon-Coles was built, tested and returned a null (§10).

### Total goals, exactly

A sum of independent Poissons is Poisson, $H + A \sim \mathrm{Poisson}(\lambda_h + \lambda_a)$,
so no grid is needed (`probability_over`):

$$P(H + A > \ell) = 1 - F_{\mathrm{Poisson}}\big(\lfloor \ell \rfloor;\ \lambda_h + \lambda_a\big)$$

For the usual betting line $\ell = 2.5$ this is $P(H + A \ge 3)$.

### Winning margin, exactly

A difference of independent Poissons is **Skellam**, with

$$P(H - A = k) = e^{-(\lambda_h + \lambda_a)}
\left(\frac{\lambda_h}{\lambda_a}\right)^{k/2}
I_{|k|}\big(2\sqrt{\lambda_h \lambda_a}\big)$$

for $I_{|k|}$ the modified Bessel function of the first kind. So
(`probability_margin_at_least`):

$$P(H - A \ge m) = 1 - F_{\mathrm{Skellam}}(m - 1;\ \lambda_h, \lambda_a)$$

Both closed forms are checked against sums over the scoreline matrix in
`tests/test_scorelines.py` — two implementations of the same arithmetic, which
is the point of having them.

### The displayed grid

`folded_matrix(bands=5)` is what the app shows. Goals $0 \dots \text{bands}-1$
keep a row and column each and the tail is gathered into the last one, using
the **exact** survival function rather than a truncated sum:

$$P(H \ge \text{bands}) = \mathrm{sf}(\text{bands} - 1;\ \lambda_h)$$

so the $6 \times 6$ grid sums to 1 to within $3\times10^{-16}$. A bare
$0\text{-}4 \times 0\text{-}4$ truncation holds only **0.849** of the mass on the
most one-sided fixture of matchday 2 — which is why the extra band exists.

The **modal scoreline** is the argmax of the matrix. It is reported because it
is asked for, not because it is informative: it rarely clears 13% and is 1-0,
1-1 or 2-0 for most fixtures.

---

## 5. GAP shot ratings and the Elo residual

`src/ucl/models/gap.py` — `replay()`, `fit_params()`, `residual_ratings()`,
`match_difference()`. Assembled by `src/ucl/models/forecast.py` —
`gap_residuals()`.

Wheatcroft's Generalised Attacking Performance ratings, from *Forecasting
football matches by predicting match statistics* (arXiv
[2001.09097](https://arxiv.org/abs/2001.09097)), implemented from the paper.

### Ratings and predictions

Each club carries **four** ratings per measure: attack and defence, home and
away. For home club $i$ against away club $j$, expected attacking output is

$$\hat S_h = \frac{H_i^{a} + A_j^{d}}{2}, \qquad
\hat S_a = \frac{A_j^{a} + H_i^{d}}{2}$$

with errors $e_h = S_h - \hat S_h$ and $e_a = S_a - \hat S_a$.

### The update

The paper's equations 7 and 8, with learning rate $\lambda$ and cross-venue
split parameters $\phi_1$ (home club) and $\phi_2$ (away club), every rating
floored at zero. For the **home** club $i$:

$$H_i^{a} \leftarrow \max\big(H_i^{a} + \lambda\phi_1 e_h,\ 0\big), \qquad
A_i^{a} \leftarrow \max\big(A_i^{a} + \lambda(1-\phi_1) e_h,\ 0\big)$$

$$H_i^{d} \leftarrow \max\big(H_i^{d} + \lambda\phi_1 e_a,\ 0\big), \qquad
A_i^{d} \leftarrow \max\big(A_i^{d} + \lambda(1-\phi_1) e_a,\ 0\big)$$

and for the **away** club $j$, with $\phi_2$ and the venue roles exchanged:

$$A_j^{a} \leftarrow \max\big(A_j^{a} + \lambda\phi_2 e_a,\ 0\big), \qquad
H_j^{a} \leftarrow \max\big(H_j^{a} + \lambda(1-\phi_2) e_a,\ 0\big)$$

$$A_j^{d} \leftarrow \max\big(A_j^{d} + \lambda\phi_2 e_h,\ 0\big), \qquad
H_j^{d} \leftarrow \max\big(H_j^{d} + \lambda(1-\phi_2) e_h,\ 0\big)$$

$\phi$ splits each update between the rating just used and its opposite-venue
twin: a club that over-performs at home is a little better away too, but not
equally so.

### Measures, initialisation, promotion

Three measures, as the paper uses (`MEASURES`): **shots on target**, **shots
off target** = $\max(\text{shots} - \text{shots on target},\ 0)$, and
**corners**. The floor exists for one row in 33,930 (Newcastle v West Ham,
2021-08-15) that records more shots on target than shots.

The paper initialises at zero in a league's first season with statistics;
`replay(initial="league_mean")` starts a club at its league's average output
instead — the same recursion from a sensible point, converging far sooner, and
with 2016-17 onwards as burn-in the two agree well before the first scored
fold. A club new to a league inherits the **mean of the clubs that left it**,
as the paper prescribes for promotion and relegation.

### Fitting the parameters

`fit_params()` minimises the paper's equation 2 — squared error between
predicted and observed attacking output:

$$\min_{\lambda, \phi_1, \phi_2} \sum_i \big(e_{h,i}^2 + e_{a,i}^2\big)$$

by L-BFGS-B over $\lambda \in [10^{-4}, 0.5]$ and $\phi_{1,2} \in [0.5, 1]$.

**The objective never looks at a match result**, only at how many shots or
corners were taken, so fitting it cannot leak the thing the forecast predicts.

Fitted at the 2026-27 cutoff (`model_2026-27.json`, `gap_params`):

| measure | $\lambda$ | $\phi_1$ (home) | $\phi_2$ (away) |
|---|---|---|---|
| shots on target | 0.13465 | 0.60363 | 0.57138 |
| shots off target | 0.17911 | 0.58912 | 0.55882 |
| corners | 0.09533 | 0.62233 | 0.60586 |

### The Elo-residual step

**The ratings have no cross-league bridge.** No cached match carries both shot
data and two clubs from different leagues — the seven extra leagues and all
6,423 European matches have no shot columns at all — and a weak league's clubs
take more shots against weaker opponents. Raw ratings put Celtic and Ajax above
Manchester City on shots on target.

So the covariate is the part of the rating **Elo does not explain**. Within each
league, each of the four ratings is regressed on the clubs' Elo by ordinary
least squares and only the residual is kept (`residual_ratings`):

$$v_c = \alpha_L + \beta_L R_c + \varepsilon_c, \qquad
\tilde v_c = v_c - \big(\hat\alpha_L + \hat\beta_L R_c\big)$$

League quality stays with Elo, where European results have already calibrated
it, and what remains is shot performance relative to what a club's rating
already predicts — a quantity that means the same thing in two leagues. Raw
ratings correlate **+0.60** with Elo; the residuals correlate **+0.000** by
construction. A league with fewer than four rated clubs is left alone, since a
slope on three points is noise, and a club with no Elo is dropped so the match
difference treats it as uncovered.

Standardising within league is deliberately **not** done: that is precisely
what deleted league quality and made candidate 1 significantly worse (§10).

The per-match covariate (`match_difference`) is the same expected-output
difference, built from residuals:

$$x = \hat S_h - \hat S_a
= \frac{\tilde H_i^{a} + \tilde A_j^{d}}{2} - \frac{\tilde A_j^{a} + \tilde H_i^{d}}{2}$$

and **0** where either club is unrated, the same convention the squad-value
covariate uses, so an uncovered match falls back on the rest of the model
rather than being dropped from the fit.

Because the regression is linear and the GAP difference is a linear combination
of ratings, the residual difference is the raw difference minus its
Elo-predicted part — one subtraction, no standardisation. The residual risk,
stated rather than hidden: the slope $\hat\beta_L$ differs by league, so "+1
shot above expectation" is not exactly the same quantity in the Eredivisie as
in La Liga. The residual removes the level, not the scale.

---

## 6. The Monte Carlo simulator

`src/ucl/sim/league_phase.py` — `simulate_match()`, `simulate_league_phase()`.
`src/ucl/sim/knockout.py` — `play_tie()`, `play_final()`, `run_knockout()`.
`src/ucl/sim/bracket.py`, `src/ucl/sim/tiebreakers.py`.
Driven by `scripts/simulate.py`.

**N = `N_SIMULATIONS` = 50,000** iterations, seed `RANDOM_SEED` = **20260907**.

### One match

$$H \sim \mathrm{Poisson}(s\,\lambda_h), \qquad A \sim \mathrm{Poisson}(s\,\lambda_a)$$

with $s$ = `scale` (1 in normal time) and $\lambda$ from §3 evaluated at the
**current** ratings and the two club ids. The draws are independent, matching
the model the probabilities come from.

### Hot Elo

Ratings update inside each iteration after every simulated match, so a club
that over- or under-performs early carries that into its later fixtures. The
update is §1's, with two differences visible in `simulate_match`: $K$ is always
`K_EUROPEAN` = 25, and the goal-difference multiplier takes its default
(World Football Elo) rule rather than the fitted parameters object.

The caller's ratings dict is never mutated — `simulate_league_phase` copies it
per iteration (`live = dict(ratings)`).

Matches already played enter the table at their actual scores and are never
re-simulated; the ratings passed in already reflect them.

### Ranking

`tiebreakers.rank_table()` applies Article 18 criteria (a)-(h) in order: goal
difference, goals scored, away goals, wins, away wins, then the collective
points, collective goal difference and collective goals of a club's league-phase
opponents. Criteria (i) disciplinary points and (j) the Annex D club
coefficient are **not simulatable** from scorelines, so a tie surviving (h) is
reported as unresolved and never silently broken. Across 50,000 iterations at
the 2026-27 cutoff, **0** ties survived.

### Two-legged ties

`play_tie()`. Aggregate over the two legs; the side with the **lower seeding
position hosts the second leg** in every round (Article 19.02/19.03/19.04). If
the aggregate is level, extra time is played at the second leg's venue with the
rates scaled by

$$s = \tfrac{1}{3}$$

(`EXTRA_TIME_SCALE`)

— thirty minutes against ninety. If still level, penalties.

### Penalties

A shoot-out is close to a coin flip with a small rating tilt:

$$P(\text{A wins}) = \mathrm{clip}\Big(0.5 + \tau\,(R_A - R_B),\ 0.35,\ 0.65\Big),
\qquad \tau = 4\times10^{-4}$$

$\tau$ is `PENALTY_TILT`.

with `PENALTY_TILT_CLIP = (0.35, 0.65)`. A 100-Elo edge is worth 4 percentage
points, and no tilt can exceed 65/35 whatever the gap.

### The final

`play_final()`: a single match at a neutral venue (`neutral=True`, so $h$ is
dropped from the home rate), then extra time, then penalties. 2027-06-05 at the
Riyadh Air Metropolitano, Madrid.

### The bracket

`bracket.draw_bracket()` randomises every choice UEFA makes by draw, within the
Article 19 constraints, **once per iteration**. Play-off bands pair
9/10 v 23/24, 11/12 v 21/22, 13/14 v 19/20 and 15/16 v 17/18; the round-of-16
bands send seeds 1/2 against a winner from 15/16/17/18, 3/4 from 13/14/19/20,
5/6 from 11/12/21/22 and 7/8 from 9/10/23/24. The two clubs of a seeded pair are
placed in opposite halves, so they can only meet in the final.

Because the draw is re-made every iteration, a bracket **slot** has no
persistent occupant across iterations, and the only reportable quantity is
$P(A \text{ meets } B \text{ in round } R)$. Annex B — which fixes exactly which
play-off winner each seeded pair meets, and the quarter and semi tree — is a
diagram in the regulations PDF with no extractable text, so the structure above
was derived from the observed 2024-25 and 2025-26 brackets, which agree exactly,
and every simulated draw is asserted legal under it.

---

## 7. Evaluation metrics

`src/ucl/evaluate.py` — `log_loss()`, `rps()`, `brier()`, `calibration()`.

All take 1X2 triples in the order (home, draw, away) and integer outcomes
$y \in \\{0, 1, 2\\}$, so the model and the bookmaker go through identical code.
Rows are **normalised first, then clipped** to `EPSILON` = $10^{-15}$, then
renormalised: clipping to 1 beforehand would turn an unnormalised row like
$(2,1,1)$ into a uniform forecast.

### Log-loss

$$\mathrm{LL} = -\frac{1}{n}\sum_{i=1}^{n} \log p_{i, y_i}
\qquad\text{weighted:}\qquad
\mathrm{LL} = \frac{\sum_i w_i \big({-\log p_{i,y_i}}\big)}{\sum_i w_i}$$

The uniform baseline is $\log 3 = 1.0986$. This is the project's primary
objective.

### Ranked probability score

1X2 is ordered home > draw > away, so predicting an away win when the home team
wins must cost more than predicting a draw. With $r = 3$ outcomes and $o$ the
one-hot realisation:

$$\mathrm{RPS} = \frac{1}{r-1}\sum_{k=1}^{r-1}
\left(\sum_{j\le k} p_j - \sum_{j\le k} o_j\right)^{2}$$

averaged over matches. The uniform baseline is 0.2222.

### Brier

Multi-class, summed over the three outcomes:

$$\mathrm{BS} = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{3}\big(p_{ij} - o_{ij}\big)^{2}$$

### Calibration

`calibration()` pools all $3n$ predicted probabilities into `bins` = 10 equal
bins and reports mean predicted against observed frequency per bin. The
expected calibration error is

$$\mathrm{ECE} = \frac{\sum_b n_b\,\big|\bar p_b - \bar o_b\big|}{3n}$$

— note the denominator counts probabilities, not matches, because the three
outcomes of a match each contribute a point.

---

## 8. Inference

`src/ucl/significance.py` — `paired_differences()`, `_resample()`,
`block_bootstrap()`, `cluster_bootstrap()`, `two_way_bootstrap()`,
`diebold_mariano()`, `newey_west_lags()`.

### Paired differences

Candidates are compared **match by match on identical matches**, not by
comparing two aggregate scores (`paired_differences`, which raises unless the
two models cover exactly the same `match_id` set):

$$d_i = \ell_i^{\text{model}} - \ell_i^{\text{reference}}$$

Negative favours the model. The primary statistic is the weighted mean, with
European matches weighted `EUROPEAN_WEIGHT` = **3**:

$$\bar d = \frac{\sum_i w_i d_i}{\sum_i w_i}$$

A candidate is paired against **the accepted model of the day**, not against
Elo: measuring a candidate that already contains an accepted term against Elo
alone would credit it with that term's gain as well as its own.

### Cluster bootstrap

The design point that drives the statistics: **ratings are frozen within a
season**, refit at each season's own cutoff and then held, so every match a club
plays in a season carries the same rating error. The dependence is per *club*.

`_resample()` draws $G$ clusters with replacement from the $G$ observed
clusters and recomputes the weighted mean over the drawn clusters, keeping
their full membership:

$$\bar d^{(b)} = \frac{\sum_{g \in \mathcal{S}_b} \sum_{i \in g} w_i d_i}
{\sum_{g \in \mathcal{S}_b} \sum_{i \in g} w_i}$$

over `draws` = **10,000** resamples, seed 20260907, reporting the percentile
interval $[q_{2.5}, q_{97.5}]$. Three clusterings are used:

| scheme | unit | count |
|---|---|---|
| block | $(\text{season}, \text{ISO week})$ | 155 |
| one-way cluster | $(\text{season}, \text{home club})$ | 1,309 |
| two-way | both margins at once | 1,309 × 1,308 |

A block cannot straddle a refit, since the ratings are rebuilt at each cutoff.

### Two-way (pigeonhole) cluster bootstrap

The one-way scheme treats a club's away matches as independent of its home
matches, which the frozen ratings make false: the same rating error travels with
the club at both ends of a fixture. The pigeonhole bootstrap
(McCullagh 2000; the resampling analogue of the Cameron-Gelbach-Miller two-way
variance) resamples the home clusters and the away clusters **independently** and
keeps each match with multiplicity equal to the product of how often its two
clusters were drawn:

$$w_i^{(b)} = w_i \cdot m^{H(b)}_{g(i)} \cdot m^{A(b)}_{k(i)}, \qquad
\bar d^{(b)} = \frac{\sum_i w_i^{(b)} d_i}{\sum_i w_i^{(b)}}$$

where $m^{H(b)}$ and $m^{A(b)}$ are multinomial counts from the two draws. It is
the widest of the three by construction, because it admits dependence along both
margins at once. A draw in which no match survives both resamples is discarded.

**This is the standing acceptance rule from milestone 8 onward.** Measured:

| comparison | mean $\bar d$ | weeks (155) | one-way (1,309) | **two-way** |
|---|---|---|---|---|
| v1 − elo | −0.000991 | [−0.001775, −0.000179] | [−0.001877, −0.000089] | **[−0.002394, +0.000455]** |
| v2 − v1 | −0.000424 | [−0.000736, −0.000123] | [−0.000765, −0.000086] | **[−0.001006, +0.000163]** |

The two-way interval is 35-60% wider in every comparison, and neither
acceptance excludes zero under it. Both were made under the pre-declared
one-way rule and were **not** revisited, because choosing the rule after seeing
the result is not inference. The honest claim is: *small, consistent in sign,
not robustly significant.* Full table in `results/significance/comparisons.csv`.

### Sign test

With four scored folds and the null that the sign of the difference in each
fold is a coin flip, all four favouring the newer model gives a one-sided

$$p = \left(\tfrac{1}{2}\right)^{4} = \tfrac{1}{16} = 0.0625$$

This one is arithmetic, not code: it is **reported** in
`results/significance/report.md`, not implemented in `significance.py`. Per-fold
means come from `per_season()`.

### Diebold-Mariano

`diebold_mariano()`, a robustness check and not a gate. The loss differential is
averaged **within each date** first — the test treats its input as a time series
and hundreds of matches share a date, so on the raw per-match series a "lag"
counts matches rather than time. With $\bar d$ the mean of the $n$ daily means
and a Newey-West long-run variance using Bartlett weights:

$$\hat\sigma^2 = \gamma_0 + 2\sum_{l=1}^{L}\left(1 - \frac{l}{L+1}\right)\gamma_l,
\qquad \gamma_l = \frac{1}{n}\sum_{t} \tilde d_t \tilde d_{t-l}$$

$$\mathrm{DM} = \frac{\bar d}{\sqrt{\hat\sigma^2 / n}}
\ \sim\ \mathcal{N}(0,1) \text{ under the null}$$

The lag truncation is Newey and West's 1994 plug-in rule, automatic rather than
tuned, because with the lag chosen to suit the answer the p-value would mean
nothing (`newey_west_lags`):

$$L = \left\lfloor 4\left(\frac{n}{100}\right)^{2/9}\right\rfloor$$

At $n = 930$ daily means this gives $L = 6$. For v2 − v1 the daily test does not
reject ($p = 0.495$), though it tests a smaller quantity with much less
precision than the bootstrap.

---

## 9. De-vigging

`src/ucl/backtest.py` — `devig()`. `src/ucl/data/odds.py` — `load_outright()`.

Bookmaker prices sum to more than 1. Both the match-odds and outright paths use
**proportional normalisation**: with decimal odds $o_k$,

$$\pi_k = \frac{1/o_k}{\sum_j 1/o_j}, \qquad
\text{overround} = \sum_j \frac{1}{o_j}$$

Match odds use the closing **average** across bookmakers
(`AvgCH/AvgCD/AvgCA`, `backtest.MARKET_COLUMNS["avg"]`, the default). Pinnacle
is sharper but covers only 32% of 2025-26, so the average is the like-for-like
benchmark across seasons; Pinnacle and Bet365 are selectable alternatives.

The 2026-27 outright snapshot (maltafootball.com, 2026-08-18) prices 27 of 36
clubs at an overround of **1.3078**.

**Proportional de-vigging is crude, and the write-ups say so.** It assumes the
margin is spread evenly across prices, which ignores **favourite-longshot
bias** — the margin is concentrated on longshots, so proportional normalisation
understates favourites. This was measured rather than assumed: a second market
(an exchange, read from publicly available pre-cutoff reporting) agrees with the
bookmaker within a few percent on favourites and differs about twofold on
longshots. Arsenal is **1.83×** the exchange's raw price, not the 2.10× the
bookmaker's de-vigged price implies. The exchange prices only 13 clubs, below
the pre-declared minimum of 15, so it supplements the benchmark rather than
replacing it.

The market is a **diagnostic only** and never enters any objective.

---

## 10. Rejected methods

Each was tested once under the pre-declared protocol — forward chaining over
2021-22…2025-26, each fold trained only on earlier seasons, the test season
scored once — and archived with its folds, paired losses, decision JSON and a
README, rejections included.

### Dixon-Coles

`src/ucl/models/dixon_coles.py` — `fit_dixon_coles()`, `tau()`,
`decay_weights()`, `score_matrix()`.

The same linear predictor plus per-club attack and defence deviations, so the
comparison isolates exactly those additions:

$$\log \lambda_h = \mu + \eta_{\text{home}} + \beta d + \alpha_i - \delta_j,
\qquad
\log \lambda_a = \mu - \beta d + \alpha_j - \delta_i$$

with the Dixon-Coles correction applied to the four low-score cells:

$$\tau(i,j) = \begin{cases}
1 - \lambda_h \lambda_a \rho & i = 0, j = 0 \\
1 + \lambda_h \rho & i = 0, j = 1 \\
1 + \lambda_a \rho & i = 1, j = 0 \\
1 - \rho & i = 1, j = 1 \\
1 & \text{otherwise}
\end{cases}$$

so that $P(i,j) = \tau(i,j)\,\mathrm{Pois}(i;\lambda_h)\,\mathrm{Pois}(j;\lambda_a)$,
floored at `TAU_FLOOR` = $10^{-6}$ to keep the log-likelihood finite, with
$\rho \in [-0.25, 0.25]$.

Fitted on a time-decayed weighted log-likelihood,

$$w_t = 2^{-\,\mathrm{age}(t)/\text{half-life}}$$

plus a ridge penalty $\varsigma \sum_c (\alpha_c^2 + \delta_c^2)$. Adding a
constant to every $\alpha$ and every $\delta$ leaves every rate unchanged, so
the model is translation-invariant and **the ridge is what identifies it**,
centring both at zero — no sum-to-zero constraint is needed. The ridge is also
the fallback to Elo: a club's deviation survives by roughly

$$\frac{n_c}{n_c + \varsigma}$$

for $n_c$ its decay-weighted match count, and at zero the rate is exactly
$\mu + \eta + \beta d$, the Elo-only baseline. At the 2026-27 cutoff the median
club has under 5 effective matches at a one-year half-life, so most clubs sit
near that fallback by construction, with no special-casing for thin records.

**Result: null.** Pooled $\Delta$ log-loss **+0.00030**, cluster CI
$[-0.00086, +0.00140]$. The code stays for later.

### Market-implied ratings

`src/ucl/models/market.py` — `fit_market_ratings()`, `_fit_one_league()`.

The bookmaker's closing price is the one input that encodes what informed people
knew and the results do not. It is turned into a rating by fitting a Davidson
model — the same three-way form as §2 — but whose **observations are the
market's own probability vectors** rather than outcomes, by weighted
cross-entropy:

$$\mathcal{L}(\theta) = -\sum_i w_i \sum_{k \in \\{H,D,A\\}}
\pi_{ik} \log P_k(\theta)$$

with $\pi$ the de-vigged market probabilities of §9 and
$w_i = 2^{-\mathrm{age}_i/\text{half-life}}$ exponentially decayed, half-life
tuned per fold on the training seasons — a flat average over five years of odds
would discard precisely what makes the market informative. Strengths are
mean-centred for identifiability, the optimiser works in log-sum-exp throughout,
and analytic gradients are supplied. A league needs `MIN_MATCHES` = 50 priced
matches and a club `MIN_CLUB_MATCHES` = 5 effective matches, or it falls back to
Elo only.

Odds compare clubs **within** a league and say nothing about how leagues rank
against each other, so each league is fitted alone. Two mappings were tried:

- **`scale="z"`**, standardised within league, covariate $z_h - z_a$ — how far
  above its own league's average each club sits.
  **Rejected: significantly worse.** European-only $\Delta$ **+0.00232**, cluster
  CI $[+0.00022, +0.00448]$, excluding zero on the wrong side. Standardising
  deletes league quality, which in a cross-league European tie is exactly the
  information that matters.
- **`scale="elo"`**, mapped into Elo points by regressing pre-cutoff Elo on
  market strength within each league, keeping the league's Elo mean and so its
  quality. Chosen after seeing the first result, so **post hoc**:
  $\Delta$ +0.00031, CI $[-0.00172, +0.00235]$ — **null**.

That failure is the direct ancestor of §5's design: the GAP ratings face the
same cross-league problem and take the *residual* on Elo rather than the
z-score, precisely so league quality stays where European results have
calibrated it.

### The rest

| candidate | form | pooled $\Delta$ | cluster 95% CI | decision |
|---|---|---|---|---|
| Elo parameters ($K$, $K_{\text{eur}}$, GD exponent, carryover) | §1 | −0.00040 | [−0.00126, +0.00049] | null, defaults kept |
| Fixture congestion (rest days, short-rest flag) | per-side covariate, §3 | +0.00005 | [−0.00001, +0.00010] | null |
| Squad value at 22-25% coverage | $\log V_h - \log V_a$ | −0.00031 | [−0.00069, +0.00008] | null |
| Squad value at ~50% coverage | same, after a name-mapping fix | **−0.00099** | [−0.00187, −0.00008] | **accepted → v1** |
| GAP shot ratings, as the Elo residual | §5 | **−0.00042** | [−0.00076, −0.00008] | **accepted → v2** |
| Minutes-weighted squad value | $v = \sum_p \frac{\text{min}_p}{\sum_q \text{min}_q} V_p$ | −0.00001 | [−0.00047, +0.00044] | null |

Negative is better. The minutes-weighted null is about the weighting and nothing
else: coverage was identical, and the two measures rank clubs at Spearman
0.9894, because a club that can afford an expensive squad also gives those
players the minutes.

**No xG.** No maintained free source covers the backtest window;
football-data has begun shipping xG columns but only 353 matches carry them at
the cutoff, which is useless for a five-season forward chain.

---

## Reproducing the numbers

```bash
# Ratings, offsets and their diagnostics at a cutoff
.venv/bin/python -c "from datetime import date; from ucl.models.elo import elo_as_of; \
    print(elo_as_of(date(2026,9,7)).diagnostics)"

# Elo plus the baseline Poisson model, with the diagnostics that say whether
# the ratings are believable before anything is simulated
.venv/bin/python scripts/fit.py --season 2026-27

# The published model is assembled and fitted inside the simulator
.venv/bin/python scripts/simulate.py --season 2026-27 --model v2 -n 50000 --seed 20260907

# The inference of §8
.venv/bin/python scripts/significance.py

.venv/bin/python -m pytest          # 679 tests
.venv-app/bin/python -m pytest      # 730, adds the Altair and AppTest modules
```

Archived fits and decisions live under `results/`: `elo_baseline_v0/`,
`model_v1/`, `model_v2/`, `candidate_*/` and `significance/`.
