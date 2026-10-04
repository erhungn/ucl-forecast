# Is v2's improvement over v1 distinguishable from noise?

Analysis only. Nothing here changes the published model, its coefficients,
`live/` or the app. Reproduce with `.venv/bin/python scripts/significance.py`.

**Headline: under a two-way cluster bootstrap, the v2-over-v1 interval includes
zero - and so does v1's own acceptance over Elo.** Only the exploratory
shots-on-target-only variant survives all three resampling schemes. Details
below.

**What was decided after this analysis** (the decisions are the author's; this
file records them so the numbers and their consequences stay together):

* **Past decisions stand.** v1 and candidate A were accepted under the
  pre-declared one-way rule. Changing them now because a stricter test
  disagrees would be choosing the rule after seeing the result. v2 remains the
  published model: the out-of-sample point estimate favours it in all four
  folds and the cost of being wrong either way is small.
* **The claims are restated** here, in the README and in the archive READMEs:
  both acceptances were made under one-way clustering, under two-way clustering
  neither interval excludes zero, and the point estimates are unchanged and
  consistent in sign. Small, consistent, **not robustly significant**.
* **Two-way clustering is the acceptance rule from here on**, recorded as a
  standing rule in CLAUDE.md.
* **The shots-on-target-only variant is not promoted.** It is post hoc, and is
  logged as an open lead that would need a fresh holdout.
* **No older season is added.** A single season's two-way standard error is
  larger than the effect in question - see below - so another fold could not
  settle it.

## Reproduction gate

Per-match out-of-sample log-loss was exported for five models over the four
scored folds - 19,188 matches each, the backtest's own matches, folds and refit
schedule. The weighted mean differences reproduce the archived deltas exactly:

| comparison | exported | archived |
|---|---|---|
| v1 - elo | -0.000990505 | -0.000990505 |
| v2 - v1 | -0.000423967 | -0.000423967 |
| (v2+B) - v2 | -0.000014807 | -0.000014807 |

The script refuses to report statistics if this fails. `per_match_losses.csv`
holds the losses; `comparisons.csv` every number below.

## The three resampling schemes, and why they differ

The ratings are **refit once per season**, at that season's cutoff, and held
frozen while the season is scored. Every match a club plays that season
therefore carries the same rating error.

1. **Weekly blocks**: `(season, ISO week)`, **155 blocks**. The refit schedule
   itself would give only four, so the finer unit is used; no block straddles a
   refit. Handles same-day and short-run serial dependence.
2. **One-way clusters**: `(season, home club)`, **1,309 clusters**. The
   backtest's own test, and the one every acceptance so far rests on.
3. **Two-way pigeonhole**: `(season, home club)` **and** `(season, away club)`,
   1,309 x 1,308. Home and away clusters are resampled independently and each
   match is kept with multiplicity equal to the product of how often its two
   clusters were drawn - the resampling analogue of the
   Cameron-Gelbach-Miller two-way variance.

Scheme 2 treats a club's away matches as independent of its home matches. The
frozen rating makes that false: a mis-rated club carries the same error at both
ends of the fixture. Scheme 3 admits both margins, and is wider by construction.

All three: 10,000 resamples, seed 20260907, weighted x3 on European matches to
match the primary objective and the gate above.

## Results

Mean d is `loss(model) - loss(reference)`; negative favours the model. A tick
means the 95% interval excludes zero.

| model vs reference | mean d | weeks (155) | one-way (1,309) | **two-way** | P(d<0) | DM p |
|---|---|---|---|---|---|---|
| **v2 vs v1** | **-0.000424** | [-0.000736, -0.000123] ✓ | [-0.000765, -0.000086] ✓ | **[-0.001006, +0.000163] ✗** | 0.997 | 0.495 |
| v1 vs elo | -0.000991 | [-0.001775, -0.000179] ✓ | [-0.001877, -0.000089] ✓ | **[-0.002394, +0.000455] ✗** | 0.991 | 0.734 |
| v2+B vs v2 | -0.000015 | [-0.000446, +0.000406] ✗ | [-0.000463, +0.000447] ✗ | [-0.000759, +0.000735] ✗ | 0.538 | 0.288 |
| v2+B vs v1 | -0.000439 | [-0.000991, +0.000096] ✗ | [-0.000993, +0.000125] ✗ | [-0.001399, +0.000502] ✗ | 0.947 | 0.264 |
| *sot-only vs v1* | *-0.000895* | *[-0.001150, -0.000641]* ✓ | *[-0.001168, -0.000630]* ✓ | ***[-0.001367, -0.000436]*** ✓ | *1.000* | *0.0015* |
| *sot-only vs v2* | *-0.000472* | *[-0.000651, -0.000294]* ✓ | *[-0.000675, -0.000278]* ✓ | ***[-0.000801, -0.000144]*** ✓ | *1.000* | *0.0003* |

The last two rows are the ablation and are exploratory - see below.

The two-way interval is 35-60% wider than the one-way one in every comparison,
and that is enough to cross zero for both v1 and v2. The point estimates do not
move: this is about how much the data supports them, not about which model
scores better.

## Diebold-Mariano

On the loss differential ordered by date, Newey-West variance, the 1994
plug-in lag rule `floor(4 (n/100)^(2/9))` - an automatic choice, because a lag
picked to suit the answer would make the p-value meaningless. Differences are
averaged within each date first: the test treats its input as a time series and
hundreds of matches share each date, so on the raw per-match series a "lag"
counts matches rather than time. The raw series is in `comparisons.csv` and
gives p = 0.021 for v2 vs v1 at 12 lags.

The daily test is much less powerful here, for measurable reasons rather than
interpretive ones. Days carry between 1 and 82 matches (median 11), so
averaging within a day and then across days gives a one-match Tuesday the same
weight as an 82-match Saturday and drops the x3 European weighting: the
quantity it tests is -0.000180, against -0.000354 unweighted per match and
-0.000424 weighted. It is not a dependence problem - the daily series'
autocorrelation is negligible (lags 1-5: +0.015, -0.047, +0.026, +0.028,
+0.006) and the statistic hardly moves with the lag (t = -0.683 at 0 lags,
-0.682 at 6, -0.864 at 30).

## Mean d per season

**v2 vs v1** - the same sign in all four folds, and the two components pull
differently in 2023-24:

| season | n | European | mean d | unweighted | domestic | European |
|---|---|---|---|---|---|---|
| 2022-23 | 4,728 | 410 | -0.000712 | -0.000695 | -0.000685 | -0.000806 |
| 2023-24 | 4,770 | 407 | -0.000200 | -0.000027 | **+0.000084** | -0.001213 |
| 2024-25 | 4,881 | 523 | -0.000383 | -0.000420 | -0.000445 | -0.000214 |
| 2025-26 | 4,809 | 492 | -0.000406 | -0.000275 | -0.000188 | -0.001043 |

**shots-on-target-only vs v2** - also the same sign in all four, but the
European component is *positive* in three of them:

| season | n | European | mean d | unweighted | domestic | European |
|---|---|---|---|---|---|---|
| 2022-23 | 4,728 | 410 | -0.000183 | -0.000188 | -0.000191 | -0.000154 |
| 2023-24 | 4,770 | 407 | -0.000595 | -0.000759 | -0.000863 | **+0.000361** |
| 2024-25 | 4,881 | 523 | -0.000645 | -0.000824 | -0.000946 | **+0.000189** |
| 2025-26 | 4,809 | 492 | -0.000450 | -0.000606 | -0.000711 | **+0.000312** |

So dropping shots off target and corners helps domestic matches consistently
and costs a little in Europe, consistently. Because the objective weights
European matches x3, the weighted gain is smaller than the unweighted one.

## Standardised coefficients

Coefficient x the covariate's standard deviation on the fitting sample: the
change in log expected goals from a one-standard-deviation covariate, which is
the only way to compare a log ratio of euros with a difference in shots.
`share_nonzero` is the fraction of fitting matches where the covariate is not
zeroed for missing coverage.

| cutoff | `d_elo` | `d_value` | `d_gap_sot` | `d_gap_shots_off` | `d_gap_corners` |
|---|---|---|---|---|---|
| 2022-23 | 0.2473 | 0.0444 | 0.0169 | -0.0066 | +0.0003 |
| 2023-24 | 0.2518 | 0.0466 | 0.0164 | -0.0075 | -0.0031 |
| 2024-25 | 0.2578 | 0.0403 | 0.0198 | -0.0105 | -0.0036 |
| 2025-26 | 0.2589 | 0.0377 | 0.0151 | -0.0072 | -0.0017 |
| 2026-27 | 0.2571 | 0.0375 | 0.0142 | -0.0054 | -0.0026 |
| share_nonzero (2026-27) | 1.00 | 0.42 | 0.56 | 0.56 | 0.56 |

Elo does an order of magnitude more work than anything else: about 17 times
shots on target and 7 times squad value. Shots off target is consistently
negative at roughly 40% of the magnitude of shots on target, which is what the
ablation then confirms; corners are near zero at every cutoff. Note also that
squad value is zeroed on 58% of fitting matches and the shot terms on 44%.

## What one more season would buy

Not run, and on these numbers not worth running. The two-way bootstrap applied
to each fold on its own, for v2 against v1:

| season | n | home x away clusters | mean d | two-way SE | 95% CI |
|---|---|---|---|---|---|
| 2022-23 | 4,728 | 332 x 331 | -0.000712 | 0.000590 | [-0.001866, +0.000463] |
| 2023-24 | 4,770 | 325 x 325 | -0.000200 | 0.000584 | [-0.001353, +0.000955] |
| 2024-25 | 4,881 | 327 x 327 | -0.000383 | 0.000625 | [-0.001631, +0.000841] |
| 2025-26 | 4,809 | 325 x 325 | -0.000406 | 0.000540 | [-0.001470, +0.000662] |
| **all four** | **19,188** | **1,309 x 1,308** | **-0.000424** | **0.000296** | **[-0.001006, +0.000163]** |

A single season gives a two-way standard error of about **0.00059 - roughly 1.4
times the effect being measured**. Pooling the four gives 0.000296, or 0.7
times the effect; excluding zero at 95% needs it below about 0.000216.
Adding one more season would take the standard error to about 0.000265,
still short, and scaling as one over the root of the fold count puts the
requirement at **roughly eight comparable seasons** - double the current
window. One extra fold cannot settle this.

## The sign of the effect

The weighted mean favours v2 in **all four** folds. Under a null of no
difference each fold is a coin flip, so that is a one-sided p of 1/16 =
**0.0625**: supportive of a real effect, consistent across folds, and not below
the conventional threshold on its own.

## Older seasons: what is available, and what adding one would require

**Available.** European results exist for **2016-17 to 2020-21** and domestic
results from 2016-17 (main leagues) and 2012 (extra leagues). Squad values
resolve at a 2020-07-06 as-of (225 clubs). None of these five seasons is in
`config.SEASONS`, so none has ever been scored.

Taking 2020-21 as the candidate unseen fold, adding it would require:

1. **A `SeasonConfig` entry**, with `tournament_format="group_stage"` - no
   obstacle in itself, since 2021-22 to 2023-24 are already group-stage and are
   scored. Its cutoff is not a September date: the 2020-21 European season began
   **2020-10-20**, and 2019-20 ran to **2020-08-23**, so the usual "day before
   matchday 1" rule lands in late October and the preceding season's European
   matches finish unusually late.
2. **A fixtures file, or a config change.** Every configured season currently
   carries one and a test asserts it exists, although `_evaluation_frames` never
   reads fixtures. A backtest-only season would be cleaner as an explicit flag
   than as a fabricated fixtures file.
3. **Name-mapping review.** The intersection gate must pass for clubs appearing
   in 2016-17 to 2020-21 European matches whose domestic leagues are cached;
   clubs that appear only in those seasons have never been through
   `check_names.py`.

And three asymmetries that would have to be stated, because they make such a
fold not quite comparable with the four already scored:

4. **A third of the European matches.** UECL did not exist before 2021-22, so
   those seasons carry ~330 European matches each against 816-961 from 2021-22
   on. The primary test weights European matches x3, so the European component
   of the objective would be about a third the size.
5. **Half the burn-in.** A 2020-21 cutoff sees 4 main-league seasons of history
   against 7 for 2022-23 and 11 for 2026-27, and shot coverage before it is
   45.2% against 57% overall. This is the same objection that moved the
   backtest window forward in milestone 5.
6. **COVID.** Home advantage was materially different in exactly these seasons:
   the home win rate across the cached main leagues is 0.4725, 0.4577, 0.4578,
   0.4469 for 2016-17 to 2019-20, then **0.4009 in 2020-21**, recovering to
   0.4290 and 0.4575. A model with a single fitted home-advantage term would be
   mis-specified for that season in a way it is not for the others.

**And one protocol point.** Extending `BACKTEST_SEASONS` would redefine
`folds()` and therefore the test that accepted v1, A and B; the archived deltas
would no longer be comparable. An older season is only usable as a **holdout
scored once**, kept out of the chain, not folded back into it.

Nothing has been run.

## Conclusion

On the 19,188 scored out-of-sample matches, v2's mean log-loss is 0.000424
lower than v1's, and the sign is the same in all four folds. Whether that is
distinguishable from noise depends on what is treated as independent: resampling
weekly blocks or one-way (season, home club) clusters puts the 95% interval
entirely below zero, while the two-way pigeonhole bootstrap, which also admits
dependence through the away club, gives [-0.001006, +0.000163] and includes
zero. The same is true of v1's own acceptance over Elo, [-0.002394, +0.000455].
A Diebold-Mariano test on daily means also fails to reject, though it tests a
smaller quantity with much less precision. The minutes-weighted variant shows
no difference under any scheme, matching its rejection. Standardised, the shot
terms are the smallest in the model - Elo is about 17 times shots on target -
and shots off target carries a consistently negative coefficient about 40% the
size of shots on target's. Separately and exploratorily, a version of v2 using
only shots on target beats full v2 by 0.000472 and clears zero under all three
schemes including the two-way one, gaining on domestic matches in all four folds
while losing slightly in Europe in three of them; that comparison was chosen
after inspecting the coefficients and is not a pre-declared test.
