# Model v3 — built, gated, **not published**

Model v2 plus candidate C: **a per-competition European goal scale and home
advantage.** This directory is the frozen pre-tournament archive the promotion
needs, produced and verified ahead of the switch. It is not live.

- Fit: identical to candidate C's accepted fit, under the published name.
  `ucl.models.forecast.fit` routes `"v3"` and `"v2+C"` to the same function
  with the same arguments, and `tests/test_promotion.py` asserts that.
- Cutoff: 2026-09-07. Simulations: 50,000 iterations, seed 20260907.
- Produced: 2026-10-23. **`config.MODEL_HISTORY` still ends at `v2`**, so
  nothing reads this archive yet and `weekly_update.py` refuses to publish v3
  into `live/` until that line changes.

## The model

    d       = (elo_home - elo_away) / 400
    d_value = log(value_home) - log(value_away)
    d_gap_m = Elo-residual GAP shot ratings, per measure m

    log(lambda_home) = a + h + b*d + b_value*d_value + sum_m b_m*d_gap_m
                       + c_level + (0 if neutral else g_level)
    log(lambda_away) = a     - b*d - b_value*d_value - sum_m b_m*d_gap_m
                       + c_level

Three competition levels: **domestic** is the baseline and carries neither new
term; **ucl_main** is the Champions League main phase; **other_european** is
UEL, UECL and UCL qualifying. `c` is a goal scale added to both log-rates, `g`
an extra home advantage added to the home side only and switched off by
`neutral` exactly as `h` is — so the neutral-venue final keeps the scale and
loses the home term, by construction rather than by a caller remembering.

| coefficient | v2 | **v3** |
|---|---|---|
| `a` | 0.16951 | 0.16989 |
| `h` | 0.23373 | **0.22654** |
| `b` | 0.69818 | 0.69746 |
| `b_d_value` | +0.04004 | +0.04005 |
| `b_d_gap_sot` | +0.02331 | +0.02308 |
| `b_d_gap_shots_off` | -0.00342 | -0.00341 |
| `b_d_gap_corners` | -0.00319 | -0.00301 |
| `c_ucl_main` | - | **+0.10404** (x1.110 on both rates) |
| `c_other_european` | - | -0.03477 |
| `g_ucl_main` | - | +0.00396 |
| `g_other_european` | - | +0.09134 |

Fitted on the same 65,389 pre-cutoff matches; log-likelihood -192,094.33
against v2's -192,131.96.

**The shared home advantage falls when the European ones are fitted beside it.**
`h` goes 0.23373 to 0.22654, and `g_other_european` is +0.091: the pooled `h`
was carrying a European average, and splitting it leaves the domestic value
lower. `g_ucl_main` is +0.004 — effectively nothing, which is the same finding
as the backtest's, where it was negative in all four folds (-0.036 to -0.010).
The main phase has no more home advantage than domestic football, and arguably
slightly less.

Per-fold terms from the accepted run, for comparison with the cutoff fit above:

| fold | `c_ucl_main` | `c_other_european` | `g_ucl_main` | `g_other_european` |
|---|---|---|---|---|
| 2022-23 | +0.1046 | -0.0305 | -0.0364 | +0.0818 |
| 2023-24 | +0.1028 | +0.0018 | -0.0392 | +0.0555 |
| 2024-25 | +0.0928 | -0.0139 | -0.0240 | +0.0739 |
| 2025-26 | +0.0964 | -0.0230 | -0.0099 | +0.0863 |
| **2026-27 (this archive)** | **+0.1040** | **-0.0348** | **+0.0040** | **+0.0913** |

## Why it was accepted

Pre-declared before the run: **European-only exact Poisson scoreline log-loss**,
two-way cluster bootstrap over (season, home club) x (season, away club), the
standing rule since milestone 8.

| | delta | two-way 95% CI | clears zero |
|---|---|---|---|
| **European scoreline (primary)** | **-0.00961** | **[-0.01892, -0.00087]** | **yes** |
| European 1X2 (declared guard) | -0.00240 | [-0.00622, +0.00138] | no |
| domestic scoreline (non-worsening) | +0.000024 | [-0.000102, +0.000145] | no |
| domestic 1X2 | -0.000081 | [-0.000146, -0.000018] | - |

1,832 European matches, 405 home clusters. The guard passes on its point
estimate as declared; its own interval includes zero and no claim rests on it.
Full archive, including the three declared weaknesses, in
`../candidate_euro_scale/`.

**The motivating evidence overlaps the test seasons.** The 288 league-phase
matches that diagnosed the gap are 2024-25 and 2025-26, two of the four folds.
The fit is clean per fold; the choice of what to test was not independent of
them. This is recorded with the acceptance, not as a footnote to it.

## What it does to the published numbers

The 144 league-phase fixtures, at the cutoff:

| | v2 | **v3** | observed (288 matches, 2024-25 + 2025-26) |
|---|---|---|---|
| goals per match | 2.825 | **3.130** | 3.32 |
| P(over 2.5) | 0.5346 | **0.6029** | 0.646 |
| P(draw) | 0.2262 | **0.2105** | 0.149 |
| P(home) | 0.4516 | **0.4615** | ~0.514 |

Goals and over-2.5 close about two thirds of their gap; the draw rate closes
about a fifth. That split was predicted before the run and is the reason the
limitation paragraph in `docs/METHODOLOGY.md` section 3 does not go away with
this model: the rest of the draw excess belongs to the independent-Poisson
form, whose Dixon-Coles correction returned a null on the pooled sample and was
never tested on European matches alone.

**And it reaches the team-level numbers, which is the part that was disputed.**

| club | P(top 8) | P(champion) | mean position |
|---|---|---|---|
| Real Madrid | 0.483 -> **0.502** | 0.073 -> 0.073 | 10.52 -> 10.19 |
| Bayern Munich | 0.747 -> **0.760** | 0.200 -> 0.204 | 6.11 -> 5.93 |
| Arsenal | 0.786 -> **0.799** | 0.220 -> 0.230 | 5.52 -> 5.33 |
| Barcelona | 0.577 -> **0.587** | 0.104 -> 0.107 | 8.89 -> 8.68 |
| Manchester City | 0.497 -> **0.506** | 0.090 -> 0.091 | 10.35 -> 10.13 |
| Atletico Madrid | 0.145 -> **0.137** | 0.009 -> 0.007 | 19.13 -> 19.20 |

mean |change| in P(top 8) is 0.0052 and the largest is 0.0194; mean position
moves by up to 0.33 places. The direction is **systematic, not idiosyncratic**:
the correlation between a club's change in P(top 8) and its v2 P(top 8) is
**+0.835**, all 4 clubs above 0.5 rise and all 22 below 0.25 fall. Fewer draws
and more goals mean less compression in points and goal difference, so the
favourites separate from the field. That is the mechanism this candidate was
reopened for, now measured rather than argued.

**It moves slightly away from the market.** Mean |model - market| over the 27
priced clubs goes 0.0183 to 0.0193, because the model was already above the
market on the favourites and this raises them further. Like v2, **v3 is not a
fix for limitation 1.**

## The gates it has passed

1. **Byte-identity, v1 and v2.** With `MODEL_HISTORY` unchanged and every new
   term absent, both published forecasts reproduce their archives exactly at
   50,000 iterations and seed 20260907:

       outputs/forecast_gate_v1.csv == results/model_v1/forecast_2026-27.csv
       outputs/forecast_gate_v2.csv == results/model_v2/forecast_2026-27.csv

   `gate_v1_v2_reproduction.log` holds both runs. v1 and v2 carry empty
   `scale`/`home_extra` dicts, so the new loops never execute for them - the
   arithmetic is untouched line for line, and the gate proves it rather than
   assuming it.
2. **All twelve simulator checks pass**, 0 unresolved league-phase ties in
   50,000 iterations. `simulate_md0.log` is the run.
3. **`model_2026-27.json` has a writer**, `scripts/write_model_json.py`, and the
   writer reproduces `../model_v2/model_2026-27.json` - an archive it did not
   write - within 1e-11. That file was assembled by hand for v1 and v2.
4. **`club_inputs.csv` is byte-identical to v2's.** The covariate *inputs* do
   not depend on the coefficients, and `scripts/export_club_inputs.py`
   regenerates the same file. Checked rather than assumed.

## A reproducibility limit, found here and not fixed here

The fit is **not bit-reproducible between processes.** `ucl.models.gap.replay`
averages the relegated clubs' ratings over a Python `set` of club names
(`gap.py:183`) and seeds the rating dict in set order (`gap.py:194`), so
`PYTHONHASHSEED` changes the floating-point summation order and every fitted
coefficient moves in about the eleventh significant digit. Two runs in one
process agree exactly; two processes agree to ~1e-11, and `PYTHONHASHSEED=0`
makes it exact, which is how the diagnosis was confirmed.

Gate 1 is the reassuring half: v2 carries the GAP terms where the noise lives
and still reproduced **byte-for-byte**, so the noise does not reach a forecast
written to six decimals from 50,000 integer counts. But the project's
reproducibility rule is stronger than what the code delivers, and the two-line
fix (`sorted(...)` in both places) would move every archived coefficient, so it
is recorded here and left for its own decision rather than folded into a
promotion.

## What promotion still needs

Everything below is in place and tested; the switch itself is one line plus one
weekly run.

- `config.MODEL_HISTORY` gains `"v3"`. `paths.ARCHIVE_DIR`,
  `paths.PREVIOUS_ARCHIVES`, `tracking.REFERENCE` and all three scripts'
  `--model` defaults derive from it.
- `scripts/weekly_update.py` then publishes through the latest complete
  matchday, moves v2's snapshots into `live/<season>/archive_v2/`, and writes
  v3's scores for the already-played matchdays **beside** the published records
  rather than over them.
- Rehearse first: `weekly_update.py --model v3 --live-dir <a copy of live/>`.
  Without `--live-dir`, publishing a model that is not in `MODEL_HISTORY`
  refuses.

## Files

`forecast_2026-27.csv`, `model_2026-27.json`, `positions_2026-27.csv`,
`knockout_paths_2026-27.csv`, `ratings_2026-27.csv`,
`remaining_fixtures_2026-27.csv`, `next_matchday_1_2026-27.csv`,
`club_inputs.csv`, `simulate_md0.log`, `gate_v1_v2_reproduction.log`.
