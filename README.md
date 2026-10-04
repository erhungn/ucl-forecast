# UCL 2026-27 Forecasting Model

A pre-tournament forecast of the 2026-27 UEFA Champions League from a frozen
information set: an Elo rating system with cross-league calibration, a Poisson
match model, and 50,000 Monte Carlo tournaments played through the actual UEFA
rules. Candidate improvements were evaluated by forward chaining over five
seasons against a pre-declared primary test with a cluster bootstrap. Seven
candidates were tried: five returned nulls, one was significantly worse than the
baseline, and one - a pre-season squad-value covariate - was accepted and became
the final model, `model-v1`.

Cutoff: **2026-09-07**, the day before Matchday 1.

Outputs, for all 36 clubs and all 144 league-phase fixtures:

- match level - home/draw/away probabilities and the scoreline distribution;
- team level - P(top 8), P(play-off, 9-24), P(eliminated, 25-36), and
  P(reach R16 / QF / SF / final / champion).

Headline numbers (`results/model_v1/forecast_2026-27.csv`):

| club | P(top 8) | P(champion) | market |
|---|---|---|---|
| Arsenal | 0.790 | 0.229 | 0.109 |
| Bayern Munich | 0.736 | 0.180 | 0.109 |
| Barcelona | 0.581 | 0.109 | 0.102 |
| Manchester City | 0.497 | 0.089 | 0.090 |
| Paris Saint-Germain | 0.434 | 0.066 | 0.127 |
| Real Madrid | 0.465 | 0.065 | 0.096 |

The market column is a de-vigged pre-season outright snapshot, used **only** as a
diagnostic - it never enters any objective. See
[Known limitations](#known-limitations) before reading anything into the gaps.

## Reproduce

The raw data payloads are gitignored, so the cache is rebuilt first. The three
fetch scripts under `scripts/colab/` are self-contained and were written to be
run from Google Colab, because football-data.co.uk is DNS-blocked on the
development network:

```bash
python scripts/colab/fetch_initial.py --out data/raw/football_data           # 73 files
python scripts/colab/fetch_history.py --out data/raw/football_data_history   # 55 files
python scripts/colab/fetch_current.py --season 2627 --prune-stale            # weekly refresh
```

Each writes a `manifest.json` beside the files recording source URL, UTC
timestamp, SHA-256 and size; the loaders verify those hashes on every load and
never fetch. European results, fixtures and squad values are fetched by
`scripts/fetch_data.py` and by hand from Kaggle (see
[Data sources](#data-sources-and-the-network-workarounds)).

Then:

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.lock.txt -e .
.venv/bin/python -m pytest                                       # 546 tests, ~60s

.venv/bin/python scripts/fit.py --season 2026-27                 # ratings + match model
.venv/bin/python scripts/simulate.py --season 2026-27 -n 50000   # the forecast, ~5 min
.venv/bin/python scripts/simulate.py --through-matchday 1        # in-season update
.venv/bin/python scripts/evaluate_matchday.py --matchday 1       # score a played matchday
```

Validation and the candidate tests (each re-runs the whole forward chain):

```bash
.venv/bin/python scripts/tune_elo.py                        # ~10 min
.venv/bin/python scripts/tune_dixon_coles.py
.venv/bin/python scripts/tune_candidate.py squad_value      # also: market, market_elo, congestion
```

Data-integrity utilities: `coverage_check.py`, `check_names.py`,
`build_fixtures.py`, `build_transfermarkt_map.py`, `validate_wikipedia.py`.

`--model elo -n 2000` reproduces the archived `elo-baseline-v0` forecast
bit-identically. Python >=3.11; numpy 2.5.2, pandas 2.3.3, scipy 1.18.1, all
pinned.

## Data sources, and the network workarounds

Two primary sources were unreachable from the development network. Neither was
worked around silently.

| source | used for | how |
|---|---|---|
| football-data.co.uk | 18 domestic leagues and closing odds | **unreachable.** Fetched from Colab by the scripts in `scripts/colab/` into three cache directories, each with its own `manifest.json`. Loaders are **cache-only**, verify every hash, and refuse a filename that appears in two caches. |
| ClubElo API | ratings at a date | **source-side outage.** Replaced by an Elo system replayed from results; `data/clubelo.py` remains as an optional cross-check. |
| openfootball (GitHub) | UCL/UEL/UECL results | reachable; Football.TXT parser, with dates resolved against each file's declared window (group-ordered files otherwise push later groups into the wrong year). |
| Wikipedia | European results openfootball lacks; 2026-27 fixtures | wikitext parser, validated against openfootball before use: **342 of 342 matches identical** on teams, date, score, a.e.t. and penalties (`scripts/validate_wikipedia.py`). |
| UEFA regulations (PDF) | Article 18 tiebreakers, Article 19 bracket rules | the PDF has no text layer in the usual sense; the text was recovered by decoding its glyph encoding (ToUnicode CMaps plus the standard glyph order). Articles 18 and 19 were read that way and are quoted in the code. **Annex B was never readable** - it is a diagram - so the bracket structure was derived from the observed 2024-25 and 2025-26 brackets, which agree exactly, and the tests assert the result is a legal draw. |
| Kaggle `davidcariboo/player-scores` | Transfermarkt squad values | manual fetch, cache-only with a manifest. Frozen: see limitations. |
| maltafootball.com | outright odds, 2026-08-18 | 27 of 36 clubs priced, overround 1.3078, de-vigged by proportional normalisation. |

Domestic coverage is 18 leagues: 11 "main" leagues with one file per division
and season, held from **2016-17** onward, and 7 "extra" leagues whose single
file per country reaches back to **2012**.

Club names across five sources are reconciled by one explicit mapping table
(`data/raw/team_names.csv`), built by **exact matching after normalisation only**
- no fuzzy or substring matching - with the remainder reviewed by hand. Three
standing checks run over it: an unmapped second spelling of a mapped club (A),
one club resolving to two ids (B), and two names merged into one id appearing on
opposite sides of the same match (C). They have caught real splits
(Bayern München / Bayern Munich, Monaco, Vaduz) that would otherwise have been
fitted as separate clubs.

## Leakage discipline

The output of the project is an out-of-sample number, so no-look-ahead is
enforced structurally rather than by care.

- There is no global cutoff read inside loaders. A frozen `SeasonConfig` per
  season carries its own `cutoff_date`, and **every loader takes the cutoff as an
  explicit argument**.
- `assert_no_lookahead` fails on a row dated **on or after** the cutoff (not
  merely after it); `filter_to_cutoff` is the only way data enters a fit.
- Association offsets are **refitted at each season's own cutoff** - reusing one
  set across folds would leak the future without touching the replay at all.
- `tests/test_leakage.py` asserts that no match on or after the cutoff reaches
  any of the three fitted stages: the offset MLE, the Elo replay, the match model.
- Squad membership is reconstructed from *appearances*, never from
  `players.csv.current_club_id`, which reflects today and would leak every
  transfer since. A test asserts an earlier as-of is unmoved by later data.

## Ratings: Elo plus association offsets

Elo is replayed over every cached domestic and European match, club by club:
K = 20 domestic / 25 European, home advantage 65 Elo points, classic World
Football Elo goal-difference weighting, and 0.90 regression toward the league
mean between seasons. The replay starts in 2012 as burn-in, though only the
seven extra leagues exist that early; the main leagues join in 2016-17. Home
advantage is applied **once**, in the Elo update: the match model carries it
separately in its own intercept, and applying it in both would count it twice.

Domestic leagues are disconnected graphs - a club only ever plays its own league
- so the ratings are league-relative and not comparable across borders. European
ties are the only edges between them, so an **association-level strength offset**
is fitted by a Davidson model (MLE in Elo points, with a draw parameter) on all
pre-cutoff European matches, and added to starting ratings. That is what makes
"Bodø/Glimt vs Barcelona" meaningful.

## Model v1

```
d       = (elo_home - elo_away) / 400
d_value = log(value_home) - log(value_away)        # 0 unless both clubs are valued

log(lambda_home) = a + h + b*d + b_value*d_value
log(lambda_away) = a     - b*d - b_value*d_value
```

| a | h | b | b_value |
|---|---|---|---|
| 0.16947 | 0.23396 | 0.69698 | **0.03987** |

Poisson MLE on 65,389 pre-cutoff matches at **pre-match** ratings. Squad value is
a pre-season prior: the snapshot is 6 July of the season's starting year, the
staleness the frozen source imposes. Where either club is unvalued the covariate
is zero for that match - the convention the backtest scored, so the forecast is
never better informed than the test that accepted it. Squad value shifts the goal
rates only; the Elo update inside the simulation uses Elo alone.

## Simulator

50,000 Monte Carlo iterations, seed 20260907, byte-identical output on a re-run.
Each iteration plays the 144 fixtures with ratings updating **inside** the
iteration ("hot"), ranks the single table, draws the knockout bracket within the
UEFA constraints, and plays through to a champion. Extra time scales the rates to
30 minutes; shoot-outs are near coin flips with a small rating tilt.

All ten Article 18 tiebreakers are implemented in order. Two of them -
disciplinary points and the Annex D club coefficient - cannot be derived from
simulated scorelines, so a tie surviving the eighth criterion is **reported as
unresolved, never silently broken** (0 per iteration in the current run).
Tiebreakers and bracket were test-covered before the simulator was trusted.

## Validation protocol

Five seasons, 2021-22 to 2025-26, by **forward chaining**: each fold tunes on the
seasons strictly before its test season - including the choice between model
variants - and scores that season **once**. 19,188 matches, of which 1,832 are
European.

The primary test was **pre-declared** for every experiment and is the only thing
that decides acceptance: x3-weighted pooled log-loss (European matches weighted
up), frozen mode, with a **cluster bootstrap over (season, home club)**. A
mis-rated club biases all of its matches the same way, so per-match resampling
understates uncertainty; the per-match interval is reported beside it as a
sensitivity. Walk-forward mode, pure-European scoring and domestic-only scoring
are sensitivities too. **A change is accepted only if the pooled 95% CI excludes
zero.**

Two implementation defects were found *after* a test had returned a null. Both
times the fix was treated as a defect rather than a new specification, and the
re-run was declared **binding in advance** - accept if the CI excludes zero,
otherwise closed for good - with the baseline side of the comparison verified
bit-identical to the original run, so that only the covariate could have moved.

## Everything that was tried

| # | change | primary test | pooled delta log-loss | cluster 95% CI | decision |
|---|---|---|---|---|---|
| - | Elo parameters (K, K_eur, GD exponent, carryover) | weighted pooled | -0.00040 | [-0.00126, +0.00049] | **null** - defaults kept |
| - | Dixon-Coles (tau, ridge att/def, time decay) | weighted pooled | +0.00030 | [-0.00086, +0.00140] | **null** |
| 1 | market-implied ratings, standardised per league | European only | **+0.00232** | **[+0.00022, +0.00448]** | **rejected - significantly worse** |
| 1b | market-implied ratings mapped into Elo units (post hoc) | European only | +0.00031 | [-0.00172, +0.00235] | **null** |
| 4 | fixture congestion (rest days, short-rest flag) | weighted pooled | +0.00005 | [-0.00001, +0.00010] | **null** |
| 2 | squad market value, 22-25% coverage | weighted pooled | -0.00031 | [-0.00069, +0.00008] | **null** |
| 2' | squad market value, ~50% coverage (binding re-run) | weighted pooled | **-0.00099** | **[-0.00187, -0.00008]** | **accepted -> v1** |

Negative is better. Each row is archived under `results/` with its folds, paired
losses, decision JSON and a README - including the rejections.

Three things in that table are worth more than the accepted row:

- **Candidate 1 is significantly *worse*.** Standardising market-implied strength
  within each league removes league quality from the covariate, which then
  actively misinforms cross-league matches.
- **Candidate 1b's all-matches number was -0.00081 [-0.00152, -0.00011]**, which
  excludes zero. Its pre-declared primary was European-only, where it is a null,
  so it stays a null. Pre-declaration only means something when it costs
  something.
- **Pure European log-loss is a bad objective**: tuning on it is significantly
  worse out of sample (+0.00285, [+0.00150, +0.00430]). ~500 European matches a
  season is too thin; the domestic sample carries the statistical power.

Candidate 2 reached acceptance only after two defects in the *feature* were
fixed, both found by printing the covariate and inspecting it rather than the
test statistic: a club-name mapping that matched 110 of 796 Transfermarkt clubs
(now 439, by deterministic routes, with checks A/B/C), and national-team
appearances counted as club membership - the 2026 World Cup ended eight days
before the 2026-27 as-of, so 16.3% of players had been assigned to their country,
leaving PSG at EUR 406m instead of EUR 1,362m. A null from a covariate is only
evidence once the covariate itself has been looked at.

## Weekly update

```bash
python scripts/colab/fetch_current.py --season 2627 --prune-stale   # from Colab
.venv/bin/python scripts/weekly_update.py                           # everything else
```

`weekly_update.py` is the whole cycle in one command: it checks the caches,
refreshes the league-phase page once, detects the last **complete** matchday,
re-forecasts through it, scores what was played, scores every candidate model
on the played matches, and publishes to `live/<season>/`, which is the only
directory the app reads. It refuses rather
than guesses - mid-round, on a skipped round, on a cache collision (unless
`--prune-stale`), or when a verification check fails - and it does no work at
all when the inputs are unchanged.

The information set is an **as-of date**, not a hard-coded cutoff: all cached
matches before it are applied, played UCL matches are fixed at their actual
scores, and every forecast prints what each source actually knew:

```
source                         status                  cached_to    used_to
domestic/football_data         present                 2026-09-21   2026-09-21
european/wikipedia             present                 2026-09-10   2026-09-10
squad_value/transfermarkt      frozen (source paused)  2026-06-28   2026-07-06
```

Results refresh weekly (UCL from the Wikipedia league-phase page; domestic
through the Colab fetch). **Squad value cannot be refreshed** - its `used_to`
stays at 2026-07-06 while the others advance, visibly, in every run.

## Local app

A single-user Streamlit app over the published forecasts, built mobile-first: a
home screen with the state of the season, the title race, the next matchday's
1X2 probabilities, movement across matchdays, the played matches scored against
a uniform baseline, and the information set with the update button.

Probabilities are drawn as cards with horizontal stacked bars rather than wide
tables, because a table is the one Streamlit layout that does not reflow on a
phone. Each fixture card also carries the expected goals, P(over 2.5) and the
favourite's chance of winning by three or more: the modal scoreline is 1-0, 1-1
or 2-0 for most fixtures however lopsided they are, so on its own it hides the
difference between a close game and a rout. Those three come from closed forms
on the same Poisson rates - the sum of two Poissons for the total, the Skellam
law for the margin - and are tested against sums over the scoreline matrix. Clubs are identified by three-letter codes on badges in their own kit
colours - no crests or logos. Every badge is checked to WCAG AA (4.5:1) by test,
and because a dozen of these clubs play in near-identical reds and yellows, two
badges in one card switch the away side to an outline whenever the colours are
too close to tell apart.

```bash
python3 -m venv .venv-app
.venv-app/bin/python -m pip install -r requirements.app.lock.txt -e .
scripts/run_app.sh            # http://localhost:8501
scripts/run_app.sh --lan      # also reachable from a phone on the same Wi-Fi
```

Streamlit lives in its own `.venv-app` so `requirements.lock.txt` stays exactly
the model's environment; `weekly_update.py` always runs the pipeline with
`.venv/bin/python` whichever interpreter serves the UI. The app only reads
`live/` and `results/model_v1/`.

The theme is dark, set in `.streamlit/config.toml`, which overrides the
viewer's own light/dark preference by design. Those colours mirror
`src/ucl/webapp/palette.py`, and a test asserts the two agree so the chrome and
the hand-drawn cards cannot drift apart.

The suite runs in both environments: `.venv` covers everything but the Altair
and AppTest modules, which skip there and run under
`.venv-app/bin/python -m pytest` - 573 tests in all.

**`--lan` serves the app unauthenticated to every device on the network.** The
five pages give nothing away, but the update button starts a five-minute job
and, with its prune box ticked, deletes cached raw data. Put an `update_token`
in `.streamlit/secrets.toml` (gitignored) and the button goes behind it; omit
`--lan` and the app is localhost-only.

## Known limitations

1. **The model sits above the market on four clubs, for two different
   reasons.** Sporting CP (2.49x the bookmaker's de-vigged P(champion)) and
   Porto (1.96x) fit a known Elo pathology: a club that dominates a weak league
   accumulates rating against weak opposition, and the association offset
   corrects the league's level but not the within-league inflation. Arsenal and
   Bayern Munich do **not** fit that explanation - both play strong domestic
   leagues - and the cause there **is not established**, though part of it was
   the benchmark: see (2).
2. **Favourite-longshot bias in the benchmark is now measured, not assumed.** A
   second market - Kalshi prices observed 2026-08-13, read from a public article
   (`scripts/compare_markets.py`) - lets the two be compared where both quote a
   club. They agree within a few percent on the favourites (PSG 0.99, Real
   Madrid 1.00, Manchester City 0.98 as a ratio of raw implied probabilities)
   and diverge sharply on the longshots, where the bookmaker's raw prices are
   roughly twice the exchange's (Aston Villa 0.51, Napoli 0.51, Dortmund 0.61).
   The bookmaker's 30.8% margin is concentrated in the longshots, so
   **proportional de-vigging - which assumes it is spread evenly - removes too
   little from the favourites.** Against raw prices rather than de-vigged ones,
   Arsenal reads 1.83x the exchange and 1.61x the bookmaker, not 2.10x. Some of
   gap (1) was the de-vig method.
   The exchange is **too thin to replace the benchmark**: of 29 clubs listed, 16
   are quoted only as "under 1 cent" and 7 of the 36 are absent, leaving 13
   usable prices against a pre-declared minimum of 15. The maltafootball
   snapshot stays the headline benchmark, with its 9 unpriced clubs holding ~1%
   of model probability, so every de-vigged number there reads ~1% high.
3. **Squad values are frozen at 2026-07-06** and cannot be refreshed while the
   source is paused. For 2026-27 the feature describes end-of-2025-26 squads: it
   knows nothing about the summer window.
4. **The accepted covariate covers about half the data.** 48-57% of European
   matches in the backtest had both clubs valued (79.2% of the 144 league-phase
   fixtures, 32 of 36 clubs). The rest fall back to Elo, so the measured effect is
   diluted, and the acceptance should be read as "at half coverage". LASK and
   Viking have no appearances in the membership window; Sabah and Slovan
   Bratislava are absent from the source entirely.
5. **The accepted effect is small and the interval is tight to zero** (upper
   bound -0.00008). It is stable across bootstrap draws and seeds, but it is one
   covariate clearing one pre-declared bar, not a large improvement.
6. **Two Article 18 criteria are not simulatable** (cards, Annex D coefficient),
   so a surviving tie is reported rather than resolved.
7. **Rest days are computed from the matches held in the cache** - 18 leagues
   plus European competition. A club playing an uncached domestic cup tie looks
   rested, which biases the congestion covariate toward zero and makes that null
   weaker evidence than the others.
8. **No xG.** No maintained free source covers the backtest window;
   football-data has begun shipping xG columns, but only 353 matches carry them
   at the cutoff.

## Layout

```
src/ucl/
  config.py          SeasonConfig, cutoffs, Elo constants, seed
  data/              cache-only loaders, name mapping, cutoff guards, manifests
  models/            elo.py, poisson.py, covariates.py (v1), dixon_coles.py, market.py
  sim/               league_phase.py, tiebreakers.py, bracket.py, knockout.py
  backtest.py        forward chaining, paired losses, cluster bootstrap
  evaluate.py        log-loss, RPS, Brier, calibration
  webapp/            logic behind the local app: palette, clubs, cards, charts
                     (no Streamlit imports but one)
scripts/             16 entry points: fetch, fit, simulate, backtest, tune_*, build_*,
                     check_*, weekly_update
scripts/colab/       the three football-data fetch scripts, run outside this network
app/                 Streamlit entry point and the six views
tests/               573 tests (546 in .venv, all of them in .venv-app)
results/             one archived folder per experiment, accepted or rejected
live/<season>/       what the weekly update publishes, and all the app reads
data/raw/            hand-fetched caches with manifests (payloads gitignored)
```

Tags: `elo-baseline-v0` (the Elo-only reference), `model-v1` (final).

## How this was built

Specification, design decisions, data decisions and review by **Cem Erhun
Günay**, with Claude Code as the implementation agent. The validation protocol,
the pre-declared tests and the binding re-runs were set by the author; the
milestone-by-milestone working rules are in `CLAUDE.md`.
