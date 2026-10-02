# Candidate 2, coverage-fix re-run: squad market value - ACCEPTED

Binding re-run declared before it was run: accepted if the pooled cluster CI
excludes zero, otherwise candidate 2 closed for good. The specification was
frozen - same covariate, same 6-July staleness, same 180-day membership window,
same forward-chaining folds, same primary test. **Only the club-name mapping
changed**, which is an implementation defect of the same kind as the PSG bug in
candidate 1a.

## What changed: the mapping, and nothing else

`transfermarkt.club_map` matched exact normalised names only, reaching 110 of
796 Transfermarkt clubs. It is now built by
`scripts/build_transfermarkt_map.py` from three deterministic routes - exact
normalised names (109), identity with an alias already approved in
`team_names.csv` (38), and a schedule bridge over `(date, home goals, away
goals)` where Transfermarkt's `games.csv` and our domestic results describe the
same fixture (438). Resolution takes the most direct evidence first; 108 of the
109 exact matches and 36 of the 38 alias matches are independently confirmed by
the schedule.

Route C accepts a club when its leading counterpart has >= 5 matched fixtures
and >= 3x the runner-up. Margin over the runner-up, not share of the total:
same-date same-score collisions across 18 leagues put a noise floor under every
club, and a share rule rejected Real Betis at 96 votes to 2. The 3x threshold
was chosen by the user after review; it accepted ten clubs a 5x rule rejected
(Nurnberg, Greuther Furth, Palermo, Ajaccio, Nancy, Veria, Sp Gijon, Inverness
C, Partick, Hull).

Two guards earned their place:

* **association guard** - a name match may not cross borders. It had mapped
  Romanian "FC Rapid 1923" onto Austrian "SK Rapid", and it rejects Kuban
  Krasnodar, whose five votes point at Slask Wroclaw.
* **schedule beats alias** - where the schedule contradicts a name route, the
  matched fixtures win. "Paris FC" had been mapped onto Paris Saint-Germain
  through UEFA's short name "Paris"; ten matched fixtures say otherwise.

One reviewed decision, in `data/raw/transfermarkt_decisions.csv`: Belenenses
split in 2018 and our id carries top-division football-data data, so it belongs
to **B SAD**, which continued the professional team; **CF Os Belenenses** is a
distinct club and is left unmapped. One club is still ambiguous and left
unmapped: **Hannover 96**, 8 votes to 3, below the 3x margin.

Checks A, B and C all pass: no unmapped spelling of a mapped club, no id claimed
by two Transfermarkt clubs, and no two merged clubs meeting each other in 88,958
Transfermarkt games.

**439 of 796 clubs mapped, up from 110. 34 of 36 league-phase teams mapped, up
from 26.** Sabah and Slovan Bratislava are absent from the Transfermarkt club
table entirely and keep the Elo fallback.

## Coverage per cutoff, before and after

| season | clubs valued | European | domestic | | clubs valued | European | domestic |
|---|---|---|---|---|---|---|---|
| 2021-22 | 78 | 21.8% | 10.2% | -> | 236 | **50.7%** | 61.0% |
| 2022-23 | 78 | 22.9% | 11.2% | -> | 222 | **48.5%** | 57.8% |
| 2023-24 | 81 | 22.9% | 10.5% | -> | 234 | **51.8%** | 58.3% |
| 2024-25 | 80 | 24.9% | 10.9% | -> | 225 | **49.9%** | 55.8% |
| 2025-26 | 82 | 23.9% | 10.9% | -> | 230 | **57.4%** | 57.4% |

**This is an acceptance at about 50% European coverage, not at full coverage.**
Half the European matches still fall back to Elo, so the measured effect is what
the covariate delivers on the half it covers, diluted over the whole sample.

## Primary test: x3-weighted pooled log-loss, frozen, cluster bootstrap

    19,188 matches in 1,309 clusters, delta -0.00099
      cluster   95% CI [-0.00187, -0.00008]  excludes zero: True
      per-match 95% CI [-0.00176, -0.00026]  excludes zero: True

    DECISION: ACCEPT

Per fold: 2022-23 -0.00066, 2023-24 +0.00054, 2024-25 **-0.00334**
([-0.00498, -0.00179], individually significant), 2025-26 -0.00038. Fitted
`b_d_value` +0.034 to +0.049 across folds, +0.0395 at the 2026-27 cutoff, beside
`b_elo` 0.709.

European-only sensitivity (not the decision): -0.00226 [-0.00443, -0.00009] over
1,832 matches.

**The interval clears zero by a hair** (upper bound -0.00008), so it was checked
for bootstrap noise. It is stable: 4,000 / 20,000 / 50,000 draws give an upper
bound of -0.00008, and five other seeds at 4,000 draws give -0.00006 to
-0.00012. The conclusion does not depend on the resampling.

The baseline side of the comparison is **bit-identical** to the original
candidate-2 run (`loss_default`, maximum absolute difference exactly 0.0 over
all 19,188 rows), so nothing but the covariate moved.

## An unrelated defect found after the run, affecting 2026-27 only

Squad membership is "the club he last appeared for in the 180 days before the
as-of date". `appearances.csv` also contains **national-team** appearances, and
those club ids are not in `clubs.csv`, so a player whose last appearance was for
his country drops out of his club's total.

June 2026 was the World Cup, so at the **2026-27** as-of this hits 16.3% of
players and badly distorts the feature: Paris Saint-Germain 406m instead of
1,362m, Manchester City 251m instead of 1,307m, Bayern Munich 193m instead of
962m, with Porto ending up above Bayern. Mean |delta log value| 0.201, worst
club 5.2x.

**The scored seasons are untouched.** At the 2022-23 to 2025-26 as-of dates the
share is 0.4%-1.1%, the rank correlation between the two constructions is
1.0000, the largest club value moves by at most 4%, and the mean
|delta log value| is 0.000. The acceptance above therefore stands exactly as
declared; what is wrong is the **forward-looking feature**, which has not been
used for anything yet. Fixing it means restricting membership to appearances
whose club id is in `clubs.csv`, which is a change to the frozen specification
and so is left for review rather than applied here.

## Status

The stopping rule declared before candidate 1b does **not** fire: it required
both 1b and candidate 2 to fail. Candidate 2 is accepted and, by the
pre-declared protocol, becomes the baseline for any later candidate. It has not
yet been wired into the simulator or the 2026-27 forecast - and should not be
until the national-team membership defect is settled, since that is precisely
the as-of date the forecast would use.

Files: `candidate_squad_value_folds.csv`, `candidate_squad_value_paired.csv`,
`candidate_squad_value_decision.json`, `run.log`.
