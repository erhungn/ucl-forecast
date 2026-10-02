# Candidate 1a: market-implied ratings, within-league standardised - REJECTED

Tested against `elo-baseline-v0`. Superseded in part: the first run of this
candidate was invalid.

## A bug that invalidated the first result

`fit_market_ratings` keyed its strengths by raw **football-data names** while
`difference()` looked them up by **canonical club id**. Only 14 of 202 names
differ - but they include Man City, Man United, Atletico Madrid, Real Betis,
Borussia Dortmund, VfB Stuttgart, Paris SG and Sporting CP, which is to say most
of the clubs that actually reach Europe. Their covariate was silently zero.

European coverage per cutoff, before and after the fix:

    2021-22  43.4% -> 55.6%      2024-25  46.0% -> 60.8%
    2022-23  49.3% -> 62.2%      2025-26  44.4% -> 60.8%
    2023-24  48.6% -> 64.1%

## Corrected result, primary test on European matches only

    1,832 matches in 405 clusters, delta +0.00232
      cluster   95% CI [+0.00022, +0.00448]  excludes zero, WRONG direction
      per-match 95% CI [+0.00036, +0.00424]  excludes zero, WRONG direction

Still significantly worse, and still better on the circular domestic comparison
(-0.00100, [-0.00164, -0.00036]). Fixing the bug moved the estimate from
+0.00254 to +0.00232 - it did not change the verdict.

## Why it fails

Standardising within a league deletes league quality by construction, so a club
that dominates a weak league gets the same covariate as one that dominates a
strong one. Bodo/Glimt scores +2.57 and Bayern +2.74 while Elo puts them 221
points apart. Candidate 1b replaces the standardisation.
