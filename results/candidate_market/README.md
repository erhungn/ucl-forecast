# Candidate 1: market-implied domestic ratings - REJECTED

Tested against `elo-baseline-v0` at commit `82e8508e7dbc8755adb03622ee03b16dc623564b`.

Within-league club strengths fitted by weighted cross-entropy against de-vigged
`AvgC*` closing odds, exponentially decayed with the half-life chosen per fold
on the training seasons, then standardised within each league. The covariate is
`d_market = z_home - z_away`; cross-league scale stays entirely with `d_elo`.

**In a cross-league European tie `d_market` is the gap between two clubs'
standing within their own leagues** - how far above its own league average each
club sits - and nothing about how those leagues compare. That is all bookmaker
odds can support, since they only ever price clubs against others in the same
competition.

## Primary test: European matches only

Pre-declared, because on domestic matches the covariate is built from the very
odds that priced the outcome being scored.

    pooled: 1,832 matches in 405 clusters, delta +0.00254
      cluster   95% CI [+0.00075, +0.00435]  excludes zero, WRONG direction
      per-match 95% CI [+0.00085, +0.00414]  excludes zero, WRONG direction

The candidate is significantly **worse** where it is tested honestly, and
significantly better where the comparison is circular (domestic -0.00071,
[-0.00130, -0.00011]). That contrast is the result.

## Two structural limits found while running it

**Coverage caps at 47.5%** of European matches: more than half involve a club
from a league the cache does not price (Czechia, Slovakia, Ukraine, Azerbaijan,
Cyprus, Israel), so the covariate is zero for them and they fall back to Elo. Of
the 36 league-phase teams, PSG, Atletico, Sporting and Sabah are among the
unrated.

**Free half-life selection destroys the covariate.** Extending the grid below 61
days, every fold picked 21 days - where no league clears the minimum match count,
no club is rated, and `d_market` is identically zero. The search chose to switch
the covariate off, and every delta came back exactly 0.00000. The grid now starts
at 61 days, where coverage is real, and the degeneracy is reported rather than
searched.

## Fitted coefficients at the 2026-27 cutoff (half-life 61d)

    a=+0.1697  h=+0.2339  b=+0.7205  b_d_market=+0.0246

`b_elo` falls from the baseline's 0.7483 to 0.7205, so the covariate does absorb
signal - it is simply signal that does not transfer across leagues.
