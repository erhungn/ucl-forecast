# Candidate 4: fixture congestion - REJECTED

Tested against `elo-baseline-v0` at commit `82e8508e7dbc8755adb03622ee03b16dc623564b`.

Days since each club's previous match, as per-side covariates: rest capped at 14
days and scaled to weeks, plus an indicator for under four days. Both fitted by
maximum likelihood; nothing tuned.

## Primary test: x3-weighted pooled log-loss, frozen

    pooled: 19,188 matches in 1,309 clusters, delta +0.00005
      cluster   95% CI [-0.00001, +0.00010]  includes zero
      per-match 95% CI [-0.00001, +0.00010]  includes zero

A null, and an unusually tight one - the confidence interval is a tenth the
width of any previous candidate's, because the covariate barely moves anything.

## Fitted coefficients at the 2026-27 cutoff

    a=+0.17741  h=+0.23382  b=+0.74521  c_rest=-0.00877  c_short_rest=+0.01592

Both congestion terms are tiny and point the wrong way: more rest very slightly
lowers a team's scoring rate, and playing on under four days' rest very slightly
raises it. With intervals this tight the honest reading is that the effect is
indistinguishable from zero, not that rest helps.

## The measurement limit

Rest is computed from the competitions in the cache - 18 domestic leagues plus
Europe - so a club playing a domestic cup tie we do not hold looks rested when it
is not. That attenuates the coefficient toward zero, so this null is weaker
evidence than the interval width alone suggests.
