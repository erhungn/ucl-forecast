# Candidate 2: squad market value - REJECTED (null, and the closest call)

Transfermarkt (Kaggle `davidcariboo/player-scores`), read cache-only from
`data/raw/transfermarkt/` with a manifest.

## Staleness, mirrored deliberately

The source froze on 2026-07-06: appearances end 2026-06-28, valuations
2026-06-12, and there are no 2026-27 squads. So for 2026-27 the feature can only
describe the squad as it stood at the end of 2025-26. Every season is held to the
same staleness - the as-of date is **6 July of the season's starting year**, not
the September cutoff - so the backtest measures what the feature can actually
deliver today. It is a **pre-season squad-quality prior and cannot be refreshed
weekly while the source is paused.**

Squad membership comes from appearances in the 180 days before the as-of date,
never from `players.csv.current_club_id`, which reflects today and would leak
every transfer since.

## Primary test: x3-weighted pooled log-loss

    19,188 matches in 1,309 clusters, delta -0.00031
      cluster   95% CI [-0.00069, +0.00008]  includes zero
      per-match 95% CI [-0.00073, +0.00011]  includes zero

A null - but the closest any candidate has come. The point estimate is negative,
the interval nearly excludes zero, the most recent fold is individually
significant (2025-26: -0.00099, [-0.00167, -0.00033]), and the European-only
sensitivity leans the same way (-0.00120, [-0.00255, +0.00016]).
Fitted `b_d_value = +0.033`.

## The caveat that matters

**European coverage is only 21.8%-24.9%.** Transfermarkt's club table holds 796
top-division clubs, of which 110 map to our ids by exact normalisation, so three
quarters of European matches have no covariate and fall back to Elo. Ten of the
36 league-phase teams are unmapped: AEK Athens, Bodo/Glimt, Feyenoord, Inter,
Real Betis, Roma, Sabah, Slavia Praha, Slovan Bratislava, Viking. Six have an
obvious Transfermarkt counterpart awaiting review (Bodo/Glimt ->
"Fotballklubben Bodø/Glimt", Feyenoord -> "Feyenoord Rotterdam", Inter ->
"Inter Milan", Real Betis -> "Real Betis Balompié", Slavia Praha -> "SK Slavia
Prague"); four have no entry at all.

Raising coverage could plausibly tip this one. That is a real caveat to the
stopping rule, not an argument against it.
