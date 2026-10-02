# Candidate 1b: market ratings mapped into Elo units - REJECTED (null)

**Post hoc**: declared after candidate 1a failed, and labelled as such in the
tuner's output and in `candidate_market_elo_decision.json` (`post_hoc: true`).

Same market strengths, but instead of standardising within each league, each
league's Elo is regressed on its market strength and the fitted value used. The
line passes through the league's Elo mean, so the covariate keeps league quality
and a cross-league difference is meaningful. At the 2024-25 cutoff this puts
Bodo/Glimt at 1652 Elo-equivalent against Bayern at 1948, where the standardised
version had them nearly level.

## Primary test: European matches only

    1,832 matches in 405 clusters, delta +0.00031
      cluster   95% CI [-0.00172, +0.00235]  includes zero
      per-match 95% CI [-0.00157, +0.00228]  includes zero

A null, not a harm: the Elo-units mapping removed the damage candidate 1a did
(+0.00232, significant) but produced no gain. Later folds trend better
(2024-25 -0.00039, 2025-26 -0.00376), consistent with odds coverage improving in
recent seasons, but not enough to matter.

Sensitivities: domestic -0.00117 [-0.00186, -0.00045] (circular), all matches
-0.00081 [-0.00152, -0.00011].
