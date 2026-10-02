"""Club strengths implied by de-vigged closing odds, fitted per league.

The bookmaker's closing price is the one thing in this project that encodes what
informed people knew before a match and the results do not. It is turned into a
rating by fitting a Davidson model whose *observations are the market's own
probability vectors* rather than outcomes, by cross-entropy.

Two properties matter and are deliberate:

**Recency.** Observations are exponentially decayed, half-life tuned per fold on
the training seasons. A flat average over five years of odds would discard
precisely what makes the market informative.

**Units.** Odds compare clubs within a league and say nothing about how leagues
rank against each other, so each league is fitted on its own and the resulting
strengths are standardised within that league. The covariate is therefore a
difference of within-league z-scores: *how far above its own league's average
each club is*. Cross-league scale stays entirely with the Elo covariate, which
is on the Elo scale and already carries the association offsets. The two are
never added together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = ["MarketRatings", "fit_market_ratings", "MIN_MATCHES", "ODDS_COLUMNS"]

ODDS_COLUMNS = ("avgch", "avgcd", "avgca")
#: A league needs this many priced matches before its strengths mean anything.
MIN_MATCHES = 50
#: And a club this many, or it is left out and falls back to Elo only.
MIN_CLUB_MATCHES = 5.0


@dataclass
class MarketRatings:
    """Within-league standardised strengths, one per club."""

    z: dict[str, float]
    raw: dict[str, float]
    league_of: dict[str, str]
    effective_matches: dict[str, float]
    half_life_days: float
    leagues: int = 0
    diagnostics: dict = field(default_factory=dict)

    def difference(self, home, away) -> np.ndarray:
        """z_home - z_away; 0 when either club has no market rating.

        In a cross-league European tie this is the gap between two clubs'
        standing *within their own leagues*, which is exactly what the odds can
        support. League quality is the Elo covariate's job.
        """
        home_z = np.array([self.z.get(c, 0.0) for c in np.atleast_1d(home)])
        away_z = np.array([self.z.get(c, 0.0) for c in np.atleast_1d(away)])
        known = np.array([c in self.z for c in np.atleast_1d(home)]) & np.array(
            [c in self.z for c in np.atleast_1d(away)])
        return np.where(known, home_z - away_z, 0.0)

    def coverage(self, home, away) -> float:
        known = [c in self.z for c in np.atleast_1d(home)]
        known_away = [c in self.z for c in np.atleast_1d(away)]
        return float(np.mean(np.array(known) & np.array(known_away)))


def _fit_one_league(
    home_index: np.ndarray, away_index: np.ndarray, probabilities: np.ndarray,
    weights: np.ndarray, n_clubs: int,
) -> tuple[np.ndarray, float, bool]:
    """Davidson strengths by weighted cross-entropy against market probabilities."""
    log10 = np.log(10.0)

    def objective(theta: np.ndarray):
        strength = np.concatenate([[0.0], theta[:n_clubs - 1]])
        strength = strength - strength.mean()
        advantage, log_nu = theta[n_clubs - 1], theta[n_clubs]
        u_home = (strength[home_index] + advantage) / 400.0 * log10
        u_away = strength[away_index] / 400.0 * log10
        # Log-sum-exp throughout: a wandering optimiser otherwise overflows exp.
        log_c = log_nu + 0.5 * (u_home + u_away)
        stacked = np.stack([u_home, log_c, u_away], axis=1)
        shift = stacked.max(axis=1, keepdims=True)
        log_total = shift[:, 0] + np.log(np.exp(stacked - shift).sum(axis=1))
        log_model = stacked - log_total[:, None]
        loss = -np.sum(weights[:, None] * probabilities * log_model)

        share = np.exp(log_model)
        a_share, c_share, b_share = share[:, 0], share[:, 1], share[:, 2]
        g_home = a_share + c_share / 2.0
        g_away = b_share + c_share / 2.0
        d_home = -(probabilities[:, 0] + probabilities[:, 1] / 2.0 - g_home) * weights
        d_away = -(probabilities[:, 2] + probabilities[:, 1] / 2.0 - g_away) * weights
        gradient = np.zeros_like(theta)
        full = np.bincount(home_index, d_home, n_clubs) + np.bincount(
            away_index, d_away, n_clubs)
        full = (full - full.mean()) / 400.0 * log10
        gradient[:n_clubs - 1] = full[1:]
        gradient[n_clubs - 1] = d_home.sum() / 400.0 * log10
        gradient[n_clubs] = -np.sum(weights * (probabilities[:, 1] - c_share))
        return loss, gradient

    start = np.zeros(n_clubs + 1)
    start[n_clubs - 1] = 60.0
    bounds = [(-2000.0, 2000.0)] * (n_clubs - 1) + [(-200.0, 300.0), (-3.0, 3.0)]
    result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 400})
    strength = np.concatenate([[0.0], result.x[:n_clubs - 1]])
    return strength - strength.mean(), float(result.x[n_clubs - 1]), bool(result.success)


def fit_market_ratings(
    domestic: pd.DataFrame, cutoff: date, *, half_life_days: float,
    weight_floor: float = 1e-3, scale: str = "z",
    elo: dict[str, float] | None = None,
) -> MarketRatings:
    """Fit one set of strengths per league from decayed closing odds.

    ``domestic`` is :func:`ucl.data.football_data.load_results` output, already
    filtered to before ``cutoff``; rows without ``AvgC*`` are skipped.

    ``scale`` selects the units the covariate is expressed in:

    * ``"z"`` - standardised within each league. Comparable only inside a
      league, so a cross-league difference says how far above its own league's
      average each club sits and nothing about the leagues themselves.
    * ``"elo"`` - mapped into Elo points by regressing pre-cutoff Elo on the
      market strength within each league, which keeps the league's Elo mean and
      so carries league quality. Requires ``elo``.
    """
    priced = domestic.dropna(subset=list(ODDS_COLUMNS)).copy()
    # Club ids, not raw football-data names: the covariate is looked up by
    # canonical id, and 14 of 202 names differ - including Man City, Atletico,
    # Dortmund, PSG and Sporting, i.e. most of the clubs that reach Europe.
    from ucl.data import names as _names

    lookup = _names.alias_to_canonical("football_data")
    priced["home"] = priced.home.map(lookup).fillna(priced.home)
    priced["away"] = priced.away.map(lookup).fillna(priced.away)
    age = (np.datetime64(cutoff) - priced.date.to_numpy().astype("datetime64[D]"))
    priced["weight"] = 0.5 ** (age.astype("timedelta64[D]").astype(float) / half_life_days)
    priced = priced[priced.weight >= weight_floor]

    odds = priced[list(ODDS_COLUMNS)].to_numpy(dtype=float)
    implied = 1.0 / odds
    priced[["p_home", "p_draw", "p_away"]] = implied / implied.sum(axis=1, keepdims=True)

    z: dict[str, float] = {}
    raw: dict[str, float] = {}
    league_of: dict[str, str] = {}
    effective: dict[str, float] = {}
    leagues = 0
    skipped = []

    for (country, league), group in priced.groupby(["country", "league"]):
        if len(group) < MIN_MATCHES:
            skipped.append(f"{country}/{league} ({len(group)})")
            continue
        clubs = sorted(set(group.home) | set(group.away))
        index = {club: i for i, club in enumerate(clubs)}
        home_index = group.home.map(index).to_numpy()
        away_index = group.away.map(index).to_numpy()
        weights = group.weight.to_numpy()
        probabilities = group[["p_home", "p_draw", "p_away"]].to_numpy()

        strength, _, _ = _fit_one_league(
            home_index, away_index, probabilities, weights, len(clubs))
        counts = (np.bincount(home_index, weights, len(clubs))
                  + np.bincount(away_index, weights, len(clubs)))

        keep = counts >= MIN_CLUB_MATCHES
        if keep.sum() < 4:
            skipped.append(f"{country}/{league} (only {int(keep.sum())} clubs)")
            continue
        spread = strength[keep].std(ddof=0) or 1.0
        centre = strength[keep].mean()
        leagues += 1

        slope = intercept = None
        if scale == "elo":
            # Regress this league's Elo on its market strength. The fitted line
            # passes through the league's Elo mean, so the mapped rating keeps
            # league quality while the market supplies the ordering within it.
            pairs = [(strength[i], elo[c]) for c, i in index.items()
                     if keep[i] and elo is not None and c in elo]
            if len(pairs) >= 4:
                x = np.array([p[0] for p in pairs])
                y = np.array([p[1] for p in pairs])
                variance = x.var()
                slope = float(np.cov(x, y, bias=True)[0, 1] / variance) if variance else 0.0
                intercept = float(y.mean() - slope * x.mean())

        for club, i in index.items():
            raw[club] = float(strength[i])
            league_of[club] = f"{country}/{league}"
            effective[club] = float(counts[i])
            if not keep[i]:
                continue
            if scale == "elo":
                if slope is None:
                    continue
                z[club] = (intercept + slope * float(strength[i])) / 400.0
            else:
                z[club] = float((strength[i] - centre) / spread)

    return MarketRatings(
        z=z, raw=raw, league_of=league_of, effective_matches=effective,
        half_life_days=float(half_life_days), leagues=leagues,
        diagnostics={"priced_matches": int(len(priced)), "clubs_rated": len(z),
                     "leagues_skipped": skipped},
    )
