"""Is v2's out-of-sample improvement over v1 distinguishable from noise?

Analysis only. Nothing here fits, promotes or publishes a model; it re-scores
the models the backtest already produced and asks what the spread of the
per-match differences supports.

Three things about the design matter for the statistics:

* **the ratings are frozen within a season.** They are refit at each season's
  own cutoff and then held, so every match a club plays in a season carries the
  same rating error. That dependence is per *club*, not per week, which is why
  the backtest's own test is a cluster bootstrap over (season, home club).
* **a block bootstrap over calendar weeks** handles same-day and short-run
  serial dependence instead. It is the test asked for here, and it is reported
  beside the cluster bootstrap rather than in place of it, because the two
  address different dependence.
* **the primary objective weights European matches x3**, so the weighted mean
  is what reproduces the archived deltas. The unweighted mean is reported too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "MODELS", "EUROPEAN_WEIGHT", "per_match_losses", "paired_differences",
    "weighted_mean", "block_bootstrap", "cluster_bootstrap",
    "two_way_bootstrap", "per_season", "standardised_coefficients",
    "diebold_mariano", "newey_west_lags",
]

#: Analysis label -> the backtest candidate that produces it. ``None`` is the
#: plain Elo baseline.
MODELS: dict[str, str | None] = {
    "elo": None,
    "v1": "squad_value",
    "v2": "gap",
    "v2+B": "gap_minutes",
    "v2_sot_only": "gap_sot",
}
EUROPEAN_WEIGHT = 3.0


def per_match_losses(seasons: tuple[str, ...], models: tuple[str, ...]
                     ) -> pd.DataFrame:
    """Out-of-sample log-loss per match per model, on identical matches.

    Columns: ``match_id, date, season, competition, home, away, model,
    log_loss, weight``.
    The matches, the folds and the refit schedule are the backtest's own - this
    only reads the losses off the scored rows.
    """
    from ucl.backtest import (
        DEFAULT_PARAMS, _losses_from_rows, _match_losses, score_season,
        score_season_covariate,
    )

    pieces = []
    for season in seasons:
        index = None
        for label in models:
            candidate = MODELS[label]
            if candidate is None:
                rows = _match_losses(
                    score_season(season, DEFAULT_PARAMS, modes=("frozen",)))
                losses = rows.loss.to_numpy()
            else:
                rows = score_season_covariate(season, candidate, modes=("frozen",))
                rows = rows[rows["mode"] == "frozen"].reset_index(drop=True)
                losses = _losses_from_rows(rows)
            identifiers = (season + "/" + rows.date.astype(str) + "/"
                           + rows.home.astype(str) + "/" + rows.away.astype(str))
            if index is None:
                index = identifiers.to_numpy()
            elif not (identifiers.to_numpy() == index).all():
                raise ValueError(
                    f"{season}: {label} scored a different set of matches than "
                    f"{models[0]}; the comparison would not be paired")
            pieces.append(pd.DataFrame({
                "match_id": identifiers.to_numpy(),
                "date": pd.to_datetime(rows.date).to_numpy(),
                "season": season,
                "competition": rows.competition.to_numpy(),
                # Carried, not parsed back out of match_id: a club called
                # "Bodo/Glimt" makes splitting on "/" quietly wrong.
                "home": rows.home.to_numpy(),
                "away": rows.away.to_numpy(),
                "model": label,
                "log_loss": losses,
                "weight": np.where(rows.competition.to_numpy() != "domestic",
                                   EUROPEAN_WEIGHT, 1.0),
            }))
    return pd.concat(pieces, ignore_index=True)


def paired_differences(losses: pd.DataFrame, model: str, reference: str
                       ) -> pd.DataFrame:
    """``d_i = loss(model) - loss(reference)`` on the matches both scored."""
    left = losses[losses.model == model].set_index("match_id")
    right = losses[losses.model == reference].set_index("match_id")
    shared = left.index.intersection(right.index)
    if len(shared) != len(left) or len(shared) != len(right):
        raise ValueError(f"{model} and {reference} do not cover the same matches")
    out = left.loc[shared, ["date", "season", "competition", "home", "away",
                            "weight"]].copy()
    out["d"] = left.loc[shared, "log_loss"] - right.loc[shared, "log_loss"]
    out["block"] = (out.season + "/W"
                    + out.date.dt.isocalendar().week.astype(str).str.zfill(2))
    out["cluster"] = out.season + "/" + out.home.astype(str)
    out["home_cluster"] = out.cluster
    out["away_cluster"] = out.season + "/" + out.away.astype(str)
    return out.sort_values("date").reset_index()


def weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float((values * weights).sum() / weights.sum())


def _resample(frame: pd.DataFrame, key: str, draws: int, seed: int,
              weighted: bool) -> dict:
    groups = [group for _, group in frame.groupby(key, sort=True)]
    values = [group.d.to_numpy() for group in groups]
    weights = [group.weight.to_numpy() if weighted else np.ones(len(group))
               for group in groups]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    count = len(groups)
    for draw in range(draws):
        picked = rng.integers(0, count, count)
        numerator = sum(float((values[i] * weights[i]).sum()) for i in picked)
        denominator = sum(float(weights[i].sum()) for i in picked)
        samples[draw] = numerator / denominator
    observed = weighted_mean(frame.d, frame.weight if weighted
                             else np.ones(len(frame)))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "mean": observed,
        "low": float(low),
        "high": float(high),
        "standard_error": float(samples.std(ddof=1)),
        "share_below_zero": float((samples < 0).mean()),
        "excludes_zero": bool(low * high > 0),
        "blocks": count,
        "n": int(len(frame)),
        "draws": draws,
    }


def block_bootstrap(differences: pd.DataFrame, *, draws: int = 10_000,
                    seed: int = 20260907, weighted: bool = True) -> dict:
    """Resample whole calendar weeks, not matches.

    Matches on the same day and in the same week share rating state and
    schedule effects, so resampling matches independently would understate the
    spread. The block is ``(season, ISO week)``, which cannot straddle a refit:
    the ratings are rebuilt at each season's cutoff.
    """
    return _resample(differences, "block", draws, seed, weighted)


def cluster_bootstrap(differences: pd.DataFrame, *, draws: int = 10_000,
                      seed: int = 20260907, weighted: bool = True) -> dict:
    """Resample ``(season, home club)`` clusters - the backtest's own test.

    Reported beside the block bootstrap because the dependence that actually
    dominates here is per club: a mis-rated club carries the same error through
    every match it plays in a season, whatever week they fall in.
    """
    return _resample(differences, "cluster", draws, seed, weighted)


def two_way_bootstrap(differences: pd.DataFrame, *, draws: int = 10_000,
                      seed: int = 20260907, weighted: bool = True) -> dict:
    """Pigeonhole bootstrap over (season, home club) and (season, away club).

    The one-way cluster bootstrap treats a club's home matches as dependent and
    its away matches as independent of them, which is not true: the frozen
    rating error travels with the club whichever end of the fixture it is. The
    pigeonhole scheme (McCullagh 2000) resamples the home clusters and the away
    clusters independently and keeps each match with multiplicity equal to the
    product of how often its two clusters were drawn - the resampling analogue
    of the Cameron-Gelbach-Miller two-way variance.

    It is the widest of the three intervals by construction, because it admits
    dependence along both margins at once.
    """
    home_codes, home_index = pd.factorize(differences.home_cluster, sort=True)
    away_codes, away_index = pd.factorize(differences.away_cluster, sort=True)
    values = differences.d.to_numpy(dtype=float)
    base = (differences.weight.to_numpy(dtype=float) if weighted
            else np.ones(len(differences)))
    home_count, away_count = len(home_index), len(away_index)

    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    kept = 0
    for draw in range(draws):
        home_multiplicity = np.bincount(
            rng.integers(0, home_count, home_count), minlength=home_count)
        away_multiplicity = np.bincount(
            rng.integers(0, away_count, away_count), minlength=away_count)
        weights = base * (home_multiplicity[home_codes]
                          * away_multiplicity[away_codes])
        total = weights.sum()
        if total <= 0:                      # no match survived both draws
            samples[draw] = np.nan
            continue
        kept += 1
        samples[draw] = float((values * weights).sum() / total)

    samples = samples[~np.isnan(samples)]
    observed = weighted_mean(values, base)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "mean": observed,
        "low": float(low),
        "high": float(high),
        "standard_error": float(samples.std(ddof=1)),
        "share_below_zero": float((samples < 0).mean()),
        "excludes_zero": bool(low * high > 0),
        "home_clusters": home_count,
        "away_clusters": away_count,
        "n": int(len(differences)),
        "draws": int(len(samples)),
    }


def per_season(differences: pd.DataFrame) -> pd.DataFrame:
    """Mean d per season, weighted and unweighted, with the match counts."""
    rows = []
    for season, group in differences.groupby("season"):
        european = group[group.competition != "domestic"]
        rows.append({
            "season": season,
            "n": len(group),
            "european": len(european),
            "mean_d": weighted_mean(group.d, group.weight),
            "unweighted_mean_d": float(group.d.mean()),
            "domestic_mean_d": float(group[group.competition == "domestic"].d.mean()),
            "european_mean_d": float(european.d.mean()) if len(european) else float("nan"),
        })
    return pd.DataFrame(rows)


def standardised_coefficients(seasons: tuple[str, ...]) -> pd.DataFrame:
    """Each v2 coefficient times its covariate's standard deviation.

    The raw coefficients are not comparable: ``d_value`` is a log ratio of
    euros, the GAP terms are differences in shots. Multiplying by the spread of
    the covariate on the fitting sample puts them in one unit - the change in
    log expected goals produced by a one-standard-deviation covariate - so the
    measures can be ranked by how much work they actually do. The Elo term is
    included for scale.
    """
    from ucl.backtest import _fitted_gap
    from ucl.config import SEASONS
    from ucl.models.covariates import value_covariate
    from ucl.models.forecast import GAP_COLUMNS, gap_covariates

    rows = []
    for season in seasons:
        cutoff = SEASONS[season].cutoff_date
        ratings, values, residuals, _, model = _fitted_gap(cutoff, season)
        frame = ratings.pre_match
        home = frame.home_id.to_numpy()
        away = frame.away_id.to_numpy()
        coefficients = model.as_dict()

        columns = {"d_value": value_covariate(values).difference(home, away)}
        for covariate in gap_covariates(residuals):
            columns[covariate.name] = covariate.difference(home, away)
        columns["d_elo"] = (frame.elo_home.to_numpy(dtype=float)
                            - frame.elo_away.to_numpy(dtype=float)) / 400.0

        for name, series in columns.items():
            key = "b" if name == "d_elo" else f"b_{name}"
            coefficient = coefficients.get(key)
            if coefficient is None:
                continue
            spread = float(np.nanstd(series, ddof=1))
            rows.append({
                "season": season, "covariate": name,
                "coefficient": float(coefficient), "covariate_sd": spread,
                "standardised": float(coefficient) * spread,
                "share_nonzero": float(np.mean(np.asarray(series) != 0.0)),
            })
    return pd.DataFrame(rows)


def newey_west_lags(n: int) -> int:
    """Newey and West's 1994 plug-in rule, ``floor(4 (n/100)^(2/9))``.

    An automatic choice rather than a tuned one: with the lag picked to suit the
    answer, the p-value would mean nothing.
    """
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def diebold_mariano(differences: pd.DataFrame, *, lags: int | None = None,
                    daily: bool = True) -> dict:
    """Diebold-Mariano on the loss differential, with a Newey-West variance.

    ``daily`` averages the differences within each date first. That matters:
    the test treats its input as a time series, and hundreds of matches share
    each date, so on the raw per-match series a "lag" counts matches rather
    than time and the autocorrelation it corrects for is mostly cross-sectional.
    The daily series is the defensible one; the raw series is reported too.
    """
    frame = differences.sort_values("date")
    if daily:
        series = frame.groupby(frame.date.dt.date).d.mean().to_numpy()
    else:
        series = frame.d.to_numpy()
    n = len(series)
    if n < 10:
        raise ValueError(f"too few observations for Diebold-Mariano: {n}")
    if lags is None:
        lags = newey_west_lags(n)

    centred = series - series.mean()
    variance = float((centred ** 2).mean())
    for lag in range(1, lags + 1):
        covariance = float((centred[lag:] * centred[:-lag]).mean())
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    variance = max(variance, 1e-30)
    statistic = float(series.mean() / np.sqrt(variance / n))

    from scipy.stats import norm

    return {
        "mean": float(series.mean()),
        "observations": n,
        "lags": int(lags),
        "statistic": statistic,
        "p_value": float(2.0 * (1.0 - norm.cdf(abs(statistic)))),
        "basis": "daily means" if daily else "per match",
    }
