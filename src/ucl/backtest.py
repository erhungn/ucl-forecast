"""Scoring the ratings on everything played after a season's cutoff.

For a backtest season the ratings and the match model are built from matches
strictly before that season's cutoff - including the association offsets, which
are refitted at each cutoff rather than reused - and then scored on the matches
that followed, to 30 June.

Two modes, because they answer different questions:

* ``frozen``: ratings stay as they were at the cutoff. This is the forecast's
  own use case, a frozen information set.
* ``walk_forward``: ratings update after each scored match, which is the only
  way to see how responsive the update rule is rather than how well it fitted
  history.

Domestic matches are also scored against de-vigged closing odds. The market is a
benchmark, never a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ucl import evaluate
from ucl.config import SEASONS
from ucl.data import european, football_data, names
from ucl.models.elo import (
    DEFAULT_PARAMS, EloParams, fit_association_offsets, goal_difference_multiplier,
    match_frame, replay,
)
from ucl.models.poisson import PoissonModel, fit_poisson

__all__ = [
    "MARKET_COLUMNS", "evaluation_window", "match_probabilities",
    "score_season", "devig",
]

#: Closing-odds columns, best first. Pinnacle is sharpest but covers only 32% of
#: 2025-26, so the market average is the like-for-like benchmark across seasons.
MARKET_COLUMNS = {
    "avg": ("avgch", "avgcd", "avgca"),
    "pinnacle": ("psch", "pscd", "psca"),
    "bet365": ("b365ch", "b365cd", "b365ca"),
}

MAX_GOALS = 10


def season_end(season_label: str) -> pd.Timestamp:
    return pd.Timestamp(f"{int(season_label[:4]) + 1}-06-30")


def devig(odds: np.ndarray) -> np.ndarray:
    """Proportional de-vig of decimal odds into probabilities."""
    implied = 1.0 / np.asarray(odds, dtype=float)
    return implied / implied.sum(axis=1, keepdims=True)


def match_probabilities(
    elo_home, elo_away, model: PoissonModel, *, neutral: bool = False
) -> np.ndarray:
    """P(home, draw, away) by summing the independent-Poisson scoreline matrix."""
    return probabilities_from_rates(*model.rates(elo_home, elo_away, neutral=neutral))


def probabilities_from_rates(rate_home, rate_away) -> np.ndarray:
    """The same sum, for a model whose rates are already in hand."""
    goals = np.arange(MAX_GOALS + 1)
    home_pmf = poisson.pmf(goals[None, :], np.asarray(rate_home)[:, None])
    away_pmf = poisson.pmf(goals[None, :], np.asarray(rate_away)[:, None])
    joint = home_pmf[:, :, None] * away_pmf[:, None, :]
    upper = np.triu(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), 1)
    draw = np.eye(MAX_GOALS + 1)
    probabilities = np.stack(
        [
            (joint * upper.T).sum(axis=(1, 2)),
            (joint * draw).sum(axis=(1, 2)),
            (joint * upper).sum(axis=(1, 2)),
        ],
        axis=1,
    )
    return probabilities / probabilities.sum(axis=1, keepdims=True)


@lru_cache(maxsize=64)
def _cached_offsets(cutoff: date, home_advantage_bucket: int) -> dict[str, float]:
    """Association offsets, cached in 5-Elo buckets of home advantage.

    The offsets barely move with home advantage - at 40 against 90 they
    correlate 0.9991, mean absolute difference 6.4 Elo - and refitting them for
    every trial value would dominate the tuning cost.
    """
    offsets, _ = fit_association_offsets(
        _training_frame(cutoff), home_advantage=float(home_advantage_bucket)
    )
    return offsets


@lru_cache(maxsize=8)
def _training_frame(cutoff: date) -> pd.DataFrame:
    """Matches available for fitting at ``cutoff``. Cached: it never varies."""
    return match_frame(cutoff)


@lru_cache(maxsize=8)
def _evaluation_frames(season_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Domestic and European matches played after the cutoff, to 30 June."""
    cfg = SEASONS[season_label]
    horizon = date(2027, 7, 1)
    start, end = pd.Timestamp(cfg.cutoff_date), season_end(season_label)

    domestic = football_data.load_results(horizon)
    domestic = domestic[(domestic.date >= start) & (domestic.date <= end)].copy()
    fd_lookup = names.alias_to_canonical("football_data")
    domestic["home_id"] = domestic.home.map(fd_lookup).fillna(domestic.home)
    domestic["away_id"] = domestic.away.map(fd_lookup).fillna(domestic.away)

    euro = european.load_results(horizon)
    euro = euro[(euro.date >= start) & (euro.date <= end)].copy()
    home_ids = pd.Series(index=euro.index, dtype=object)
    away_ids = pd.Series(index=euro.index, dtype=object)
    for source in euro.source.unique():
        mask = euro.source == source
        lookup = names.alias_to_canonical(source)
        home_ids[mask] = euro.loc[mask, "home"].map(lookup).fillna(euro.loc[mask, "home"])
        away_ids[mask] = euro.loc[mask, "away"].map(lookup).fillna(euro.loc[mask, "away"])
    euro["home_id"], euro["away_id"] = home_ids, away_ids
    return domestic, euro


def evaluation_window(season_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _evaluation_frames(season_label)


def _outcomes(frame: pd.DataFrame) -> np.ndarray:
    return np.where(frame.hg > frame.ag, 0, np.where(frame.hg == frame.ag, 1, 2))


@dataclass
class SeasonScore:
    season: str
    params: EloParams
    model: PoissonModel
    rows: pd.DataFrame
    summary: pd.DataFrame

    def objective(self, european_weight: float = 3.0) -> float:
        """Weighted pooled log-loss, the tuning objective."""
        frame = self.rows[self.rows["mode"] == "frozen"]
        weights = np.where(frame.competition != "domestic", european_weight, 1.0)
        return evaluate.log_loss(
            frame[["p_home", "p_draw", "p_away"]].to_numpy(), frame.outcome.to_numpy(),
            weights=weights,
        )

    def european_log_loss(self) -> float:
        frame = self.rows[(self.rows["mode"] == "frozen")
                          & (self.rows.competition != "domestic")]
        return evaluate.log_loss(
            frame[["p_home", "p_draw", "p_away"]].to_numpy(), frame.outcome.to_numpy()
        )


def score_season(
    season_label: str,
    params: EloParams = DEFAULT_PARAMS,
    *,
    market: str = "avg",
    modes: tuple[str, ...] = ("frozen", "walk_forward"),
) -> SeasonScore:
    """Build ratings at the cutoff and score everything after it."""
    cfg = SEASONS[season_label]
    training = _training_frame(cfg.cutoff_date)
    offsets = _cached_offsets(cfg.cutoff_date, int(round(params.home_advantage / 5) * 5))
    ratings = replay(training, offsets, params)
    model = fit_poisson(ratings.pre_match)

    domestic, euro = _evaluation_frames(season_label)
    pieces = []
    for mode in modes:
        for frame, competition in ((domestic, "domestic"), (euro, "european")):
            scored = _score_frame(frame, competition, ratings.ratings, model, params, mode)
            scored["mode"] = mode
            pieces.append(scored)
    rows = pd.concat(pieces, ignore_index=True)

    # The market, on the domestic matches that carry closing odds.
    columns = MARKET_COLUMNS[market]
    priced = domestic.dropna(subset=list(columns))
    market_rows = pd.DataFrame()
    if len(priced):
        probabilities = devig(priced[list(columns)].to_numpy())
        market_rows = pd.DataFrame(
            {
                "competition": "domestic",
                "mode": "market",
                "p_home": probabilities[:, 0],
                "p_draw": probabilities[:, 1],
                "p_away": probabilities[:, 2],
                "outcome": _outcomes(priced),
            }
        )
        rows = pd.concat([rows, market_rows], ignore_index=True)

    summary = []
    for (mode, competition), group in rows.groupby(["mode", "competition"]):
        stats = evaluate.summarise(
            group[["p_home", "p_draw", "p_away"]].to_numpy(),
            group.outcome.to_numpy(),
            label=f"{mode}/{competition}",
        )
        stats.update({"season": season_label, "mode": mode, "competition": competition})
        summary.append(stats)
    return SeasonScore(season_label, params, model, rows, pd.DataFrame(summary))


def _score_frame(
    frame: pd.DataFrame,
    competition: str,
    ratings: dict[str, float],
    model: PoissonModel,
    params: EloParams,
    mode: str,
) -> pd.DataFrame:
    """Probabilities for one set of matches, optionally updating as it goes."""
    frame = frame.sort_values("date")
    known = {c for c in set(frame.home_id) | set(frame.away_id) if c in ratings}
    usable = frame[frame.home_id.isin(known) & frame.away_id.isin(known)]

    if mode == "frozen":
        home = usable.home_id.map(ratings).to_numpy(dtype=float)
        away = usable.away_id.map(ratings).to_numpy(dtype=float)
        probabilities = match_probabilities(home, away, model)
    else:
        live = dict(ratings)
        collected = []
        for row in usable.itertuples():
            elo_home, elo_away = live[row.home_id], live[row.away_id]
            collected.append(match_probabilities(
                np.array([elo_home]), np.array([elo_away]), model)[0])
            expected = 1.0 / (
                1.0 + 10.0 ** (-(elo_home + params.home_advantage - elo_away) / 400.0)
            )
            actual = 1.0 if row.hg > row.ag else (0.5 if row.hg == row.ag else 0.0)
            k = params.k_european if competition == "european" else params.k_domestic
            change = (
                k
                * goal_difference_multiplier(row.hg - row.ag, params.gd_exponent)
                * (actual - expected)
            )
            live[row.home_id] = elo_home + change
            live[row.away_id] = elo_away - change
        probabilities = np.array(collected)

    return pd.DataFrame(
        {
            "date": usable.date.to_numpy(),
            "competition": competition,
            "home": usable.home_id.to_numpy(),
            "away": usable.away_id.to_numpy(),
            "p_home": probabilities[:, 0],
            "p_draw": probabilities[:, 1],
            "p_away": probabilities[:, 2],
            "outcome": _outcomes(usable),
        }
    )


# --- pooled scoring, paired differences, and bootstraps --------------------

def _match_losses(score: SeasonScore, mode: str = "frozen") -> pd.DataFrame:
    frame = score.rows[score.rows["mode"] == mode].reset_index(drop=True)
    probabilities = frame[["p_home", "p_draw", "p_away"]].to_numpy()
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    probabilities = np.clip(probabilities, 1e-15, 1.0)
    losses = -np.log(probabilities[np.arange(len(frame)), frame.outcome.to_numpy()])
    return frame.assign(loss=losses)


def paired_losses(
    season_label: str,
    params: EloParams,
    reference: EloParams,
    *,
    mode: str = "frozen",
    european_weight: float = 3.0,
) -> pd.DataFrame:
    """Per-match loss under two parameter sets, aligned, with weights and clusters.

    The cluster is ``(season, home club)``: every match belongs to exactly one,
    so no match is counted twice. Clustering matters because a mis-rated club
    biases every one of its matches the same way, which per-match resampling
    treats as independent evidence and so understates the uncertainty.
    """
    tuned = _match_losses(score_season(season_label, params, modes=(mode,)), mode)
    base = _match_losses(score_season(season_label, reference, modes=(mode,)), mode)
    if len(tuned) != len(base) or not (tuned.home.to_numpy() == base.home.to_numpy()).all():
        raise ValueError(f"{season_label}: the two scorings do not align match for match")
    return pd.DataFrame(
        {
            "season": season_label,
            "competition": tuned.competition,
            "home": tuned.home,
            "cluster": season_label + "/" + tuned.home.astype(str),
            "loss_tuned": tuned.loss,
            "loss_default": base.loss,
            "difference": tuned.loss - base.loss,
            "weight": np.where(tuned.competition != "domestic", european_weight, 1.0),
        }
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float((values * weights).sum() / weights.sum())


def bootstrap_difference(
    frame: pd.DataFrame,
    *,
    cluster: bool,
    draws: int = 4000,
    seed: int = 20260907,
) -> dict:
    """Paired bootstrap of the weighted mean difference, by match or by cluster."""
    rng = np.random.default_rng(seed)
    differences = frame.difference.to_numpy()
    weights = frame.weight.to_numpy()
    observed = weighted_mean(differences, weights)

    if cluster:
        keys = frame.cluster.to_numpy()
        groups = pd.Series(range(len(frame))).groupby(keys).apply(list)
        index = [np.asarray(v) for v in groups]
        samples = np.empty(draws)
        for draw in range(draws):
            picked = rng.integers(0, len(index), len(index))
            rows = np.concatenate([index[i] for i in picked])
            samples[draw] = weighted_mean(differences[rows], weights[rows])
    else:
        samples = np.empty(draws)
        size = len(frame)
        for draw in range(draws):
            rows = rng.integers(0, size, size)
            samples[draw] = weighted_mean(differences[rows], weights[rows])

    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "difference": observed,
        "low": float(low),
        "high": float(high),
        "excludes_zero": bool(low * high > 0),
        "clusters": int(frame.cluster.nunique()) if cluster else None,
        "n": int(len(frame)),
    }


def pooled_objective(
    seasons: tuple[str, ...],
    params: EloParams,
    *,
    european_weight: float = 3.0,
    european_only: bool = False,
) -> float:
    """Log-loss over every match of the given seasons, pooled, frozen mode.

    Matches are concatenated rather than averaged per season, so the objective
    is one likelihood over everything the training window saw.
    """
    losses, weights = [], []
    for season_label in seasons:
        frame = _match_losses(score_season(season_label, params, modes=("frozen",)))
        if european_only:
            frame = frame[frame.competition != "domestic"]
            weight = np.ones(len(frame))
        else:
            weight = np.where(frame.competition != "domestic", european_weight, 1.0)
        losses.append(frame.loss.to_numpy())
        weights.append(weight)
    return weighted_mean(np.concatenate(losses), np.concatenate(weights))


# --- Dixon-Coles scoring ---------------------------------------------------

def match_probabilities_dc(rate_home, rate_away, rho: float) -> np.ndarray:
    """P(home, draw, away) from the corrected scoreline matrix, vectorised."""
    from ucl.models.dixon_coles import TAU_FLOOR

    goals = np.arange(MAX_GOALS + 1)
    rate_home = np.asarray(rate_home, dtype=float)
    rate_away = np.asarray(rate_away, dtype=float)
    home_pmf = poisson.pmf(goals[None, :], rate_home[:, None])
    away_pmf = poisson.pmf(goals[None, :], rate_away[:, None])
    joint = home_pmf[:, :, None] * away_pmf[:, None, :]

    correction = np.ones((MAX_GOALS + 1, MAX_GOALS + 1))[None, :, :].repeat(len(rate_home), 0)
    correction[:, 0, 0] = 1.0 - rate_home * rate_away * rho
    correction[:, 0, 1] = 1.0 + rate_home * rho
    correction[:, 1, 0] = 1.0 + rate_away * rho
    correction[:, 1, 1] = 1.0 - rho
    joint = joint * np.maximum(correction, TAU_FLOOR)

    upper = np.triu(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), 1)
    draw = np.eye(MAX_GOALS + 1)
    probabilities = np.stack(
        [(joint * upper.T).sum(axis=(1, 2)),
         (joint * draw).sum(axis=(1, 2)),
         (joint * upper).sum(axis=(1, 2))], axis=1)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


@lru_cache(maxsize=512)
def _fitted_dc(cutoff: date, half_life_days: float, ridge: float):
    """Elo ratings at the cutoff plus a Dixon-Coles fit on the same frame."""
    from ucl.models.dixon_coles import fit_dixon_coles

    training = _training_frame(cutoff)
    offsets = _cached_offsets(cutoff, int(round(DEFAULT_PARAMS.home_advantage / 5) * 5))
    ratings = replay(training, offsets, DEFAULT_PARAMS)
    model = fit_dixon_coles(
        ratings.pre_match, cutoff, half_life_days=half_life_days, ridge=ridge)
    return ratings, model


def score_season_dc(
    season_label: str,
    half_life_days: float,
    ridge: float,
    *,
    modes: tuple[str, ...] = ("frozen",),
) -> pd.DataFrame:
    """Score a season's post-cutoff matches with Dixon-Coles.

    Attack and defence are frozen at the cutoff; in walk-forward mode the Elo
    ratings still update after every scored match, which is the same split the
    simulator uses.
    """
    cfg = SEASONS[season_label]
    ratings, model = _fitted_dc(cfg.cutoff_date, half_life_days, ridge)
    domestic, euro = _evaluation_frames(season_label)

    pieces = []
    for mode in modes:
        for frame, competition in ((domestic, "domestic"), (euro, "european")):
            frame = frame.sort_values("date")
            known = {c for c in set(frame.home_id) | set(frame.away_id) if c in ratings.ratings}
            usable = frame[frame.home_id.isin(known) & frame.away_id.isin(known)]
            if mode == "frozen":
                elo_home = usable.home_id.map(ratings.ratings).to_numpy(dtype=float)
                elo_away = usable.away_id.map(ratings.ratings).to_numpy(dtype=float)
                rate_home, rate_away = model.rates(
                    elo_home, elo_away, usable.home_id.to_numpy(), usable.away_id.to_numpy())
                probabilities = match_probabilities_dc(rate_home, rate_away, model.rho)
            else:
                live = dict(ratings.ratings)
                collected = []
                for row in usable.itertuples():
                    elo_home, elo_away = live[row.home_id], live[row.away_id]
                    rate_home, rate_away = model.rates(
                        [elo_home], [elo_away], [row.home_id], [row.away_id])
                    collected.append(
                        match_probabilities_dc(rate_home, rate_away, model.rho)[0])
                    expected = 1.0 / (1.0 + 10.0 ** (
                        -(elo_home + DEFAULT_PARAMS.home_advantage - elo_away) / 400.0))
                    actual = 1.0 if row.hg > row.ag else (0.5 if row.hg == row.ag else 0.0)
                    k = (DEFAULT_PARAMS.k_european if competition == "european"
                         else DEFAULT_PARAMS.k_domestic)
                    change = k * goal_difference_multiplier(
                        row.hg - row.ag, DEFAULT_PARAMS.gd_exponent) * (actual - expected)
                    live[row.home_id] = elo_home + change
                    live[row.away_id] = elo_away - change
                probabilities = np.array(collected)
            pieces.append(pd.DataFrame({
                "date": usable.date.to_numpy(), "competition": competition, "mode": mode,
                "home": usable.home_id.to_numpy(), "away": usable.away_id.to_numpy(),
                "p_home": probabilities[:, 0], "p_draw": probabilities[:, 1],
                "p_away": probabilities[:, 2], "outcome": _outcomes(usable),
            }))
    return pd.concat(pieces, ignore_index=True)


def _losses_from_rows(frame: pd.DataFrame) -> np.ndarray:
    probabilities = frame[["p_home", "p_draw", "p_away"]].to_numpy()
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    probabilities = np.clip(probabilities, 1e-15, 1.0)
    return -np.log(probabilities[np.arange(len(frame)), frame.outcome.to_numpy()])


def paired_losses_dc(
    season_label: str,
    half_life_days: float,
    ridge: float,
    *,
    mode: str = "frozen",
    european_weight: float = 3.0,
) -> pd.DataFrame:
    """Dixon-Coles against the Elo baseline on the same matches, paired."""
    dc = score_season_dc(season_label, half_life_days, ridge, modes=(mode,))
    dc = dc[dc["mode"] == mode].reset_index(drop=True)
    base = _match_losses(score_season(season_label, DEFAULT_PARAMS, modes=(mode,)), mode)
    if len(dc) != len(base) or not (dc.home.to_numpy() == base.home.to_numpy()).all():
        raise ValueError(f"{season_label}: Dixon-Coles and the baseline do not align")
    losses = _losses_from_rows(dc)
    return pd.DataFrame({
        "season": season_label, "competition": dc.competition, "home": dc.home,
        "cluster": season_label + "/" + dc.home.astype(str),
        "loss_tuned": losses, "loss_default": base.loss.to_numpy(),
        "difference": losses - base.loss.to_numpy(),
        "weight": np.where(dc.competition != "domestic", european_weight, 1.0),
    })


def pooled_objective_dc(
    seasons: tuple[str, ...],
    half_life_days: float,
    ridge: float,
    *,
    european_weight: float = 3.0,
    european_only: bool = False,
) -> float:
    """Dixon-Coles pooled log-loss over the given seasons, frozen mode."""
    losses, weights = [], []
    for season_label in seasons:
        frame = score_season_dc(season_label, half_life_days, ridge)
        if european_only:
            frame = frame[frame.competition != "domestic"]
            weight = np.ones(len(frame))
        else:
            weight = np.where(frame.competition != "domestic", european_weight, 1.0)
        losses.append(_losses_from_rows(frame))
        weights.append(weight)
    return weighted_mean(np.concatenate(losses), np.concatenate(weights))


# --- candidate covariates --------------------------------------------------

REST_CAP_DAYS = 14.0
SHORT_REST_DAYS = 4.0


def add_rest_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Days since each club's previous match, as per-side covariates.

    Causal by construction: a match's rest depends only on earlier matches, and
    because fixtures are scheduled it is known before kick-off, so this is the
    one covariate available to the forecast as well as the backtest.

    Measurement error worth remembering: rest is computed from the competitions
    in the cache - 18 domestic leagues plus Europe - so a club playing a
    domestic cup tie we do not hold looks rested when it is not. That pushes the
    coefficient toward zero.
    """
    ordered = frame.sort_values("date").reset_index(drop=True)
    last: dict[str, pd.Timestamp] = {}
    rest_home = np.full(len(ordered), np.nan)
    rest_away = np.full(len(ordered), np.nan)
    for position, row in enumerate(ordered.itertuples()):
        for club, store in ((row.home_id, "home"), (row.away_id, "away")):
            previous = last.get(club)
            if previous is not None:
                days = (row.date - previous).days
                if store == "home":
                    rest_home[position] = days
                else:
                    rest_away[position] = days
            last[club] = row.date
    for name, values in (("home", rest_home), ("away", rest_away)):
        capped = np.clip(values, 0.0, REST_CAP_DAYS) / 7.0
        ordered[f"rest_{name}"] = np.where(np.isnan(values), REST_CAP_DAYS / 7.0, capped)
        ordered[f"short_rest_{name}"] = np.where(
            np.isnan(values), 0.0, (values < SHORT_REST_DAYS).astype(float))
    return ordered


@lru_cache(maxsize=64)
def _fitted_market(cutoff: date, half_life_days: float, scale: str = "z"):
    """Elo ratings plus market-implied ratings and the two-covariate model."""
    from ucl.data import football_data
    from ucl.models.covariates import fit_covariate_model
    from ucl.models.market import fit_market_ratings

    training = _training_frame(cutoff)
    offsets = _cached_offsets(cutoff, int(round(DEFAULT_PARAMS.home_advantage / 5) * 5))
    ratings = replay(training, offsets, DEFAULT_PARAMS)
    market = fit_market_ratings(
        football_data.load_results(cutoff), cutoff, half_life_days=half_life_days,
        scale=scale, elo=ratings.ratings)
    frame = ratings.pre_match.copy()
    frame["d_market"] = market.difference(frame.home_id.to_numpy(), frame.away_id.to_numpy())
    model = fit_covariate_model(frame, symmetric=("d_market",))
    return ratings, market, model


@lru_cache(maxsize=16)
def _fitted_squad_value(cutoff: date, season_label: str):
    """Elo ratings plus the pre-season squad-value model.

    The as-of date is 6 July of the season's starting year, not the cutoff,
    because the source is frozen at that point for 2026-27 and every season is
    held to the same staleness.
    """
    from ucl.data import transfermarkt
    from ucl.models.covariates import fit_covariate_model

    training = _training_frame(cutoff)
    offsets = _cached_offsets(cutoff, int(round(DEFAULT_PARAMS.home_advantage / 5) * 5))
    ratings = replay(training, offsets, DEFAULT_PARAMS)
    values = transfermarkt.squad_values(transfermarkt.as_of_for_season(season_label))
    frame = ratings.pre_match.copy()
    frame["d_value"] = transfermarkt.value_difference(
        values, frame.home_id.to_numpy(), frame.away_id.to_numpy())
    model = fit_covariate_model(frame, symmetric=("d_value",))
    return ratings, values, model


#: The GAP covariate columns, in a fixed order so the fitted coefficients can
#: be read back by name. Defined with the model assembly it belongs to.
from ucl.models.forecast import GAP_COLUMNS  # noqa: E402


@lru_cache(maxsize=16)
def _fitted_gap(cutoff: date, season_label: str, weighting: str = "total",
                measures: tuple[str, ...] | None = None):
    """Model v1 plus the three GAP residual covariates.

    The tuned side of candidate A is v1 *and* the shot ratings, so the
    squad-value term is fitted here too and the test is paired against v1
    rather than against Elo.

    Each measure's learning rate and phis are fitted on this cutoff's data
    alone, by least squares against observed attacking output - an objective
    that never sees a match result. The ratings are then regressed on Elo
    within each league and only the residual is kept.

    ``weighting`` chooses the squad-value measure: ``"total"`` is v1's sum over
    the squad, ``"minutes"`` is candidate B's minutes-weighted mean. Everything
    else is identical, so a test between the two isolates the weighting.
    """
    from ucl.data import transfermarkt
    from ucl.models.forecast import fit_v2

    training = _training_frame(cutoff)
    offsets = _cached_offsets(cutoff, int(round(DEFAULT_PARAMS.home_advantage / 5) * 5))
    ratings = replay(training, offsets, DEFAULT_PARAMS)

    as_of = transfermarkt.as_of_for_season(season_label)
    values = (transfermarkt.squad_values_minutes_weighted(as_of)
              if weighting == "minutes" else transfermarkt.squad_values(as_of))
    fitted = fit_v2(ratings.pre_match, values, ratings.ratings,
                    as_of=cutoff, value_as_of=as_of, measures=measures)
    return (ratings, values, fitted.details["residuals"],
            fitted.details["gap_params"], fitted.model)


@lru_cache(maxsize=8)
def _fitted_congestion(cutoff: date):
    """Elo ratings plus the rest-day model."""
    from ucl.models.covariates import fit_covariate_model

    training = _training_frame(cutoff)
    offsets = _cached_offsets(cutoff, int(round(DEFAULT_PARAMS.home_advantage / 5) * 5))
    ratings = replay(training, offsets, DEFAULT_PARAMS)
    frame = add_rest_days(ratings.pre_match)
    model = fit_covariate_model(frame, per_side=("rest", "short_rest"))
    return ratings, model


def score_season_covariate(
    season_label: str, candidate: str, *, half_life_days: float = 365.0,
    scale: str = "z", modes: tuple[str, ...] = ("frozen",),
) -> pd.DataFrame:
    """Score a season under a candidate covariate model, frozen ratings."""
    cfg = SEASONS[season_label]
    domestic, euro = _evaluation_frames(season_label)

    if candidate in ("market", "market_elo"):
        ratings, market, model = _fitted_market(
            cfg.cutoff_date, half_life_days, "elo" if candidate == "market_elo" else scale)
    elif candidate == "squad_value":
        from ucl.data import transfermarkt

        ratings, values, model = _fitted_squad_value(cfg.cutoff_date, season_label)
    elif candidate in ("gap", "gap_minutes", "gap_sot"):
        from ucl.data import transfermarkt
        from ucl.models import gap as gap_model

        ratings, values, residuals, _, model = _fitted_gap(
            cfg.cutoff_date, season_label,
            "minutes" if candidate == "gap_minutes" else "total",
            ("sot",) if candidate == "gap_sot" else None)
    elif candidate == "congestion":
        ratings, model = _fitted_congestion(cfg.cutoff_date)
        history = _training_frame(cfg.cutoff_date)[["date", "home_id", "away_id"]]
    else:
        raise ValueError(f"unknown candidate {candidate!r}")

    pieces = []
    for mode in modes:
        for frame, competition in ((domestic, "domestic"), (euro, "european")):
            frame = frame.sort_values("date")
            known = {c for c in set(frame.home_id) | set(frame.away_id) if c in ratings.ratings}
            usable = frame[frame.home_id.isin(known) & frame.away_id.isin(known)].copy()
            elo_home = usable.home_id.map(ratings.ratings).to_numpy(dtype=float)
            elo_away = usable.away_id.map(ratings.ratings).to_numpy(dtype=float)

            if candidate in ("market", "market_elo"):
                difference = market.difference(
                    usable.home_id.to_numpy(), usable.away_id.to_numpy())
                rate_home, rate_away = model.rates(
                    elo_home, elo_away, symmetric={"d_market": difference})
            elif candidate == "squad_value":
                difference = transfermarkt.value_difference(
                    values, usable.home_id.to_numpy(), usable.away_id.to_numpy())
                rate_home, rate_away = model.rates(
                    elo_home, elo_away, symmetric={"d_value": difference})
            elif candidate in ("gap", "gap_minutes", "gap_sot"):
                symmetric = {"d_value": transfermarkt.value_difference(
                    values, usable.home_id.to_numpy(), usable.away_id.to_numpy())}
                for measure, column in zip(gap_model.MEASURES, GAP_COLUMNS):
                    if measure not in residuals:
                        continue
                    symmetric[column] = gap_model.match_difference(
                        residuals[measure], usable.home_id.to_numpy(),
                        usable.away_id.to_numpy())
                rate_home, rate_away = model.rates(
                    elo_home, elo_away, symmetric=symmetric)
            else:
                combined = pd.concat(
                    [history, usable[["date", "home_id", "away_id"]]], ignore_index=True)
                with_rest = add_rest_days(combined).tail(len(usable))
                rate_home, rate_away = model.rates(
                    elo_home, elo_away,
                    per_side_home={"rest": with_rest.rest_home.to_numpy(),
                                   "short_rest": with_rest.short_rest_home.to_numpy()},
                    per_side_away={"rest": with_rest.rest_away.to_numpy(),
                                   "short_rest": with_rest.short_rest_away.to_numpy()})

            goals = np.arange(MAX_GOALS + 1)
            home_pmf = poisson.pmf(goals[None, :], np.asarray(rate_home)[:, None])
            away_pmf = poisson.pmf(goals[None, :], np.asarray(rate_away)[:, None])
            joint = home_pmf[:, :, None] * away_pmf[:, None, :]
            upper = np.triu(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), 1)
            draw = np.eye(MAX_GOALS + 1)
            probabilities = np.stack(
                [(joint * upper.T).sum(axis=(1, 2)), (joint * draw).sum(axis=(1, 2)),
                 (joint * upper).sum(axis=(1, 2))], axis=1)
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
            pieces.append(pd.DataFrame({
                "date": usable.date.to_numpy(), "competition": competition, "mode": mode,
                "home": usable.home_id.to_numpy(), "away": usable.away_id.to_numpy(),
                "p_home": probabilities[:, 0], "p_draw": probabilities[:, 1],
                "p_away": probabilities[:, 2], "outcome": _outcomes(usable),
            }))
    return pd.concat(pieces, ignore_index=True)


def paired_losses_covariate(
    season_label: str, candidate: str, *, half_life_days: float = 365.0,
    mode: str = "frozen", european_weight: float = 3.0,
    baseline: str | None = None,
) -> pd.DataFrame:
    """A candidate against the accepted baseline on the same matches, paired.

    ``baseline`` names what the candidate is measured against: ``None`` is the
    Elo baseline, which is what the earlier candidates were tested against
    because that was the accepted model at the time. A candidate declared
    against **model v1** must pass ``"squad_value"`` instead - pairing it
    against Elo alone would credit it with the squad-value term's gain as well
    as its own, and accept it for someone else's work.
    """
    rows = score_season_covariate(
        season_label, candidate, half_life_days=half_life_days, modes=(mode,))
    rows = rows[rows["mode"] == mode].reset_index(drop=True)
    if baseline is None:
        base = _match_losses(
            score_season(season_label, DEFAULT_PARAMS, modes=(mode,)), mode)
        base_losses, base_home = base.loss.to_numpy(), base.home.to_numpy()
    else:
        base_rows = score_season_covariate(
            season_label, baseline, half_life_days=half_life_days, modes=(mode,))
        base_rows = base_rows[base_rows["mode"] == mode].reset_index(drop=True)
        base_losses = _losses_from_rows(base_rows)
        base_home = base_rows.home.to_numpy()
    if len(rows) != len(base_losses) or not (rows.home.to_numpy() == base_home).all():
        raise ValueError(f"{season_label}: {candidate} and the baseline do not align")
    losses = _losses_from_rows(rows)
    return pd.DataFrame({
        "season": season_label, "competition": rows.competition, "home": rows.home,
        "cluster": season_label + "/" + rows.home.astype(str),
        "loss_tuned": losses, "loss_default": base_losses,
        "difference": losses - base_losses,
        "weight": np.where(rows.competition != "domestic", european_weight, 1.0),
    })


def pooled_objective_covariate(
    seasons: tuple[str, ...], candidate: str, *, half_life_days: float = 365.0,
    european_weight: float = 3.0, european_only: bool = False,
) -> float:
    losses, weights = [], []
    for season_label in seasons:
        frame = score_season_covariate(
            season_label, candidate, half_life_days=half_life_days)
        if european_only:
            frame = frame[frame.competition != "domestic"]
            weight = np.ones(len(frame))
        else:
            weight = np.where(frame.competition != "domestic", european_weight, 1.0)
        losses.append(_losses_from_rows(frame))
        weights.append(weight)
    return weighted_mean(np.concatenate(losses), np.concatenate(weights))
