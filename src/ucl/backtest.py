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
    rate_home, rate_away = model.rates(elo_home, elo_away, neutral=neutral)
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
