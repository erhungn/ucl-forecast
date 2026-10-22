"""Our own Elo, computed from domestic and European results.

ClubElo is unreachable, so ratings are built here by replaying every match
before the cutoff. Two design points matter:

**Cross-league scale.** The big-league files start in 2021 and European results
are the only thing connecting national leagues, so starting every club at 1500
would leave the between-league spread to emerge far too slowly from a few
hundred European matches. Instead, association strength offsets are fitted once
from all pre-cutoff European matches (:func:`fit_association_offsets`) and added
to the starting ratings, so the replay begins with leagues already on a sensible
scale and only has to learn club-level deviations.

**Home advantage is applied once.** :data:`ucl.config.HOME_ADVANTAGE` enters the
Elo expectation for the rating update. The Poisson match model must use the raw
rating difference and carry home advantage in its own intercept; applying both
would count it twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ucl import config
from ucl.data import european, football_data, names
from ucl.data.cutoff import assert_no_lookahead

__all__ = [
    "MATCH_COLUMNS", "EloParams", "EloResult", "match_frame",
    "fit_association_offsets", "replay", "elo_as_of", "goal_difference_multiplier",
]


@dataclass(frozen=True)
class EloParams:
    """Every tunable of the rating system, so a backtest can vary them.

    ``gd_exponent`` None keeps the World Football Elo weighting (1, 1.5, then
    (11+gd)/8); a number uses ``(1 + gd) ** exponent`` instead, which at 0 makes
    every win count the same regardless of margin.

    ``carryover_target`` chooses what a club regresses toward between seasons:
    its league's mean, its own long-run level, or nothing at all.
    """

    k_domestic: float = config.K_DOMESTIC
    k_european: float = config.K_EUROPEAN
    home_advantage: float = config.HOME_ADVANTAGE
    gd_exponent: float | None = config.GD_EXPONENT
    season_carryover: float = config.SEASON_CARRYOVER
    carryover_target: str = config.CARRYOVER_TARGET
    new_club_offset: float = config.NEW_CLUB_OFFSET

    def as_dict(self) -> dict:
        return {
            "k_domestic": self.k_domestic, "k_european": self.k_european,
            "home_advantage": self.home_advantage, "gd_exponent": self.gd_exponent,
            "season_carryover": self.season_carryover,
            "carryover_target": self.carryover_target,
        }


DEFAULT_PARAMS = EloParams()
CARRYOVER_TARGETS = ("league_mean", "club_long_run", "none")

MATCH_COLUMNS = [
    "date", "home_id", "away_id", "home_assoc", "away_assoc", "hg", "ag",
    "season", "competition", "is_european",
]


# --- assembling one match frame -------------------------------------------

def _canonical(frame: pd.DataFrame, source: str, column: str) -> pd.Series:
    """Map a source's club names to ids, keeping the raw name when unmapped.

    Clubs outside the mapping's scope - a domestic-only club, or a European club
    from an uncached league - keep their own name as their id, which is the
    agreed rule.
    """
    lookup = names.alias_to_canonical(source)
    return frame[column].map(lookup).fillna(frame[column])


def _levels(euro: pd.DataFrame) -> pd.Series:
    """The competition level each European match is fitted under.

    Three levels, because the three competitions do not score alike: across the
    cache the UCL main phase runs 3.13 goals a match against 2.83 for everything
    else European and 2.78 domestic. UCL **qualifying** sits with the others at
    2.80, so it belongs there rather than with the draw it feeds.
    """
    main = european.is_main_phase(euro)
    return pd.Series(
        np.where(main & (euro.competition == "UCL"), "ucl_main", "other_european"),
        index=euro.index,
    )


def match_frame(cutoff: date) -> pd.DataFrame:
    """Every match before ``cutoff``, domestic and European, with club ids."""
    domestic = football_data.load_results(cutoff)
    domestic = domestic[domestic.date >= pd.Timestamp(config.ELO_BURN_IN_START)]
    home_assoc = domestic.country.map(names.association)
    dom = pd.DataFrame(
        {
            "date": domestic.date,
            "home_id": _canonical(domestic, "football_data", "home"),
            "away_id": _canonical(domestic, "football_data", "away"),
            "home_assoc": home_assoc,
            "away_assoc": home_assoc,
            "hg": domestic.hg,
            "ag": domestic.ag,
            "season": domestic.season,
            "competition": "domestic",
            "is_european": False,
            "level": "domestic",
        }
    )

    euro = european.load_results(cutoff)
    euro = euro[euro.date >= pd.Timestamp(config.ELO_BURN_IN_START)]
    home_ids = pd.Series(index=euro.index, dtype=object)
    away_ids = pd.Series(index=euro.index, dtype=object)
    for source in euro.source.unique():
        mask = euro.source == source
        home_ids[mask] = _canonical(euro[mask], source, "home")
        away_ids[mask] = _canonical(euro[mask], source, "away")
    eur = pd.DataFrame(
        {
            "date": euro.date,
            "home_id": home_ids,
            "away_id": away_ids,
            "home_assoc": euro.home_country.map(names.association),
            "away_assoc": euro.away_country.map(names.association),
            "hg": euro.hg,
            "ag": euro.ag,
            "season": euro.season,
            "competition": euro.competition,
            "is_european": True,
            "level": _levels(euro),
        }
    )

    # `level` is carried beside MATCH_COLUMNS rather than inside it: a test
    # builds a synthetic frame and projects it through MATCH_COLUMNS, so adding
    # a column there would break callers that have no competition to declare.
    # `replay` copies the whole frame, so pre_match gets it for free and no
    # existing fit sees it.
    columns = MATCH_COLUMNS + ["level"]
    frame = pd.concat([dom[columns], eur[columns]], ignore_index=True)
    frame = frame.dropna(subset=["home_id", "away_id", "hg", "ag"])
    assert_no_lookahead(frame, cutoff, source="elo match frame")
    return frame.sort_values(["date", "home_id"]).reset_index(drop=True)


# --- association offsets ---------------------------------------------------

def fit_association_offsets(
    european_matches: pd.DataFrame,
    *,
    home_advantage: float = config.HOME_ADVANTAGE,
    min_matches: int = 10,
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit association strength offsets, in Elo points, from European matches.

    A Davidson model: with ``p = 10 ** ((r_home + home_advantage) / 400)`` and
    ``q = 10 ** (r_away / 400)``, a home win has probability ``p / (p + q + v*sqrt(p*q))``,
    an away win ``q / (...)`` and a draw ``v*sqrt(p*q) / (...)``. Working directly
    in Elo points avoids converting through a goal-rate scale, and the draw
    parameter ``v`` keeps draws from biasing the offsets.

    Associations with fewer than ``min_matches`` matches are held at 0: there is
    not enough evidence to place them, and letting them float would chase noise.
    Returns ``(offsets, diagnostics)``.
    """
    euro = european_matches[european_matches.is_european]
    counts = pd.concat([euro.home_assoc, euro.away_assoc]).value_counts()
    fitted = sorted(counts[counts >= min_matches].index)
    index = {name: i for i, name in enumerate(fitted)}

    home = euro.home_assoc.map(index).to_numpy()
    away = euro.away_assoc.map(index).to_numpy()
    keep = ~(pd.isna(home) | pd.isna(away))
    home, away = home[keep].astype(int), away[keep].astype(int)
    hg = euro.hg.to_numpy()[keep]
    ag = euro.ag.to_numpy()[keep]
    outcome = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))

    def negative_log_likelihood(theta: np.ndarray) -> float:
        offsets = np.concatenate([[0.0], theta[:-1]])
        offsets = offsets - offsets.mean()          # identifiable: mean zero
        nu = np.exp(theta[-1])
        log_p = (offsets[home] + home_advantage) / 400.0 * np.log(10.0)
        log_q = offsets[away] / 400.0 * np.log(10.0)
        p, q = np.exp(log_p), np.exp(log_q)
        draw = nu * np.sqrt(p * q)
        total = p + q + draw
        chosen = np.where(outcome == 0, p, np.where(outcome == 1, draw, q))
        return -np.sum(np.log(np.clip(chosen / total, 1e-300, None)))

    start = np.concatenate([np.zeros(len(fitted) - 1), [np.log(1.0)]])
    result = minimize(negative_log_likelihood, start, method="L-BFGS-B")
    raw = np.concatenate([[0.0], result.x[:-1]])
    raw = raw - raw.mean()
    offsets = {name: float(raw[index[name]]) for name in fitted}

    everyone = sorted(set(euro.home_assoc) | set(euro.away_assoc))
    for name in everyone:
        offsets.setdefault(name, 0.0)
    diagnostics = {
        "matches": int(keep.sum()),
        "associations_fitted": len(fitted),
        "associations_held_at_zero": len(everyone) - len(fitted),
        "draw_parameter": float(np.exp(result.x[-1])),
        "converged": bool(result.success),
    }
    return offsets, diagnostics


# --- the replay ------------------------------------------------------------

def goal_difference_multiplier(
    goal_difference: int, exponent: float | None = None
) -> float:
    """Weight an update by the margin.

    ``exponent`` None is the World Football Elo rule (1, 1.5, then (11+gd)/8);
    otherwise ``(1 + gd) ** exponent``, so 0 ignores the margin entirely.
    """
    gd = abs(int(goal_difference))
    if exponent is not None:
        return float((1.0 + gd) ** exponent)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


@dataclass
class EloResult:
    ratings: dict[str, float]
    pre_match: pd.DataFrame
    offsets: dict[str, float]
    association_of: dict[str, str]
    domestic_associations: set[str] = field(default_factory=set)
    diagnostics: dict = field(default_factory=dict)

    def association_means(self) -> pd.DataFrame:
        """Mean rating per association at the cutoff, strongest first."""
        rows = [
            {"association": self.association_of.get(club, "?"), "club": club, "elo": rating}
            for club, rating in self.ratings.items()
        ]
        frame = pd.DataFrame(rows)
        out = (
            frame.groupby("association")
            .agg(clubs=("club", "size"), mean_elo=("elo", "mean"), max_elo=("elo", "max"))
            .reset_index()
            .sort_values("mean_elo", ascending=False)
            .reset_index(drop=True)
        )
        out["offset"] = out.association.map(self.offsets)
        # Associations with no cached league are represented only by the clubs
        # that reached Europe, i.e. their best, so their mean reads high. The
        # flag keeps that visible rather than letting the ranking mislead.
        out["has_domestic_data"] = out.association.isin(self.domestic_associations)
        return out

    def top(self, n: int = 20) -> pd.DataFrame:
        frame = pd.DataFrame(
            {"club": list(self.ratings), "elo": list(self.ratings.values())}
        )
        frame["association"] = frame.club.map(self.association_of)
        return frame.sort_values("elo", ascending=False).head(n).reset_index(drop=True)


def replay(
    matches: pd.DataFrame,
    offsets: dict[str, float],
    params: EloParams = DEFAULT_PARAMS,
) -> EloResult:
    """Replay every match in date order and return ratings and pre-match snapshots.

    Between seasons each club is regressed toward its target: its league's mean
    where that league is in the cache, and otherwise its association baseline,
    which is the only sensible target for a club we never see domestically.
    """
    ratings: dict[str, float] = {}
    association_of: dict[str, str] = {}
    domestic_assoc: set[str] = set(
        matches.loc[~matches.is_european, "home_assoc"].unique()
    )

    def baseline(assoc: str) -> float:
        return config.START_RATING + offsets.get(assoc, 0.0)

    def enter(club: str, assoc: str) -> float:
        """Rating for a club seen for the first time."""
        association_of.setdefault(club, assoc)
        if assoc not in domestic_assoc:
            # No cached league, so no league mean exists: the association
            # baseline is the only information we have about this club.
            return baseline(assoc)
        peers = [r for c, r in ratings.items() if association_of.get(c) == assoc]
        mean = float(np.mean(peers)) if peers else baseline(assoc)
        return mean + params.new_club_offset

    history: dict[str, list[float]] = {}

    def league_mean(assoc: str) -> float:
        if assoc not in domestic_assoc:
            return baseline(assoc)
        peers = [r for c, r in ratings.items() if association_of.get(c) == assoc]
        return float(np.mean(peers)) if peers else baseline(assoc)

    def club_long_run(club: str) -> float:
        """Exponentially weighted mean of the club's own season-end ratings.

        Decay 0.5 per season, most recent first. A club with fewer than two
        seasons behind it has no long-run level yet, so it falls back to the
        league mean.
        """
        past = history.get(club, [])
        if len(past) < 2:
            return league_mean(association_of.get(club, ""))
        weights = np.array([0.5 ** i for i in range(len(past))])
        values = np.array(past[::-1])
        return float((weights * values).sum() / weights.sum())

    def regression_target(club: str) -> float:
        if params.carryover_target == "club_long_run":
            return club_long_run(club)
        return league_mean(association_of.get(club, ""))

    pre_home = np.empty(len(matches))
    pre_away = np.empty(len(matches))
    # Season boundaries are per association: leagues label seasons differently
    # ("2025-26" in England, "2026" in Norway) and run on different calendars,
    # so a single global season counter would fire on almost every match and
    # regress every club to its league mean over and over.
    season_of: dict[str, str] = {}
    clubs_by_assoc: dict[str, set[str]] = {}

    for position, row in enumerate(matches.itertuples()):
        if not row.is_european:
            assoc = row.home_assoc
            previous = season_of.get(assoc)
            if previous is not None and row.season != previous:
                # The target is computed once, from the ratings as they stand
                # before any club is regressed. Recomputing it inside the loop
                # would make each club's result depend on the order of the
                # others - and set iteration order is not stable between runs.
                members = sorted(clubs_by_assoc.get(assoc, set()))
                if members:
                    # Targets are computed from the ratings as they stand before
                    # any club is regressed, so the result cannot depend on order.
                    for club in members:
                        history.setdefault(club, []).append(ratings[club])
                    if params.carryover_target != "none":
                        targets = {club: regression_target(club) for club in members}
                        for club in members:
                            target = targets[club]
                            ratings[club] = target + params.season_carryover * (
                                ratings[club] - target
                            )
            season_of[assoc] = row.season

        for club, assoc in ((row.home_id, row.home_assoc), (row.away_id, row.away_assoc)):
            if club not in ratings:
                ratings[club] = enter(club, assoc)
            association_of.setdefault(club, assoc)
            clubs_by_assoc.setdefault(association_of[club], set()).add(club)

        home, away = ratings[row.home_id], ratings[row.away_id]
        pre_home[position], pre_away[position] = home, away

        expected = 1.0 / (1.0 + 10.0 ** (-(home + params.home_advantage - away) / 400.0))
        actual = 1.0 if row.hg > row.ag else (0.5 if row.hg == row.ag else 0.0)
        k = params.k_european if row.is_european else params.k_domestic
        change = (
            k
            * goal_difference_multiplier(row.hg - row.ag, params.gd_exponent)
            * (actual - expected)
        )
        ratings[row.home_id] = home + change
        ratings[row.away_id] = away - change

    snapshot = matches.copy()
    snapshot["elo_home"] = pre_home
    snapshot["elo_away"] = pre_away
    return EloResult(
        ratings=ratings,
        pre_match=snapshot,
        offsets=offsets,
        association_of=association_of,
        domestic_associations=domestic_assoc,
        diagnostics={"matches": len(matches), "clubs": len(ratings)},
    )


def elo_as_of(cutoff: date, params: EloParams = DEFAULT_PARAMS) -> EloResult:
    """Ratings at ``cutoff``: offsets fitted, then every prior match replayed.

    Both steps see only matches before ``cutoff``: ``match_frame`` filters and
    asserts it, and the association offsets are fitted on that same frame, so a
    backtest at an earlier cutoff cannot inherit anything from later football.
    """
    matches = match_frame(cutoff)
    offsets, offset_diagnostics = fit_association_offsets(
        matches, home_advantage=params.home_advantage
    )
    result = replay(matches, offsets, params)
    result.diagnostics.update({f"offset_{k}": v for k, v in offset_diagnostics.items()})
    return result
