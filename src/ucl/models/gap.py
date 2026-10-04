"""GAP ratings: shot-based attacking and defensive strength.

Wheatcroft's Generalised Attacking Performance ratings, from *Forecasting
football matches by predicting match statistics* (arXiv 2001.09097). Each club
carries four ratings per attacking measure - attacking and defensive, home and
away - and the expected output in a match is

    S_h_hat = (H_i^a + A_j^d) / 2        S_a_hat = (A_j^a + H_i^d) / 2

with the paper's equations 7 and 8 updating all eight ratings afterwards
towards the error, each floored at zero:

    H_i^a <- max(H_i^a + lam * phi_1       * (S_h - S_h_hat), 0)
    A_i^a <- max(A_i^a + lam * (1 - phi_1) * (S_h - S_h_hat), 0)
    H_i^d <- max(H_i^d + lam * phi_1       * (S_a - S_a_hat), 0)
    A_i^d <- max(A_i^d + lam * (1 - phi_1) * (S_a - S_a_hat), 0)

and symmetrically for the away club with ``phi_2``. ``phi`` splits the update
between the rating that was just used and its opposite-venue twin: a club that
over-performs at home is a little better away too, but not equally so.

Three measures, as the paper uses: shots on target, shots off target (shots
minus shots on target) and corners.

**These ratings have no cross-league bridge.** No match in the cache carries
both shot data and two clubs from different leagues, so a Norwegian club's
shot counts and a Spanish club's cannot be put on one scale from this data -
and a weak league's clubs take more shots against weaker opponents, so a raw
cross-league comparison would misinform. :func:`residual_ratings` is the answer
used here: within each league the ratings are regressed on Elo and the residual
is kept, so league quality stays with Elo, where European results have already
calibrated it, and what remains is shot performance relative to what a club's
rating already predicts. Standardising within league is deliberately *not*
done - that is what deleted league quality in candidate 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "MEASURES", "VENUES", "GapParams", "DEFAULT_PARAMS", "GapState",
    "domestic_frame", "prepare", "replay", "fit_params", "residual_ratings",
    "match_difference",
]

#: The attacking measures, and the column pair each reads.
MEASURES: dict[str, tuple[str, str]] = {
    "sot": ("sot_home", "sot_away"),
    "shots_off": ("shots_off_home", "shots_off_away"),
    "corners": ("corners_home", "corners_away"),
}

#: The four ratings a club carries per measure.
VENUES = ("home_attack", "away_attack", "home_defence", "away_defence")


@dataclass(frozen=True)
class GapParams:
    """Learning rate and the two cross-venue split parameters."""

    learning_rate: float = 0.05
    phi_home: float = 0.7
    phi_away: float = 0.7

    def as_tuple(self) -> tuple[float, float, float]:
        return self.learning_rate, self.phi_home, self.phi_away


#: Only a starting point for the optimiser; fit_params replaces it per fold.
DEFAULT_PARAMS = GapParams()


@dataclass
class GapState:
    """Ratings after a replay, and the per-match predictions it produced."""

    measure: str
    params: GapParams
    ratings: dict[str, dict[str, float]] = field(default_factory=dict)
    league: dict[str, str] = field(default_factory=dict)
    matches: int = 0
    sum_squared_error: float = 0.0

    def rating(self, club: str, venue: str) -> float | None:
        held = self.ratings.get(club)
        return None if held is None else held[venue]

    @property
    def mean_squared_error(self) -> float:
        return self.sum_squared_error / self.matches if self.matches else float("nan")


def domestic_frame(cutoff) -> pd.DataFrame:
    """Domestic matches before ``cutoff``, with club ids and the measures.

    Read straight from the football-data loader rather than from
    :func:`ucl.models.elo.match_frame`: that frame is the Elo replay's input and
    carries neither the shot columns nor the league label, and widening it would
    touch every model in the project for the benefit of one covariate. No
    European match has shot data, so nothing is lost by looking only here.
    """
    from ucl.data import football_data, names

    frame = football_data.load_results(cutoff)
    lookup = names.alias_to_canonical("football_data")
    frame = frame.assign(
        home_id=frame.home.map(lookup).fillna(frame.home),
        away_id=frame.away.map(lookup).fillna(frame.away),
    )
    return prepare(frame)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the shots-off-target columns the paper's third measure needs.

    Floored at zero: one row in 33,930 (Newcastle v West Ham, 2021-08-15)
    records more shots on target than shots, and a negative count of attacking
    plays would send the replay chasing an impossible observation.
    """
    out = frame.copy()
    out["shots_off_home"] = (out.shots_home - out.sot_home).clip(lower=0)
    out["shots_off_away"] = (out.shots_away - out.sot_away).clip(lower=0)
    return out


def _measured(frame: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Matches that actually carry the measure, in date order."""
    home, away = MEASURES[measure]
    needed = ["date", "home_id", "away_id", "league", "season", home, away]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing {missing}")
    usable = frame.dropna(subset=[home, away])
    return usable.sort_values(["date", "home_id"]).reset_index(drop=True)


def replay(frame: pd.DataFrame, measure: str,
           params: GapParams = DEFAULT_PARAMS, *,
           initial: str = "league_mean") -> GapState:
    """Replay the ratings over every match carrying ``measure``, in date order.

    ``initial`` sets where a club starts. The paper initialises at zero in a
    league's first season with statistics; ``"league_mean"`` instead starts a
    club at its league's average output, which is the same recursion from a
    sensible point and converges far sooner. With 2016-17 onwards as burn-in
    the two agree well before the first fold - there is a test for that.

    A club new to a league inherits the mean of the clubs that left it, as the
    paper prescribes for promotion and relegation.
    """
    if initial not in ("league_mean", "zero"):
        raise ValueError(f"unknown initial {initial!r}")
    home_column, away_column = MEASURES[measure]
    usable = _measured(frame, measure)
    state = GapState(measure=measure, params=params)
    learning_rate, phi_home, phi_away = params.as_tuple()

    # Starting level per league, and the roster of each league-season, so a
    # promoted club can inherit from the clubs it replaced.
    if initial == "league_mean":
        levels = (usable.groupby("league")[[home_column, away_column]].mean()
                  .mean(axis=1).to_dict())
    else:
        levels = {}
    rosters: dict[tuple[str, str], set[str]] = {}
    for row in usable.itertuples():
        rosters.setdefault((row.league, row.season), set()).update(
            (row.home_id, row.away_id))

    seen_seasons: dict[str, str] = {}

    def enter(club: str, league: str, season: str) -> dict[str, float]:
        start = float(levels.get(league, 0.0))
        previous = seen_seasons.get(league)
        if previous is not None:
            left = rosters.get((league, previous), set()) - rosters.get(
                (league, season), set())
            departed = [state.ratings[name] for name in left
                        if name in state.ratings]
            if departed:
                # Promoted clubs inherit the mean of the relegated ones.
                return {venue: float(np.mean([held[venue] for held in departed]))
                        for venue in VENUES}
        return {venue: start for venue in VENUES}

    for row in usable.itertuples():
        league, season = row.league, row.season
        if seen_seasons.get(league) != season:
            # First match of a league-season: settle who is new.
            for club in rosters.get((league, season), set()):
                if club not in state.ratings:
                    state.ratings[club] = enter(club, league, season)
            seen_seasons[league] = season

        home, away = row.home_id, row.away_id
        for club in (home, away):
            state.ratings.setdefault(club, {venue: float(levels.get(league, 0.0))
                                            for venue in VENUES})
            state.league[club] = league

        held_home, held_away = state.ratings[home], state.ratings[away]
        expected_home = (held_home["home_attack"] + held_away["away_defence"]) / 2
        expected_away = (held_away["away_attack"] + held_home["home_defence"]) / 2
        observed_home = float(getattr(row, home_column))
        observed_away = float(getattr(row, away_column))
        error_home = observed_home - expected_home
        error_away = observed_away - expected_away

        state.matches += 1
        state.sum_squared_error += error_home ** 2 + error_away ** 2

        # Equation 7: the home club.
        held_home["home_attack"] = max(
            held_home["home_attack"] + learning_rate * phi_home * error_home, 0.0)
        held_home["away_attack"] = max(
            held_home["away_attack"] + learning_rate * (1 - phi_home) * error_home, 0.0)
        held_home["home_defence"] = max(
            held_home["home_defence"] + learning_rate * phi_home * error_away, 0.0)
        held_home["away_defence"] = max(
            held_home["away_defence"] + learning_rate * (1 - phi_home) * error_away, 0.0)
        # Equation 8: the away club.
        held_away["away_attack"] = max(
            held_away["away_attack"] + learning_rate * phi_away * error_away, 0.0)
        held_away["home_attack"] = max(
            held_away["home_attack"] + learning_rate * (1 - phi_away) * error_away, 0.0)
        held_away["away_defence"] = max(
            held_away["away_defence"] + learning_rate * phi_away * error_home, 0.0)
        held_away["home_defence"] = max(
            held_away["home_defence"] + learning_rate * (1 - phi_away) * error_home, 0.0)

    return state


def fit_params(frame: pd.DataFrame, measure: str, *,
               initial: str = "league_mean") -> GapParams:
    """Fit the learning rate and the two phis by least squares.

    The paper's equation 2: minimise the squared error between predicted and
    observed attacking output. The objective never looks at a match *result*,
    only at how many shots or corners were taken, so fitting it cannot leak the
    thing the forecast is trying to predict.
    """
    prepared = _measured(frame, measure)

    def objective(theta: np.ndarray) -> float:
        params = GapParams(float(theta[0]), float(theta[1]), float(theta[2]))
        return replay(prepared, measure, params, initial=initial).sum_squared_error

    start = np.array(DEFAULT_PARAMS.as_tuple())
    result = minimize(objective, start, method="L-BFGS-B",
                      bounds=[(1e-4, 0.5), (0.5, 1.0), (0.5, 1.0)],
                      options={"maxiter": 40, "eps": 1e-3})
    return GapParams(float(result.x[0]), float(result.x[1]), float(result.x[2]))


def residual_ratings(state: GapState, elo: dict[str, float]) -> dict[str, dict[str, float]]:
    """Strip out the part of each rating that Elo already explains.

    Within each league, regress the clubs' ratings on their Elo and keep the
    residual. The level of a league's shot counts goes with the Elo term, so
    what is left is a club's shot performance relative to its rating - a
    quantity that means the same thing in two different leagues, which the raw
    ratings do not.

    A league with fewer than four rated clubs is left alone: a slope fitted on
    three points is noise.
    """
    by_league: dict[str, list[str]] = {}
    for club, league in state.league.items():
        if club in elo:
            by_league.setdefault(league, []).append(club)

    residuals: dict[str, dict[str, float]] = {
        club: dict(held) for club, held in state.ratings.items()}
    for league, clubs in by_league.items():
        if len(clubs) < 4:
            continue
        ratings = np.array([elo[club] for club in clubs], dtype=float)
        design = np.column_stack([np.ones(len(clubs)), ratings])
        for venue in VENUES:
            values = np.array([state.ratings[club][venue] for club in clubs],
                              dtype=float)
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            fitted = design @ coefficients
            for club, residual in zip(clubs, values - fitted):
                residuals[club][venue] = float(residual)
    # A club with no Elo was never regressed; it cannot be compared and is
    # dropped so match_difference treats it as uncovered.
    return {club: held for club, held in residuals.items() if club in elo}


def match_difference(residuals: dict[str, dict[str, float]], home, away) -> np.ndarray:
    """The expected-output difference from residual ratings, per match.

    ``(S_h_hat - S_a_hat)`` built from residuals instead of raw ratings. Zero
    where either club is unrated, the same convention the squad-value covariate
    uses, so an uncovered match falls back on the rest of the model rather than
    being dropped.
    """
    home = np.atleast_1d(np.asarray(home, dtype=object))
    away = np.atleast_1d(np.asarray(away, dtype=object))
    out = np.zeros(len(home), dtype=float)
    for index, (one, other) in enumerate(zip(home, away)):
        held_home, held_away = residuals.get(one), residuals.get(other)
        if held_home is None or held_away is None:
            continue
        expected_home = (held_home["home_attack"] + held_away["away_defence"]) / 2
        expected_away = (held_away["away_attack"] + held_home["home_defence"]) / 2
        out[index] = expected_home - expected_away
    return out
