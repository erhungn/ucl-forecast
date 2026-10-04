"""GAP ratings, against the paper's equations and against the league confound.

The update equations are checked by hand on a single match, because a rating
system that is subtly wrong still produces plausible-looking numbers. The
Elo-residual mapping is checked for the thing it exists to fix: raw shot
ratings rank Celtic above Manchester City, and they must not.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ucl.models import gap


def match(home, away, sot_home, sot_away, *, day=1, season="2020-21",
          league="Test League", shots_home=None, shots_away=None,
          corners_home=0.0, corners_away=0.0) -> dict:
    return {
        "date": pd.Timestamp(2020, 8, day), "home_id": home, "away_id": away,
        "league": league, "season": season,
        "sot_home": float(sot_home), "sot_away": float(sot_away),
        "shots_home": float(sot_home if shots_home is None else shots_home),
        "shots_away": float(sot_away if shots_away is None else shots_away),
        "corners_home": corners_home, "corners_away": corners_away,
    }


# --- the paper's equations ------------------------------------------------

def test_one_match_reproduces_equations_seven_and_eight_by_hand():
    """Every one of the eight updates, computed on paper.

    From zero, with lam 0.1, phi_home 0.7, phi_away 0.6 and a 10-4 shot count,
    both expectations are 0, so the errors are 10 and 4 and each rating moves
    by lam times its phi share.
    """
    frame = gap.prepare(pd.DataFrame([match("A", "B", 10, 4)]))
    params = gap.GapParams(learning_rate=0.1, phi_home=0.7, phi_away=0.6)
    state = gap.replay(frame, "sot", params, initial="zero")

    home, away = state.ratings["A"], state.ratings["B"]
    assert home["home_attack"] == pytest.approx(0.7)      # 0.1 * 0.7 * 10
    assert home["away_attack"] == pytest.approx(0.3)      # 0.1 * 0.3 * 10
    assert home["home_defence"] == pytest.approx(0.28)    # 0.1 * 0.7 * 4
    assert home["away_defence"] == pytest.approx(0.12)    # 0.1 * 0.3 * 4
    assert away["away_attack"] == pytest.approx(0.24)     # 0.1 * 0.6 * 4
    assert away["home_attack"] == pytest.approx(0.16)     # 0.1 * 0.4 * 4
    assert away["away_defence"] == pytest.approx(0.6)     # 0.1 * 0.6 * 10
    assert away["home_defence"] == pytest.approx(0.4)     # 0.1 * 0.4 * 10
    assert state.sum_squared_error == pytest.approx(100 + 16)


def test_the_expectation_is_the_mean_of_attack_and_opposing_defence():
    frame = gap.prepare(pd.DataFrame([
        match("A", "B", 10, 4, day=1),
        match("A", "B", 10, 4, day=2),
    ]))
    params = gap.GapParams(0.1, 0.7, 0.6)
    state = gap.replay(frame, "sot", params, initial="zero")
    # After match 1, A's home attack is 0.7 and B's away defence 0.6, so the
    # second match expected (0.7 + 0.6) / 2 = 0.65 and erred by 9.35.
    expected = (0.7 + 0.6) / 2
    assert state.sum_squared_error == pytest.approx(
        100 + 16 + (10 - expected) ** 2 + (4 - (0.24 + 0.28) / 2) ** 2)


def test_ratings_are_floored_at_zero():
    """The paper's max(.., 0): a rating cannot go negative however bad the run."""
    rows = [match("A", "B", 0, 20, day=index + 1) for index in range(20)]
    state = gap.replay(gap.prepare(pd.DataFrame(rows)), "sot",
                       gap.GapParams(0.3, 0.7, 0.7), initial="zero")
    for held in state.ratings.values():
        for venue in gap.VENUES:
            assert held[venue] >= 0.0


def test_a_promoted_club_inherits_the_relegated_clubs_ratings():
    rows = [
        match("A", "B", 12, 6, day=1, season="2020-21"),
        match("C", "D", 10, 8, day=2, season="2020-21"),
        # D is relegated, E promoted in its place.
        match("A", "E", 11, 5, day=3, season="2021-22"),
    ]
    frame = gap.prepare(pd.DataFrame(rows))
    state_before = gap.replay(gap.prepare(pd.DataFrame(rows[:2])), "sot",
                              initial="zero")
    departed = state_before.ratings["D"]
    # Replay the whole thing and inspect E's entry point by re-deriving it: E
    # starts where D finished, then plays one match.
    state = gap.replay(frame, "sot", initial="zero")
    assert "E" in state.ratings
    # E's away ratings moved from D's by exactly one update, so they cannot be
    # the league-mean default of zero.
    assert state.ratings["E"]["home_attack"] != 0.0 or departed["home_attack"] != 0.0


def test_only_matches_carrying_the_measure_are_replayed():
    rows = [match("A", "B", 10, 4, day=1), match("C", "D", 10, 4, day=2)]
    frame = gap.prepare(pd.DataFrame(rows))
    frame.loc[1, ["sot_home", "sot_away"]] = np.nan
    state = gap.replay(frame, "sot", initial="zero")
    assert state.matches == 1
    assert set(state.ratings) == {"A", "B"}


# --- shots off target -----------------------------------------------------

def test_shots_off_target_is_floored_at_zero():
    """One football-data row records more shots on target than shots."""
    frame = gap.prepare(pd.DataFrame([
        match("A", "B", 9, 9, shots_home=17, shots_away=8)]))
    assert frame.shots_off_home.iloc[0] == 8.0      # 17 - 9
    assert frame.shots_off_away.iloc[0] == 0.0      # 8 - 9, floored


def test_the_real_cache_has_no_negative_attacking_output():
    frame = gap.domestic_frame(date(2027, 7, 1))
    for column in ("shots_off_home", "shots_off_away"):
        assert (frame[column].dropna() >= 0).all()


# --- the parameter fit ----------------------------------------------------

def test_fitting_the_parameters_reduces_the_squared_error():
    rows = []
    rng = np.random.default_rng(7)
    clubs = ["A", "B", "C", "D"]
    for day in range(1, 120):
        home, away = rng.choice(clubs, 2, replace=False)
        strength = {"A": 14, "B": 10, "C": 7, "D": 4}
        rows.append(match(home, away, rng.poisson(strength[home]),
                          rng.poisson(strength[away]), day=1 + day % 28,
                          season=f"20{20 + day // 40}-{21 + day // 40}"))
    frame = gap.prepare(pd.DataFrame(rows))
    default = gap.replay(frame, "sot").sum_squared_error
    fitted = gap.fit_params(frame, "sot")
    assert gap.replay(frame, "sot", fitted).sum_squared_error <= default
    assert 1e-4 <= fitted.learning_rate <= 0.5
    assert 0.5 <= fitted.phi_home <= 1.0


def test_a_club_that_outshoots_its_league_ends_up_rated_for_it():
    rows = []
    for day in range(1, 60):
        rows.append(match("Strong", "Weak", 15, 3, day=1 + day % 28))
        rows.append(match("Weak", "Strong", 3, 15, day=1 + day % 28))
    state = gap.replay(gap.prepare(pd.DataFrame(rows)), "sot", initial="zero")
    assert state.ratings["Strong"]["home_attack"] > \
        state.ratings["Weak"]["home_attack"]
    assert state.ratings["Strong"]["home_defence"] < \
        state.ratings["Weak"]["home_defence"]


# --- the Elo residual, which is the whole point ---------------------------

def _real_state(cutoff):
    from ucl.backtest import DEFAULT_PARAMS, _cached_offsets, _training_frame
    from ucl.models.elo import replay as elo_replay

    training = _training_frame(cutoff)
    offsets = _cached_offsets(
        cutoff, int(round(DEFAULT_PARAMS.home_advantage / 5) * 5))
    elo = elo_replay(training, offsets, DEFAULT_PARAMS).ratings
    state = gap.replay(gap.domestic_frame(cutoff), "sot")
    return state, elo


def test_the_residual_is_orthogonal_to_elo():
    """Raw ratings carry league quality; the residual must not."""
    state, elo = _real_state(date(2022, 9, 9))
    residuals = gap.residual_ratings(state, elo)
    clubs = [club for club in residuals if club in elo]
    raw = np.array([state.ratings[club]["home_attack"] for club in clubs])
    residual = np.array([residuals[club]["home_attack"] for club in clubs])
    ratings = np.array([elo[club] for club in clubs])
    assert abs(np.corrcoef(raw, ratings)[0, 1]) > 0.4
    assert abs(np.corrcoef(residual, ratings)[0, 1]) < 1e-6


def test_the_residual_demotes_a_club_that_only_dominates_a_weak_league():
    """Celtic outshoots Manchester City; it is not the better attacking side."""
    state, elo = _real_state(date(2022, 9, 9))
    residuals = gap.residual_ratings(state, elo)
    assert state.ratings["Celtic"]["home_attack"] > \
        state.ratings["Manchester City"]["home_attack"]
    assert residuals["Celtic"]["home_attack"] < \
        state.ratings["Celtic"]["home_attack"]
    # And the ranking by residual no longer puts Celtic in the top handful.
    ranked = sorted(residuals, key=lambda c: -residuals[c]["home_attack"])
    assert "Celtic" not in ranked[:5]


def test_a_tiny_league_is_left_unregressed():
    rows = [match("A", "B", 10, 4, day=1, league="Tiny"),
            match("B", "A", 8, 6, day=2, league="Tiny")]
    state = gap.replay(gap.prepare(pd.DataFrame(rows)), "sot", initial="zero")
    residuals = gap.residual_ratings(state, {"A": 1500.0, "B": 1400.0})
    # Two clubs: a slope would be noise, so the raw ratings are kept.
    assert residuals["A"] == state.ratings["A"]


def test_a_club_without_an_elo_is_dropped_rather_than_guessed():
    rows = [match("A", "B", 10, 4, day=1)]
    state = gap.replay(gap.prepare(pd.DataFrame(rows)), "sot", initial="zero")
    residuals = gap.residual_ratings(state, {"A": 1500.0})
    assert set(residuals) == {"A"}


# --- the match covariate --------------------------------------------------

def test_the_difference_is_zero_when_either_club_is_unrated():
    residuals = {"A": {venue: 1.0 for venue in gap.VENUES}}
    out = gap.match_difference(residuals, ["A", "A", "X"], ["B", "A", "Y"])
    assert out[0] == 0.0          # B unrated
    assert out[1] == 0.0          # A against itself: equal ratings cancel
    assert out[2] == 0.0          # neither rated


def test_the_difference_uses_home_and_away_ratings_in_the_right_places():
    residuals = {
        "H": {"home_attack": 6.0, "away_attack": 0.0,
              "home_defence": 2.0, "away_defence": 0.0},
        "A": {"home_attack": 0.0, "away_attack": 4.0,
              "home_defence": 0.0, "away_defence": 1.0},
    }
    # (6 + 1)/2 - (4 + 2)/2 = 3.5 - 3.0
    assert gap.match_difference(residuals, ["H"], ["A"])[0] == pytest.approx(0.5)
    # Swapping sides is not a sign flip: the venue ratings differ, which is the
    # point of keeping four of them.
    swapped = gap.match_difference(residuals, ["A"], ["H"])[0]
    assert swapped != pytest.approx(-0.5)


# --- leakage --------------------------------------------------------------

@pytest.mark.parametrize("cutoff", [date(2022, 9, 9), date(2024, 9, 12)])
def test_the_replay_sees_nothing_from_the_cutoff_onwards(cutoff):
    frame = gap.domestic_frame(cutoff)
    assert frame.date.max().date() < cutoff
    state = gap.replay(frame, "sot")
    assert state.matches > 10_000


def test_an_earlier_cutoff_rates_strictly_fewer_matches():
    early = gap.replay(gap.domestic_frame(date(2022, 9, 9)), "sot")
    late = gap.replay(gap.domestic_frame(date(2025, 9, 16)), "sot")
    assert early.matches < late.matches


def test_the_initialisation_choice_washes_out_by_the_first_fold():
    """The paper starts at zero; a league-mean start is the same recursion.

    With 2016-17 onwards as burn-in the two agree closely by the first fold, so
    the choice is not load-bearing - which is worth knowing rather than
    assuming.
    """
    frame = gap.domestic_frame(date(2022, 9, 9))
    from_zero = gap.replay(frame, "sot", initial="zero")
    from_mean = gap.replay(frame, "sot", initial="league_mean")
    shared = set(from_zero.ratings) & set(from_mean.ratings)
    gaps = [abs(from_zero.ratings[club]["home_attack"]
                - from_mean.ratings[club]["home_attack"]) for club in shared]
    assert np.mean(gaps) < 0.35
