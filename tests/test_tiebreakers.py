"""Article 18 of the UEFA Champions League 2026/27 regulations, in order.

Verbatim from the official PDF, criteria a-j: superior goal difference; goals
scored; away goals scored; wins; away wins; points obtained collectively by
league phase opponents; collective goal difference of those opponents; goals
scored collectively by them; lower disciplinary points; higher club coefficient.

Criteria i and j are not simulatable from scorelines, so the implementation must
report them as unresolved rather than invent a winner.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ucl.sim.tiebreakers import CRITERIA, rank_table


def results(rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["home", "away", "hg", "ag"])


def _two_team_league(extra: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    """A and B both beat C and D, so they finish level on points."""
    base = [
        ("A", "C", 1, 0), ("D", "A", 0, 1),
        ("B", "C", 1, 0), ("D", "B", 0, 1),
    ]
    return results(base + extra)


def test_points_come_first():
    table = rank_table(results([("A", "B", 3, 0), ("B", "A", 0, 1)]))
    assert list(table.team) == ["A", "B"]
    assert table.iloc[0].points == 6


def test_goal_difference_separates_equal_points():
    table = rank_table(_two_team_league([("A", "D", 5, 0), ("C", "B", 0, 1)]))
    assert list(table.team)[:2] == ["A", "B"]
    # separated_by says how a club was separated from the one above it.
    assert table.iloc[1].separated_by == "goal difference"


def test_goals_scored_when_goal_difference_is_equal():
    table = rank_table(_two_team_league([("A", "D", 3, 2), ("B", "D", 1, 0)]))
    assert list(table.team)[:2] == ["A", "B"]
    assert table.iloc[1].separated_by == "goals scored"


def test_away_goals_scored_is_the_third_criterion():
    # Same goal difference and same goals scored, but A scored more away.
    table = rank_table(_two_team_league([("C", "A", 0, 2), ("B", "C", 2, 0)]))
    assert list(table.team)[:2] == ["A", "B"]
    assert table.iloc[1].separated_by == "away goals scored"


def test_criteria_are_applied_in_the_regulation_order():
    assert [name for name, _ in CRITERIA] == [
        "goal difference",
        "goals scored",
        "away goals scored",
        "wins",
        "away wins",
        "opponents' points",
        "opponents' goal difference",
        "opponents' goals scored",
    ]


def test_unresolved_ties_are_reported_not_invented():
    # Identical records in every simulatable criterion.
    table = rank_table(results([("A", "C", 1, 0), ("B", "D", 1, 0),
                                ("C", "A", 0, 1), ("D", "B", 0, 1)]))
    # The lower of the two tied clubs carries the flag; the upper one is "top".
    lower = table[table.team.isin(["A", "B"])].iloc[1]
    assert lower.separated_by == "unresolved"
    assert lower.unresolved


def test_club_coefficient_breaks_an_otherwise_unresolved_tie():
    table = rank_table(
        results([("A", "C", 1, 0), ("B", "D", 1, 0), ("C", "A", 0, 1), ("D", "B", 0, 1)]),
        coefficients={"A": 80.0, "B": 90.0},
    )
    assert list(table.team)[:2] == ["B", "A"]
    assert table.iloc[1].separated_by == "club coefficient"
    assert not table.iloc[1].unresolved


def test_ranking_is_deterministic_under_a_seed():
    rows = results([("A", "C", 1, 0), ("B", "D", 1, 0), ("C", "A", 0, 1), ("D", "B", 0, 1)])
    first = rank_table(rows, seed=7)
    second = rank_table(rows, seed=7)
    assert list(first.team) == list(second.team)


@pytest.mark.parametrize("season_label", ["2024-25", "2025-26"])
def test_real_league_phase_tables_place_every_club_in_its_play_off_band(season_label):
    """The strongest check: a wrong criterion order breaks the observed draw.

    Article 19.02 bands clubs 9/10 against 23/24, 11/12 against 21/22, 13/14
    against 19/20 and 15/16 against 17/18. Recompute the real final table from
    the cached results and every observed play-off tie must fall in its band.
    """
    from ucl.data.openfootball import CACHE_DIR, parse_file

    played = parse_file(CACHE_DIR / f"{season_label}_cl.txt",
                        season=season_label, competition="UCL")
    league = played[played.stage.str.startswith("League") & played.played]
    table = rank_table(league.rename(columns={"home": "home", "away": "away"}))
    position = dict(zip(table.team, table.position))

    bands = {(9, 10): (23, 24), (11, 12): (21, 22), (13, 14): (19, 20), (15, 16): (17, 18)}
    play_offs = played[played.stage.str.contains("Playoff", case=False) & played.played]
    seen = set()
    for row in play_offs.itertuples():
        pair = tuple(sorted((position[row.home], position[row.away])))
        if pair in seen:
            continue
        seen.add(pair)
        seeded, unseeded = pair
        band = next((v for k, v in bands.items() if seeded in k), None)
        assert band is not None, f"{seeded} is not a seeded play-off position"
        assert unseeded in band, f"{seeded} v {unseeded} violates Article 19.02"
    assert len(seen) == 8
