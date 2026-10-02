"""League-phase ranking, per Article 18 of the 2026/27 regulations.

The criteria, in order, verbatim from the official PDF:

    a. Superior goal difference in the league phase
    b. Higher number of goals scored in the league phase
    c. Higher number of away goals scored in the league phase
    d. Higher number of wins in the league phase
    e. Higher number of away wins in the league phase
    f. Higher number of points obtained collectively by league phase opponents
    g. Superior collective goal difference of league phase opponents
    h. Higher number of goals scored collectively by league phase opponents
    i. Lower disciplinary points total ... (red card = 3 points, yellow card = 1
       point, expulsion for two yellow cards in one match = 3 points)
    j. Higher club coefficient (see Annex D)

All of a-h are league-phase totals, not head-to-head, so one sort key decides
the table. Criterion i cannot be simulated from scorelines - we model goals, not
cards - and j needs the Annex D coefficient, which the repo has only as a
seeding pot. So a tie surviving h is marked unresolved and broken by the club
coefficient when one is supplied, otherwise by a seeded shuffle that is recorded
rather than hidden.
"""

from __future__ import annotations

import pandas as pd

from ucl import config

__all__ = ["CRITERIA", "team_table", "rank_table"]

#: (name, column) in the order Article 18 applies them. All are "higher is better".
CRITERIA: list[tuple[str, str]] = [
    ("goal difference", "gd"),
    ("goals scored", "gf"),
    ("away goals scored", "away_gf"),
    ("wins", "wins"),
    ("away wins", "away_wins"),
    ("opponents' points", "opp_points"),
    ("opponents' goal difference", "opp_gd"),
    ("opponents' goals scored", "opp_gf"),
]


def team_table(results: pd.DataFrame) -> pd.DataFrame:
    """Per-club league-phase totals, including the three opponent aggregates."""
    stats: dict[str, dict[str, int]] = {}
    opponents: dict[str, list[str]] = {}

    def slot(team: str) -> dict[str, int]:
        return stats.setdefault(
            team,
            {"played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0,
             "away_gf": 0, "away_wins": 0, "points": 0},
        )

    for row in results.itertuples():
        home, away = slot(row.home), slot(row.away)
        opponents.setdefault(row.home, []).append(row.away)
        opponents.setdefault(row.away, []).append(row.home)
        hg, ag = int(row.hg), int(row.ag)
        home["played"] += 1
        away["played"] += 1
        home["gf"] += hg
        home["ga"] += ag
        away["gf"] += ag
        away["ga"] += hg
        away["away_gf"] += ag
        if hg > ag:
            home["wins"] += 1
            home["points"] += 3
            away["losses"] += 1
        elif hg < ag:
            away["wins"] += 1
            away["away_wins"] += 1
            away["points"] += 3
            home["losses"] += 1
        else:
            home["draws"] += 1
            away["draws"] += 1
            home["points"] += 1
            away["points"] += 1

    table = pd.DataFrame(stats).T.reset_index(names="team")
    table["gd"] = table.gf - table.ga

    points = dict(zip(table.team, table.points))
    gd = dict(zip(table.team, table.gd))
    gf = dict(zip(table.team, table.gf))
    table["opp_points"] = table.team.map(lambda t: sum(points[o] for o in opponents[t]))
    table["opp_gd"] = table.team.map(lambda t: sum(gd[o] for o in opponents[t]))
    table["opp_gf"] = table.team.map(lambda t: sum(gf[o] for o in opponents[t]))
    return table


def rank_table(
    results: pd.DataFrame,
    *,
    coefficients: dict[str, float] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Rank the league phase, recording which criterion separated each club.

    ``coefficients`` supplies Annex D club coefficients for criterion j. Without
    them a tie surviving criterion h is broken by a seeded shuffle and flagged in
    the ``unresolved`` column, never silently resolved.
    """
    table = team_table(results)
    keys = ["points"] + [column for _, column in CRITERIA]

    rng_order = pd.Series(range(len(table)), index=table.index)
    if seed is not None or coefficients is None:
        import numpy as np

        rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
        rng_order = pd.Series(rng.permutation(len(table)), index=table.index)
    table["_coefficient"] = table.team.map(coefficients or {}).fillna(0.0)
    table["_tiebreak"] = rng_order

    ordered = table.sort_values(
        keys + ["_coefficient", "_tiebreak"],
        ascending=[False] * (len(keys) + 1) + [True],
    ).reset_index(drop=True)

    # Which criterion separated each club from the one above it.
    separated, unresolved = [], []
    for position in range(len(ordered)):
        if position == 0:
            separated.append("top")
            unresolved.append(False)
            continue
        above, here = ordered.iloc[position - 1], ordered.iloc[position]
        if above.points != here.points:
            separated.append("points")
            unresolved.append(False)
            continue
        reason = None
        for name, column in CRITERIA:
            if above[column] != here[column]:
                reason = name
                break
        if reason is None:
            if above._coefficient != here._coefficient:
                reason = "club coefficient"
            else:
                reason = "unresolved"
        separated.append(reason)
        unresolved.append(reason == "unresolved")

    ordered["separated_by"] = separated
    ordered["unresolved"] = unresolved
    ordered["position"] = range(1, len(ordered) + 1)
    return ordered.drop(columns=["_coefficient", "_tiebreak"])
