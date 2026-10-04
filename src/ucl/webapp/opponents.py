"""Who a club might meet, from the simulated knockout pairings.

The simulator draws the bracket inside the Article 19 constraints **once per
iteration**, so bracket position 3 is a different fixture in every one of the
50,000 runs. That makes "the most likely occupant of slot 3" a number with no
referent, and it is deliberately not computed here. What the simulation can say
honestly is P(club A meets club B in round R), which is what this module
reports.

No Streamlit and no Altair: this runs in the model virtualenv too, where
neither is installed.
"""

from __future__ import annotations

import pandas as pd

from ucl.sim.knockout import ROUNDS

__all__ = ["ROUNDS", "TIES_PER_ROUND", "LABELS", "iterations", "reach",
           "per_round", "likeliest"]

#: Ties played in each round per iteration, fixed by the competition format:
#: 16 clubs in the play-off, 16 in the round of 16, then 8, 4, 2. Used to turn
#: the stored counts into probabilities without trusting a status file. The
#: tests check the result against the stage probabilities the forecast
#: publishes, which is the real guarantee.
TIES_PER_ROUND = {"play_off": 8, "round_of_16": 8, "quarter_finals": 4,
                  "semi_finals": 2, "final": 1}

LABELS = {"play_off": "Play-off", "round_of_16": "Round of 16",
          "quarter_finals": "Quarter-final", "semi_finals": "Semi-final",
          "final": "Final"}


def iterations(frame: pd.DataFrame) -> int:
    """The iteration count the pairings were counted over.

    Every tie contributes two rows, so a round's counts sum to twice its ties
    times the iteration count. Each round gives the same answer, and they are
    cross-checked here rather than one being trusted.
    """
    found = set()
    for round_name, group in frame.groupby("round"):
        ties = TIES_PER_ROUND.get(str(round_name))
        if ties:
            found.add(int(group["count"].sum()) // (2 * ties))
    if len(found) != 1:
        raise ValueError(f"rounds disagree about the iteration count: {sorted(found)}")
    return found.pop()


def reach(frame: pd.DataFrame, club: str) -> pd.DataFrame:
    """P(``club`` plays in each round), in competition order.

    A club plays exactly one tie in a round it reaches, so this is also its
    probability of reaching that round - which is why it has to agree with the
    forecast's own stage columns.
    """
    total = iterations(frame)
    mine = frame[frame.club == club]
    rows = []
    for round_name in ROUNDS:
        count = int(mine.loc[mine["round"] == round_name, "count"].sum())
        rows.append({"round": round_name, "label": LABELS[round_name],
                     "probability": count / total})
    return pd.DataFrame(rows)


def per_round(frame: pd.DataFrame, club: str, round_name: str, *,
              count: int = 6) -> pd.DataFrame:
    """``club``'s likeliest opponents in one round, most likely first.

    The probabilities are unconditional - P(this tie happens at all) - so they
    sum to P(the club plays in the round), not to 1. Conditioning on reaching
    the round would hide the thing most worth seeing, which is that a club with
    a 12% chance of a quarter-final cannot have a likely quarter-final
    opponent.
    """
    total = iterations(frame)
    mine = frame[(frame.club == club) & (frame["round"] == round_name)]
    out = mine.nlargest(count, "count")[["opponent", "count"]].copy()
    out["probability"] = out["count"] / total
    return out.reset_index(drop=True)


def likeliest(frame: pd.DataFrame, round_name: str, *,
              count: int = 8) -> pd.DataFrame:
    """The likeliest ties in a round, each listed once rather than twice.

    Both directions of every tie are stored, so the pair is normalised before
    de-duplicating; otherwise every tie would appear twice in the table.
    """
    mine = frame[frame["round"] == round_name].copy()
    if mine.empty:
        return mine.assign(probability=[])
    total = iterations(frame)
    pair = mine[["club", "opponent"]].apply(
        lambda row: " v ".join(sorted((row.club, row.opponent))), axis=1)
    mine["tie"] = pair
    unique = mine.drop_duplicates("tie").nlargest(count, "count")
    unique["probability"] = unique["count"] / total
    return unique[["tie", "club", "opponent", "probability"]].reset_index(drop=True)
