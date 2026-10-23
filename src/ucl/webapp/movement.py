"""How the forecast moved from one matchday to the next.

``simulate.py`` writes ``d_p_champion``/``d_p_top8``/``d_p_eliminated`` into the
forecast CSV, computed against the previous matchday's file **in** ``outputs/``.
That directory is scratch, so those columns can in principle have been computed
against a smoke-test file. The deltas here are therefore recomputed from the
published ``live/`` snapshots, and :func:`chain_disagreement` reports any row
where the stored column and the recomputed one differ - silent when healthy, a
visible flag when the chain was poisoned.
"""

from __future__ import annotations

import pandas as pd

from ucl.webapp import loaders, paths

__all__ = [
    "TRACKED", "forecast_series", "movement", "biggest_movers",
    "chain_disagreement", "history_series", "history_note",
]

#: The columns worth following across matchdays.
TRACKED = ("p_top8", "p_play_off", "p_eliminated", "p_champion", "mean_position")


def forecast_series(season: str = paths.SEASON, matchdays=None,
                    **kwargs) -> pd.DataFrame:
    """Long frame of ``matchday, team, <TRACKED...>`` over every snapshot.

    Carries a ``retroactive`` flag per row, because a series that mixes
    forecasts made before a round with snapshots re-forecast afterwards is two
    different things drawn as one line. It is False for every published
    snapshot unless a backfill said otherwise.
    """
    if matchdays is None:
        matchdays = paths.available_matchdays(season, **kwargs) or [0]
    backfilled = loaders.retroactive_matchdays(season, **kwargs)
    pieces = []
    for matchday in matchdays:
        frame = loaders.load_forecast(matchday, season, **kwargs)
        piece = frame[["team", *TRACKED]].copy()
        piece.insert(0, "matchday", matchday)
        piece["retroactive"] = matchday in backfilled
        piece["produced_by"] = backfilled.get(matchday, "")
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def history_series(model: str, season: str = paths.SEASON,
                   **kwargs) -> pd.DataFrame:
    """What a superseded model published, read back from its archive.

    On a model switch the earlier model's snapshots are moved into
    ``live/<season>/archive_<model>/`` rather than deleted. They are the record
    of what was published at the time, so they are readable and labelled - but
    never concatenated with the current model's series, because the step
    between two models is not movement in anything.
    """
    matchdays = paths.history_matchdays(model, season, **kwargs)
    pieces = []
    for matchday in matchdays:
        frame = loaders.load_history_forecast(model, matchday, season, **kwargs)
        piece = frame[["team", *TRACKED]].copy()
        piece.insert(0, "matchday", matchday)
        piece.insert(0, "model", model)
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=["model", "matchday", "team", *TRACKED])
    return pd.concat(pieces, ignore_index=True)


def history_note(season: str = paths.SEASON, **kwargs) -> str:
    """One line naming the superseded models that still have snapshots."""
    models = paths.history_models(season, **kwargs)
    if not models:
        return ""
    parts = []
    for model in models:
        days = paths.history_matchdays(model, season, **kwargs)
        span = (f"matchday {days[0]}" if len(days) == 1
                else f"matchdays {days[0]}-{days[-1]}")
        parts.append(f"{model} ({span})")
    return ("Published by an earlier model and kept as the record: "
            + ", ".join(parts))


def movement(season: str = paths.SEASON, matchday: int | None = None,
             **kwargs) -> pd.DataFrame:
    """Change in each club's probabilities from the previous matchday.

    Returns an empty frame when there is no earlier snapshot to compare with.
    """
    available = paths.available_matchdays(season, **kwargs)
    if matchday is None:
        matchday = max(available, default=0)
    earlier = [m for m in available if m < matchday]
    if not earlier:
        return pd.DataFrame(columns=["team"])
    previous_matchday = max(earlier)

    current = loaders.load_forecast(matchday, season, **kwargs)
    previous = loaders.load_forecast(previous_matchday, season, **kwargs)
    merged = current.merge(
        previous[["team", *TRACKED]], on="team", suffixes=("", "_prev"))
    for column in TRACKED:
        # Keep whatever the simulator wrote before overwriting it, so
        # chain_disagreement has something to compare against.
        stored = f"d_{column}"
        if stored in merged.columns:
            merged[f"{stored}_stored"] = merged[stored]
        merged[stored] = merged[column] - merged[f"{column}_prev"]
    merged.attrs["matchday"] = matchday
    merged.attrs["previous_matchday"] = previous_matchday
    return merged


def biggest_movers(moved: pd.DataFrame, column: str = "d_p_top8",
                   count: int = 10) -> pd.DataFrame:
    """The clubs that moved most on ``column``, largest absolute change first."""
    if not len(moved) or column not in moved.columns:
        return moved
    order = moved[column].abs().sort_values(ascending=False).index
    return moved.reindex(order).head(count)


def chain_disagreement(moved: pd.DataFrame, *, tolerance: float = 1e-6
                       ) -> pd.DataFrame:
    """Rows where the forecast's own ``d_*`` column disagrees with ``live/``.

    A non-empty result means ``simulate.py`` computed its deltas against a file
    in ``outputs/`` that was not the published previous matchday - which is what
    happens if the test suite has run since.
    """
    if not len(moved):
        return pd.DataFrame(columns=["team"])
    rows = []
    for column in ("p_top8", "p_eliminated", "p_champion"):
        name = f"d_{column}"
        stored = f"{name}_stored"
        if stored not in moved.columns:
            # Matchday 0 carries no deltas; nothing to disagree with.
            continue
        recomputed = moved[column] - moved[f"{column}_prev"]
        gap = (moved[stored] - recomputed).abs()
        for index in moved.index[gap > tolerance]:
            rows.append({
                "team": moved.at[index, "team"], "column": name,
                "stored": float(moved.at[index, stored]),
                "recomputed": float(recomputed.at[index]),
            })
    return pd.DataFrame(rows, columns=["team", "column", "stored", "recomputed"])
