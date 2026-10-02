"""Bookmaker outright odds, used only as a diagnostic.

The market is never a fitting target: it is read alongside the forecast to show
where the model disagrees with informed opinion, which is how the ratings'
problems surface. Prices are stored as quoted, with their source and date, and
de-vigged on load.

De-vigging is proportional normalisation over the listed teams, which is the
crude method: it assumes the overround is spread evenly across prices. It also
assigns zero to any league-phase club the bookmaker did not list, so the listed
clubs absorb the missing probability and read slightly high.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ucl.config import DATA_RAW

__all__ = ["outright_path", "load_outright"]


def outright_path(season_label: str) -> Path:
    return DATA_RAW / f"outright_odds_{season_label}.csv"


def load_outright(season_label: str = "2026-27", *, path: Path | None = None) -> pd.DataFrame:
    """Load outright odds and add de-vigged probabilities.

    Returns the quoted ``decimal_odds``, the raw implied probability, the
    overround, and ``p_market`` after proportional normalisation.
    """
    frame = pd.read_csv(path or outright_path(season_label))
    frame["p_raw"] = 1.0 / frame.decimal_odds
    overround = float(frame.p_raw.sum())
    frame["p_market"] = frame.p_raw / overround
    frame.attrs["overround"] = overround
    frame.attrs["listed"] = len(frame)
    return frame
