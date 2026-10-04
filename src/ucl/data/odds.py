"""Bookmaker outright odds, used only as a diagnostic.

The market is never a fitting target: it is read alongside the forecast to show
where the model disagrees with informed opinion, which is how the ratings'
problems surface. Prices are stored as quoted, with their source and date, and
de-vigged on load.

De-vigging is proportional normalisation over the listed teams, which is the
crude method: it assumes the overround is spread evenly across prices. It also
assigns zero to any league-phase club the bookmaker did not list, so the listed
clubs absorb the missing probability and read slightly high.

More than one market can be held at once - a bookmaker and a prediction
exchange price the same tournament very differently, and an exchange's much
smaller overround makes it the better yardstick where it has prices at all.
Each market lives in its own file, is de-vigged on its own, and is held to the
same no-look-ahead rule as the model: a price observed on or after the cutoff
has seen matchday 1 and cannot be used to judge a pre-tournament forecast.

A price the market quoted only as a bound - "under 1 cent" - is kept in the
file for completeness, with an empty ``decimal_odds``, and excluded from the
de-vig. Turning it into a number would be inventing one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ucl.config import DATA_RAW
from ucl.data.cutoff import assert_no_lookahead

__all__ = [
    "outright_path", "outright_paths", "load_outright", "load_all_outrights",
]


def outright_path(season_label: str) -> Path:
    return DATA_RAW / f"outright_odds_{season_label}.csv"


def outright_paths(season_label: str) -> list[Path]:
    """Every outright file for the season: the bare one plus any per-market."""
    bare = outright_path(season_label)
    extra = sorted(DATA_RAW.glob(f"outright_odds_*_{season_label}.csv"))
    return ([bare] if bare.exists() else []) + extra


def load_outright(season_label: str = "2026-27", *, path: Path | None = None) -> pd.DataFrame:
    """Load outright odds and add de-vigged probabilities.

    Returns the quoted ``decimal_odds``, the raw implied probability, the
    overround, and ``p_market`` after proportional normalisation.
    """
    frame = pd.read_csv(path or outright_path(season_label))
    listed = len(frame)
    # A bound rather than a price ("<1c") carries no decimal_odds and cannot be
    # de-vigged; it still counts as listed.
    frame = frame[frame.decimal_odds.notna()].copy()
    frame["p_raw"] = 1.0 / frame.decimal_odds
    overround = float(frame.p_raw.sum())
    frame["p_market"] = frame.p_raw / overround
    frame.attrs["overround"] = overround
    frame.attrs["listed"] = listed
    frame.attrs["priced"] = len(frame)
    return frame


def load_all_outrights(season_label: str = "2026-27", *, cutoff=None
                       ) -> pd.DataFrame:
    """Every market's de-vigged prices in one long frame, with a ``market`` column.

    ``cutoff`` asserts the benchmark's own no-look-ahead rule: pass the season's
    cutoff and any price observed on or after it raises
    :class:`ucl.data.cutoff.LookAheadError`. The market is a diagnostic, but a
    diagnostic computed from post-cutoff prices would flatter or damn the
    forecast for knowing something it did not.
    """
    pieces = []
    for path in outright_paths(season_label):
        frame = load_outright(season_label, path=path)
        market = frame.source.iloc[0] if len(frame) else path.stem
        frame["market"] = market
        frame.attrs["path"] = path.name
        if cutoff is not None:
            assert_no_lookahead(frame, cutoff, date_column="odds_date",
                                source=f"outright odds ({market})")
        pieces.append(frame)
    if not pieces:
        raise FileNotFoundError(
            f"no outright odds files for {season_label} in {DATA_RAW}")
    combined = pd.concat(pieces, ignore_index=True)
    combined.attrs["markets"] = {
        piece.market.iloc[0]: {"overround": piece.attrs["overround"],
                               "listed": piece.attrs["listed"],
                               "priced": piece.attrs["priced"],
                               "path": piece.attrs["path"]}
        for piece in pieces
    }
    return combined
