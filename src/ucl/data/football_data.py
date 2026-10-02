"""Domestic league results from the football-data.co.uk cache.

This loader never touches the network: the site is DNS-blocked on the author's
machine, so the CSVs were fetched once by hand and recorded in
``data/raw/football_data/manifest.json``. Every load re-verifies their hashes.

Two schemas live in the cache and are normalised into one frame here:

* main leagues, one file per division and season (``E0_2425.csv``), with
  ``Div, Date, HomeTeam, AwayTeam, FTHG, FTAG`` and a wide block of odds;
* "extra" leagues, one file per country covering all seasons (``NOR_all.csv``),
  with ``Country, League, Season, Date, Home, Away, HG, AG`` and closing odds only.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ucl.config import DATA_RAW
from ucl.data.cutoff import filter_to_cutoff
from ucl.data.manifest import verify_cache

__all__ = ["CACHE_DIR", "MAIN_LEAGUES", "EXTRA_LEAGUES", "load_results", "available_files"]

CACHE_DIR = DATA_RAW / "football_data"

# football-data division code -> (country code, league name)
MAIN_LEAGUES: dict[str, tuple[str, str]] = {
    "E0": ("ENG", "Premier League"),
    "SC0": ("SCO", "Premiership"),
    "D1": ("GER", "Bundesliga"),
    "I1": ("ITA", "Serie A"),
    "SP1": ("ESP", "LaLiga"),
    "F1": ("FRA", "Ligue 1"),
    "N1": ("NED", "Eredivisie"),
    "B1": ("BEL", "Pro League"),
    "P1": ("POR", "Primeira Liga"),
    "T1": ("TUR", "Super Lig"),
    "G1": ("GRE", "Super League"),
}

# extra-leagues file prefix -> country code
EXTRA_LEAGUES: dict[str, str] = {
    "AUT": "AUT",
    "DNK": "DEN",
    "NOR": "NOR",
    "POL": "POL",
    "ROU": "ROU",
    "SWE": "SWE",
    "SWZ": "SUI",
}

# Closing odds kept for the de-vigged bookmaker benchmark; absent in older files.
ODDS_COLUMNS = ["PSCH", "PSCD", "PSCA", "B365CH", "B365CD", "B365CA", "AvgCH", "AvgCD", "AvgCA"]

COLUMNS = [
    "date", "home", "away", "hg", "ag", "country", "league", "season",
    "competition", "source_file",
]


def _season_label(yy: str) -> str:
    """``'2425'`` -> ``'2024-25'``."""
    return f"20{yy[:2]}-{yy[2:]}"


def _parse_dates(raw: pd.Series) -> pd.Series:
    """football-data uses d/m/Y, with d/m/y in some older files."""
    parsed = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    short = parsed.isna() & raw.notna()
    if short.any():
        parsed.loc[short] = pd.to_datetime(
            raw[short], format="%d/%m/%y", errors="coerce"
        )
    return parsed


def _read(path: Path) -> pd.DataFrame:
    # The files are Windows-encoded and a few carry blank trailing rows.
    return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False)


def available_files(cache_dir: Path | None = None) -> dict[str, list[str]]:
    """Group the cached file names into ``{'main': [...], 'extra': [...]}``."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    main, extra = [], []
    for path in sorted(cache_dir.glob("*.csv")):
        (extra if path.stem.endswith("_all") else main).append(path.name)
    return {"main": main, "extra": extra}


def _load_main(path: Path) -> pd.DataFrame:
    code, yy = path.stem.split("_")
    country, league = MAIN_LEAGUES.get(code, ("???", code))
    raw = _read(path)
    frame = pd.DataFrame(
        {
            "date": _parse_dates(raw["Date"]),
            "home": raw["HomeTeam"],
            "away": raw["AwayTeam"],
            "hg": pd.to_numeric(raw["FTHG"], errors="coerce"),
            "ag": pd.to_numeric(raw["FTAG"], errors="coerce"),
            "country": country,
            "league": league,
            "season": _season_label(yy),
            "competition": "domestic",
            "source_file": path.name,
        }
    )
    for column in ODDS_COLUMNS:
        frame[column.lower()] = pd.to_numeric(raw.get(column), errors="coerce")
    for source, target in (("HxG", "hxg"), ("AxG", "axg")):
        frame[target] = pd.to_numeric(raw.get(source), errors="coerce")
    return frame


def _load_extra(path: Path) -> pd.DataFrame:
    prefix = path.stem.split("_")[0]
    raw = _read(path)
    seasons = raw["Season"].astype(str)
    frame = pd.DataFrame(
        {
            "date": _parse_dates(raw["Date"]),
            "home": raw["Home"],
            "away": raw["Away"],
            "hg": pd.to_numeric(raw["HG"], errors="coerce"),
            "ag": pd.to_numeric(raw["AG"], errors="coerce"),
            "country": EXTRA_LEAGUES.get(prefix, prefix),
            "league": raw["League"],
            # Calendar-year leagues report '2026'; split-year ones '2026/2027'.
            "season": seasons.str.replace("/20", "-", regex=False).str.replace(
                "/", "-", regex=False
            ),
            "competition": "domestic",
            "source_file": path.name,
        }
    )
    for column in ODDS_COLUMNS:
        frame[column.lower()] = pd.to_numeric(raw.get(column), errors="coerce")
    # No shot data in the extra-leagues files; keep the column shape identical.
    frame["hxg"] = np.nan
    frame["axg"] = np.nan
    return frame


def load_results(
    cutoff: date,
    *,
    cache_dir: Path | None = None,
    codes: list[str] | None = None,
    verify: bool = True,
) -> pd.DataFrame:
    """Load every cached domestic result dated strictly before ``cutoff``.

    ``cutoff`` is required and explicit: pass ``SeasonConfig.cutoff_date``, never
    a module-level constant. ``codes`` optionally restricts to division codes
    (``'E0'``) or extra-league prefixes (``'NOR'``).
    """
    cache_dir = Path(cache_dir or CACHE_DIR)
    names = available_files(cache_dir)
    chosen = [
        *(n for n in names["main"] if codes is None or n.split("_")[0] in codes),
        *(n for n in names["extra"] if codes is None or n.split("_")[0] in codes),
    ]
    if not chosen:
        raise FileNotFoundError(f"no cached football-data files in {cache_dir} for codes={codes}")
    if verify:
        verify_cache(cache_dir, files=chosen)

    frames = [
        _load_extra(cache_dir / name) if name.endswith("_all.csv") else _load_main(cache_dir / name)
        for name in chosen
    ]
    combined = pd.concat(frames, ignore_index=True)

    # Rows without a score are fixtures, not results.
    combined = combined.dropna(subset=["home", "away", "hg", "ag"])
    combined = filter_to_cutoff(combined, cutoff, source=f"football-data cache ({len(chosen)} files)")
    combined["hg"] = combined["hg"].astype(int)
    combined["ag"] = combined["ag"].astype(int)

    ordered = COLUMNS + [c for c in combined.columns if c not in COLUMNS]
    return combined[ordered].sort_values(["date", "country", "home"]).reset_index(drop=True)
