"""Global, frozen configuration.

The cutoff date is per season and is passed explicitly into every loader and
check. Nothing in :mod:`ucl.data` or :mod:`ucl.models` may read a module-level
cutoff: a global would make a backtest silently fit on post-cutoff data.
``CUTOFF_DATE`` below is the 2026-27 default for scripts only.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Global seed for every RNG in the project.
RANDOM_SEED = 20260907

# Default number of Monte Carlo iterations.
N_SIMULATIONS = 50_000


@dataclass(frozen=True)
class SeasonConfig:
    """One tournament edition and the information set frozen for it.

    ``cutoff_date`` is the day before matchday 1: no row dated on or after it
    may be used for fitting. Pass it explicitly; never read it from a global.
    """

    label: str
    cutoff_date: date
    league_phase_start: date
    league_phase_end: date
    fixtures_path: Path


def _fixtures(label: str) -> Path:
    return DATA_RAW / f"fixtures_{label}.csv"


SEASONS: dict[str, SeasonConfig] = {
    "2024-25": SeasonConfig(
        label="2024-25",
        cutoff_date=date(2024, 9, 16),
        league_phase_start=date(2024, 9, 17),
        league_phase_end=date(2025, 1, 29),
        fixtures_path=_fixtures("2024-25"),
    ),
    "2025-26": SeasonConfig(
        label="2025-26",
        cutoff_date=date(2025, 9, 15),
        league_phase_start=date(2025, 9, 16),
        league_phase_end=date(2026, 1, 28),
        fixtures_path=_fixtures("2025-26"),
    ),
    "2026-27": SeasonConfig(
        label="2026-27",
        cutoff_date=date(2026, 9, 7),
        league_phase_start=date(2026, 9, 8),
        league_phase_end=date(2027, 1, 27),
        fixtures_path=_fixtures("2026-27"),
    ),
}

DEFAULT_SEASON = "2026-27"

# 2026-27 default, for scripts and reports. Loaders take a cutoff argument.
CUTOFF_DATE = SEASONS[DEFAULT_SEASON].cutoff_date

# Final: single match at the Riyadh Air Metropolitano (Madrid), neutral venue.
FINAL_DATE = date(2027, 6, 5)


def season(label: str | None = None) -> SeasonConfig:
    """Return the configuration for ``label``, defaulting to 2026-27."""
    key = label or DEFAULT_SEASON
    if key not in SEASONS:
        raise KeyError(f"unknown season {key!r}; known: {sorted(SEASONS)}")
    return SEASONS[key]
