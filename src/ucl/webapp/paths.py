"""Where the published artefacts live, and which matchdays exist.

``live/<season>/`` is written only by ``scripts/weekly_update.py``. It exists
because ``outputs/`` is scratch: the test suite runs the simulator into it with
a handful of iterations, so a file there may be a 5-iteration smoke test rather
than a forecast.
"""

from __future__ import annotations

import re
from pathlib import Path

from ucl.config import LIVE_DIR, PROJECT_ROOT

__all__ = [
    "LIVE_DIR", "ARCHIVE_DIR", "SEASON", "season_dir", "forecast_path",
    "next_matchday_path", "evaluation_path", "information_path", "status_path",
    "log_path", "available_matchdays", "latest_matchday", "scored_matchdays",
]

#: The frozen pre-tournament archive, used as the matchday-0 fallback so the
#: app works before the first weekly run.
ARCHIVE_DIR = PROJECT_ROOT / "results" / "model_v1"
SEASON = "2026-27"

_FORECAST = re.compile(r"^forecast_md(\d+)\.csv$")
_EVALUATION = re.compile(r"^matchday_(\d+)_evaluation\.csv$")
_NEXT = re.compile(r"^next_matchday_(\d+)\.csv$")


def season_dir(season: str = SEASON, *, live_dir: Path | None = None) -> Path:
    return (Path(live_dir) if live_dir is not None else LIVE_DIR) / season


def forecast_path(matchday: int, season: str = SEASON, **kwargs) -> Path:
    return season_dir(season, **kwargs) / f"forecast_md{matchday}.csv"


def next_matchday_path(matchday: int, season: str = SEASON, **kwargs) -> Path:
    return season_dir(season, **kwargs) / f"next_matchday_{matchday}.csv"


def evaluation_path(matchday: int, season: str = SEASON, **kwargs) -> Path:
    return season_dir(season, **kwargs) / f"matchday_{matchday}_evaluation.csv"


def information_path(season: str = SEASON, **kwargs) -> Path:
    return season_dir(season, **kwargs) / "information_set.csv"


def status_path(season: str = SEASON, **kwargs) -> Path:
    return season_dir(season, **kwargs) / "status.json"


def log_path(season: str = SEASON, **kwargs) -> Path:
    return season_dir(season, **kwargs) / "update.log"


def _numbers(directory: Path, pattern: re.Pattern[str]) -> list[int]:
    if not directory.is_dir():
        return []
    found = (pattern.match(path.name) for path in directory.iterdir())
    return sorted(int(match.group(1)) for match in found if match)


def available_matchdays(season: str = SEASON, **kwargs) -> list[int]:
    """Matchdays with a published forecast, ascending. Numeric, not lexical."""
    return _numbers(season_dir(season, **kwargs), _FORECAST)


def scored_matchdays(season: str = SEASON, **kwargs) -> list[int]:
    """Matchdays with a published evaluation."""
    return _numbers(season_dir(season, **kwargs), _EVALUATION)


def published_next_matchdays(season: str = SEASON, **kwargs) -> list[int]:
    """Matchdays with a published pre-match 1X2 table."""
    return _numbers(season_dir(season, **kwargs), _NEXT)


def latest_matchday(season: str = SEASON, **kwargs) -> int:
    """The newest published forecast, or 0 when only the archive exists."""
    return max(available_matchdays(season, **kwargs), default=0)
