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
#: Published artefacts of the weekly update, and the only thing the local app
#: reads. Separate from ``outputs/`` because that directory is scratch: the
#: test suite runs the simulator into it with a handful of iterations.
LIVE_DIR = PROJECT_ROOT / "live"

# Global seed for every RNG in the project.
RANDOM_SEED = 20260907

# Default number of Monte Carlo iterations.
N_SIMULATIONS = 50_000

# --- which model is published ---------------------------------------------
#: The models that have been *published*, oldest first. The last one is what
#: ``scripts/weekly_update.py`` publishes by default, what the app falls back
#: on for matchday 0, and what ``ucl.tracking`` measures every other model
#: against. Promoting a model is appending to this tuple and running the
#: weekly update once; ``tests/test_promotion.py`` pins every other place that
#: has to agree, so a half-done promotion fails the suite rather than
#: publishing a forecast whose provenance nobody can read afterwards.
#:
#: This is deliberately NOT the list of models the code can build - see
#: ``ucl.models.forecast.MODELS`` for that. A model has to be buildable and
#: runnable (and rehearsable, with ``weekly_update.py --live-dir``) before it
#: is published, so the two lists differ in exactly the window that matters.
MODEL_HISTORY: tuple[str, ...] = ("v1", "v2")

#: The published model. One name, read by everything.
PUBLISHED_MODEL = MODEL_HISTORY[-1]

# --- Elo ------------------------------------------------------------------
# Tuned on the backtest seasons in milestone 5; these are the starting points.

START_RATING = 1500.0

#: Added to the home club's rating when forming the Elo expectation. This is the
#: Elo-update home advantage only. The Poisson match model must NOT reuse it:
#: its own intercept `h` carries home advantage, and counting both would double it.
HOME_ADVANTAGE = 65.0

#: Tuning on 2024-25 preferred 15 and 8, but the 2025-26 holdout did not
#: confirm it and no difference was statistically significant, so the
#: pre-registered protocol rejects the tuning and these stay as they were.
#: See outputs/backtest_summary.csv and the milestone 2.5 report.
K_DOMESTIC = 20.0
K_EUROPEAN = 25.0

#: A club first seen in a league starts below that league's mean, since most
#: arrive by promotion. Clubs whose association has no cached league start at
#: their association baseline instead, with no penalty to apply.
NEW_CLUB_OFFSET = -50.0

#: Fraction of a club's deviation from its regression target carried into the
#: next season; 0.90 keeps most of it, because squads persist. Tuning preferred
#: 0.95, rejected on the holdout.
SEASON_CARRYOVER = 0.90

#: Weight of an update by margin, as (1 + goal difference) ** exponent; None is
#: the World Football Elo rule. Tuning preferred 0.0 - margin ignored entirely -
#: and that was the single clearest qualitative fix, dropping Como from 19th to
#: 32nd, but the holdout did not confirm it, so the rule stands until a
#: multi-season backtest can resolve it.
GD_EXPONENT: float | None = None

#: What a club regresses toward between seasons: its league's mean.
CARRYOVER_TARGET = "league_mean"

#: Burn-in start. The extra-league files reach back this far.
ELO_BURN_IN_START = date(2012, 1, 1)

# --- Baseline Poisson match model -----------------------------------------
# log(lambda_home) = a + h + b * d,  log(lambda_away) = a - b * d,
# where d = (elo_home - elo_away) / 400 WITHOUT the Elo home advantage.
POISSON_FALLBACK = {"a": 0.3001, "h": 0.25, "b": 1.0}

# --- Knockout -------------------------------------------------------------
EXTRA_TIME_SCALE = 1.0 / 3.0
#: Shoot-out win probability shifts this much per Elo point of difference.
PENALTY_TILT = 0.0004
PENALTY_TILT_CLIP = (0.35, 0.65)


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
    #: "league_phase" (36 teams, one table, from 2024-25) or "group_stage"
    #: (32 teams in 8 groups, before that). The simulator only handles the
    #: league phase; the group-stage seasons exist for the ratings backtest,
    #: where only `cutoff_date` is used, and have no fixtures file.
    tournament_format: str = "league_phase"


def _fixtures(label: str) -> Path:
    return DATA_RAW / f"fixtures_{label}.csv"


SEASONS: dict[str, SeasonConfig] = {
    "2021-22": SeasonConfig(
        label="2021-22",
        cutoff_date=date(2021, 9, 13),
        league_phase_start=date(2021, 9, 14),
        league_phase_end=date(2021, 12, 9),
        fixtures_path=_fixtures("2021-22"),
        tournament_format="group_stage",
    ),
    "2022-23": SeasonConfig(
        label="2022-23",
        cutoff_date=date(2022, 9, 5),
        league_phase_start=date(2022, 9, 6),
        league_phase_end=date(2022, 11, 2),
        fixtures_path=_fixtures("2022-23"),
        tournament_format="group_stage",
    ),
    "2023-24": SeasonConfig(
        label="2023-24",
        cutoff_date=date(2023, 9, 18),
        league_phase_start=date(2023, 9, 19),
        league_phase_end=date(2023, 12, 13),
        fixtures_path=_fixtures("2023-24"),
        tournament_format="group_stage",
    ),
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

#: Seasons the ratings backtest scores, oldest first.
BACKTEST_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")

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
