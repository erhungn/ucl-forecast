"""Global, frozen configuration. Single source of truth for the cutoff and seed."""

from datetime import date
from pathlib import Path

# Information set is frozen the day before Matchday 1. No row dated on or after
# this date may be used for fitting; every loader asserts this.
CUTOFF_DATE = date(2026, 9, 7)

# Final: single match at the Riyadh Air Metropolitano (Madrid), neutral venue.
FINAL_DATE = date(2027, 6, 5)

# Global seed for every RNG in the project.
RANDOM_SEED = 20260907

# Default number of Monte Carlo iterations.
N_SIMULATIONS = 50_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
