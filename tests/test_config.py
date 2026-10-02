"""Smoke test: the package imports and the frozen configuration is what we expect."""

from datetime import date

from ucl import config


def test_cutoff_date_is_frozen() -> None:
    assert config.CUTOFF_DATE == date(2026, 9, 7)
    assert isinstance(config.RANDOM_SEED, int)
