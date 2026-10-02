"""Tests for the no-look-ahead guard. The simulator is only trustworthy if these hold."""

from datetime import date

import pandas as pd
import pytest

from ucl.data.cutoff import LookAheadError, assert_no_lookahead, filter_to_cutoff

CUTOFF = date(2026, 9, 7)


def _frame(*dates: str) -> pd.DataFrame:
    return pd.DataFrame({"date": list(dates), "home": ["A"] * len(dates)})


def test_row_on_the_cutoff_fails() -> None:
    with pytest.raises(LookAheadError, match="on or after the cutoff"):
        assert_no_lookahead(_frame("2026-09-07"), CUTOFF)


def test_day_before_the_cutoff_passes() -> None:
    assert_no_lookahead(_frame("2026-09-06"), CUTOFF)


def test_row_after_the_cutoff_fails() -> None:
    with pytest.raises(LookAheadError):
        assert_no_lookahead(_frame("2026-09-06", "2026-10-02"), CUTOFF)


def test_empty_frame_passes() -> None:
    assert_no_lookahead(_frame(), CUTOFF)


def test_unparseable_date_fails() -> None:
    with pytest.raises(LookAheadError, match="unparseable"):
        assert_no_lookahead(_frame("2026-09-06", "not a date"), CUTOFF)


def test_missing_date_column_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="kickoff"):
        assert_no_lookahead(_frame("2026-09-06"), CUTOFF, date_column="kickoff")


def test_error_message_names_the_source_and_the_offending_date() -> None:
    with pytest.raises(LookAheadError) as excinfo:
        assert_no_lookahead(_frame("2026-09-09"), CUTOFF, source="E0_2627.csv")
    message = str(excinfo.value)
    assert "E0_2627.csv" in message
    assert "2026-09-09" in message


def test_each_season_cutoff_excludes_its_own_matchday_one() -> None:
    from ucl.config import SEASONS

    for cfg in SEASONS.values():
        assert cfg.cutoff_date < cfg.league_phase_start
        with pytest.raises(LookAheadError):
            assert_no_lookahead(
                _frame(cfg.league_phase_start.isoformat()), cfg.cutoff_date
            )


def test_filter_to_cutoff_drops_late_and_unparseable_rows() -> None:
    frame = _frame("2026-09-06", "2026-09-07", "2026-10-02", "")
    kept = filter_to_cutoff(frame, CUTOFF, source="test")
    assert list(kept["date"]) == ["2026-09-06"]


def test_filter_to_cutoff_accepts_dayfirst_strings_as_timestamps() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["06/09/2026"], dayfirst=True)})
    assert len(filter_to_cutoff(frame, CUTOFF)) == 1
