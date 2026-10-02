"""The cached domestic results must load, normalise, and respect the cutoff."""

from datetime import date

from ucl.config import season
from ucl.data.football_data import available_files, load_results


def test_cache_has_both_schemas():
    files = available_files()
    assert len(files["main"]) == 66
    assert len(files["extra"]) == 7


def test_october_rows_of_the_current_season_are_filtered_out():
    # E0_2627.csv was downloaded in October 2026 and runs past the cutoff.
    cfg = season("2026-27")
    frame = load_results(cfg.cutoff_date, codes=["E0"])
    assert frame.date.max().date() < cfg.cutoff_date
    assert frame.date.max().date() >= date(2026, 8, 1)


def test_backtest_cutoffs_see_strictly_less_data():
    counts = {
        label: len(load_results(season(label).cutoff_date, codes=["E0"]))
        for label in ("2024-25", "2025-26", "2026-27")
    }
    assert counts["2024-25"] < counts["2025-26"] < counts["2026-27"]


def test_normalised_columns_and_types():
    frame = load_results(season("2024-25").cutoff_date, codes=["E0", "NOR"])
    assert {"date", "home", "away", "hg", "ag", "country", "league", "season"} <= set(frame.columns)
    assert frame.hg.dtype.kind == "i"
    assert set(frame.country) == {"ENG", "NOR"}
    assert frame.psch.notna().any()
