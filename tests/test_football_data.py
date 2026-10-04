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


def test_the_shot_columns_are_kept_for_the_main_leagues_only():
    """Shots, shots on target and corners: present in 11 leagues, nowhere else.

    The extra-league schema is goals and closing odds alone, so those files get
    the columns as NaN to keep one frame shape. Without this the GAP candidate
    would silently rate a Norwegian club on no data at all.
    """
    import pandas as pd

    from ucl.data.football_data import SHOT_COLUMNS

    frame = load_results(date(2027, 7, 1))
    columns = list(SHOT_COLUMNS.values())
    assert columns == ["shots_home", "shots_away", "sot_home", "sot_away",
                       "corners_home", "corners_away"]
    assert all(column in frame.columns for column in columns)

    complete = frame[columns].notna().all(axis=1)
    assert complete.sum() == 33930

    main = frame.source_file.str.match(r"^[A-Z]+\d_")
    assert complete[~main].sum() == 0, "extra leagues have no shot data"
    assert complete[main].mean() > 0.95

    # All six are non-null on the same rows, so one coverage figure covers them.
    for column in columns:
        assert frame[column].notna().sum() == 33930


def test_the_two_schemas_produce_the_same_columns():
    extra = load_results(date(2027, 7, 1), codes=["NOR"])
    main = load_results(date(2027, 7, 1), codes=["E0"])
    assert list(extra.columns) == list(main.columns)


def test_shots_on_target_exceed_shots_in_exactly_one_known_row():
    """One football-data typo in 33,930 matches, and it must stay known.

    Shots off target is shots minus shots on target, so this row would make it
    -1. Newcastle v West Ham, 2021-08-15: 8 away shots, 9 on target. Anything
    that derives a count from these columns has to floor it at zero.
    """
    frame = load_results(date(2027, 7, 1))
    bad = frame[(frame.sot_home > frame.shots_home)
                | (frame.sot_away > frame.shots_away)]
    assert len(bad) == 1
    row = bad.iloc[0]
    assert (row.home, row.away) == ("Newcastle", "West Ham")
    assert row.date.date() == date(2021, 8, 15)


def test_the_shot_counts_are_plausible():
    frame = load_results(date(2027, 7, 1))
    for column in ("shots_home", "shots_away", "sot_home", "sot_away",
                   "corners_home", "corners_away"):
        values = frame[column].dropna()
        assert (values >= 0).all()
        assert values.max() < 60, column
        assert (values == values.round()).all(), f"{column} should be whole"
