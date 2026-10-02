"""Fixture lists must match the published league-phase format, loudly."""

import pandas as pd
import pytest

from ucl.data import fixtures as fx

COLUMNS = fx.COLUMNS


def _real(season_label: str) -> pd.DataFrame:
    return pd.read_csv(fx.fixtures_path(season_label), parse_dates=["date"])


def _pots(season_label: str) -> pd.DataFrame:
    return pd.read_csv(fx.pots_path(season_label))


@pytest.mark.parametrize("season_label", ["2024-25", "2025-26", "2026-27"])
def test_built_fixtures_pass_every_structural_check(season_label):
    assert fx.validate(_real(season_label), _pots(season_label), season_label=season_label) == []


@pytest.mark.parametrize("season_label", ["2024-25", "2025-26", "2026-27"])
def test_shape_is_the_published_format(season_label):
    frame = _real(season_label)
    assert len(frame) == 144
    assert len(set(frame.home) | set(frame.away)) == 36
    assert sorted(frame.matchday.unique()) == list(range(1, 9))


def test_fixture_files_carry_no_results():
    # A schedule that carried scores could leak outcomes into fitting.
    for season_label in ("2024-25", "2025-26", "2026-27"):
        assert list(_real(season_label).columns) == COLUMNS


def test_a_missing_match_is_reported():
    frame = _real("2026-27").iloc[:-1]
    problems = fx.validate(frame, _pots("2026-27"))
    assert any("143 matches, expected 144" in p for p in problems)


def test_a_same_association_pairing_is_reported():
    frame = _real("2026-27").copy()
    row = frame.index[0]
    frame.loc[row, ["away", "away_country"]] = ["Arsenal", "ENG"]
    frame.loc[row, ["home", "home_country"]] = ["Liverpool", "ENG"]
    assert any("same-association pairing" in p for p in fx.validate(frame))


def test_monaco_counts_as_a_french_club():
    frame = pd.DataFrame(
        [{"season": "x", "matchday": 1, "date": pd.Timestamp("2025-09-16"),
          "home": "AS Monaco FC", "away": "Lille", "home_country": "MCO",
          "away_country": "FRA"}],
        columns=COLUMNS,
    )
    assert any("same-association" in p for p in fx.validate(frame))


def test_an_unbalanced_home_away_split_is_reported():
    frame = _real("2026-27").copy()
    frame.loc[frame.index[0], "home"] = frame.loc[frame.index[0], "away"]
    assert any("home /" in p for p in fx.validate(frame))


def test_a_wrong_pot_distribution_is_reported():
    pots = _pots("2026-27").copy()
    pots.loc[pots.index[0], "pot"] = 4
    problems = fx.validate(_real("2026-27"), pots, season_label="2026-27")
    assert any("from pot" in p or "expected 9" in p for p in problems)


def test_bridge_names_resolves_by_schedule():
    source = _real("2024-25")
    # A reference that renames one club: the schedule must still identify it.
    reference = source.copy()
    reference["home"] = reference.home.replace({"Arsenal FC": "Arsenal F.C."})
    reference["away"] = reference.away.replace({"Arsenal FC": "Arsenal F.C."})
    mapping, problems = fx.bridge_names(source, reference)
    assert problems == []
    assert mapping["Arsenal FC"] == "Arsenal F.C."
    assert len(mapping) == 36


def test_wikitext_parsing_reads_a_football_box():
    text = """
===Matchday 1===
{{#invoke:Football box|main
|date       = {{Start date|2026|9|8|df=y}}
|team1      = [[AEK Athens F.C.|AEK Athens]] {{fbaicon|GRE}}
|score      = 1-0
|team2      = {{fbaicon|AUT}} [[LASK]]
}}
"""
    frame = fx.from_wikitext(text, "2026-27")
    assert len(frame) == 1
    row = frame.iloc[0]
    assert (row.home, row.home_country) == ("AEK Athens", "GRE")
    assert (row.away, row.away_country) == ("LASK", "AUT")
    assert row.matchday == 1 and row.date == pd.Timestamp("2026-09-08")
