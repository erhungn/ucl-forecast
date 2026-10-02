"""The Transfermarkt club mapping, and the point-in-time guarantees.

The mapping is the part that went wrong once already: the first version matched
exact normalised names only and silently left 10 of the 36 league-phase teams
unvalued, which is indistinguishable from "the covariate does not help".
"""

from __future__ import annotations

import collections
from datetime import date

import pandas as pd
import pytest

from ucl.data import names, transfermarkt


@pytest.fixture(scope="module")
def mapping() -> pd.DataFrame:
    return pd.read_csv(transfermarkt.MAP_PATH)


def test_every_transfermarkt_club_maps_to_at_most_one_id(mapping):
    assert not mapping.tm_club_id.duplicated().any()


def test_one_club_one_id(mapping):
    """Check B: two Transfermarkt clubs sharing an id would merge two squads."""
    counts = collections.Counter(mapping.canonical)
    assert [c for c, n in counts.items() if n > 1] == []


def test_every_mapped_club_exists_in_the_cached_club_table(mapping):
    clubs = pd.read_csv(transfermarkt.CACHE_DIR / "clubs.csv")
    assert set(mapping.tm_club_id) <= set(clubs.club_id.astype(int))


def test_reviewed_decisions_are_honoured():
    import sys

    sys.path.insert(0, "scripts")
    from build_transfermarkt_map import load_decisions

    decisions = load_decisions()
    mapped = transfermarkt.club_map()
    for row in decisions.itertuples():
        if row.kind == "distinct-club":
            assert int(row.tm_club_id) not in mapped
        else:
            assert mapped[int(row.tm_club_id)] == row.canonical


def test_the_league_phase_is_mapped_except_the_two_clubs_the_source_lacks():
    """Sabah and Slovan Bratislava are absent from the club table entirely."""
    teams = set(names.load_teams().team)
    mapped = set(transfermarkt.club_map().values())
    assert sorted(teams - mapped) == ["Sabah", "Slovan Bratislava"]


def test_two_mapped_clubs_are_still_unvalued_because_appearances_run_out():
    """Mapped is not the same as valued, and the gap is the source's, not ours.

    ``appearances.csv`` covers a club only while it plays in a competition the
    dataset follows, so LASK and Viking - mapped, and present in the club table -
    have no appearance inside the 180-day membership window ending 2026-07-06.
    Widening that window would change the frozen specification, so they fall
    back to Elo like any unvalued club.
    """
    values = transfermarkt.squad_values(transfermarkt.as_of_for_season("2026-27"))
    teams = set(names.load_teams().team)
    assert sorted(teams - set(values)) == ["LASK", "Sabah", "Slovan Bratislava", "Viking"]


def test_the_as_of_date_is_the_sources_freeze_date_not_the_cutoff():
    assert transfermarkt.as_of_for_season("2026-27") == date(2026, 7, 6)
    assert transfermarkt.as_of_for_season("2021-22") == date(2021, 7, 6)


def test_no_valuation_or_appearance_on_or_after_the_as_of_date():
    valuations, appearances = transfermarkt._tables()
    as_of = pd.Timestamp(transfermarkt.as_of_for_season("2023-24"))
    used_valuations = valuations[valuations.date < as_of]
    used_appearances = appearances[
        (appearances.date < as_of)
        & (appearances.date >= as_of - pd.Timedelta(transfermarkt.MEMBERSHIP_WINDOW_DAYS, "D"))
    ]
    assert used_valuations.date.max() < as_of
    assert used_appearances.date.max() < as_of
    # And the filter is not vacuous: later rows do exist in the source.
    assert (valuations.date >= as_of).any()


def test_an_earlier_as_of_is_unmoved_by_later_data():
    """The 2022-23 squad values must not depend on anything after their as-of."""
    early = transfermarkt.squad_values(transfermarkt.as_of_for_season("2022-23"))
    valuations, appearances = transfermarkt._tables()
    as_of = pd.Timestamp(transfermarkt.as_of_for_season("2022-23"))
    trimmed_valuations = valuations[valuations.date < as_of]
    trimmed_appearances = appearances[appearances.date < as_of]
    transfermarkt._tables.cache_clear()
    transfermarkt.squad_values.cache_clear()
    try:
        transfermarkt._tables.__wrapped__ = None  # guard against silent reuse
        recomputed = _squad_values_from(trimmed_valuations, trimmed_appearances, as_of)
    finally:
        del transfermarkt._tables.__wrapped__
        transfermarkt._tables.cache_clear()
        transfermarkt.squad_values.cache_clear()
    assert recomputed == pytest.approx(early)


def _squad_values_from(valuations, appearances, as_of) -> dict[str, float]:
    """Recompute squad values from pre-trimmed tables, as the loader would."""
    window = as_of - pd.Timedelta(transfermarkt.MEMBERSHIP_WINDOW_DAYS, "D")
    recent = appearances[(appearances.date < as_of) & (appearances.date >= window)]
    recent = recent[recent.player_club_id.isin(transfermarkt._club_ids())]
    membership = recent.sort_values("date").groupby("player_id").player_club_id.last()
    priced = valuations[valuations.date < as_of]
    latest = priced.sort_values("date").groupby("player_id").market_value_in_eur.last()
    frame = pd.DataFrame({"club_id": membership}).join(latest, how="inner")
    frame = frame.dropna(subset=["market_value_in_eur"])
    frame["canonical"] = frame.club_id.map(transfermarkt.club_map())
    frame = frame.dropna(subset=["canonical"])
    totals = frame.groupby("canonical").market_value_in_eur.sum()
    return {club: float(value) for club, value in totals.items() if value > 0}


def test_value_difference_is_zero_when_either_club_is_unknown():
    values = {"A": 100.0, "B": 50.0}
    difference = transfermarkt.value_difference(values, ["A", "A"], ["B", "Sabah"])
    assert difference[0] == pytest.approx(0.6931471805599453)
    assert difference[1] == 0.0


def test_membership_counts_club_appearances_only():
    """A national team is not a club, so its matches say nothing about squads.

    ``appearances.csv`` carries national-team rows for the 2026 World Cup and
    the 2025-26 Africa Cup of Nations. The World Cup ran 11-28 June 2026, six
    days before the 2026-27 as-of, so without this filter 16.3% of players were
    assigned to their country and dropped out of their club's value.
    """
    appearances = transfermarkt._tables()[1]
    club_ids = transfermarkt._club_ids()
    as_of = pd.Timestamp(transfermarkt.as_of_for_season("2026-27"))
    window = as_of - pd.Timedelta(transfermarkt.MEMBERSHIP_WINDOW_DAYS, "D")
    recent = appearances[(appearances.date < as_of) & (appearances.date >= window)]
    # The filter is not vacuous: non-club ids do appear in the window.
    assert not recent.player_club_id.isin(club_ids).all()

    values = transfermarkt.squad_values(transfermarkt.as_of_for_season("2026-27"))
    # Under the bug these three read 406m, 251m and 193m.
    assert values["Paris Saint-Germain"] > 1_000e6
    assert values["Manchester City"] > 1_000e6
    assert values["Bayern Munich"] > 900e6
    assert values["Bayern Munich"] > values["Porto"]


def test_only_club_ids_from_the_club_table_reach_the_mapping():
    clubs = pd.read_csv(transfermarkt.CACHE_DIR / "clubs.csv")
    assert transfermarkt._club_ids() == frozenset(int(c) for c in clubs.club_id)
    assert set(transfermarkt.club_map()) <= transfermarkt._club_ids()
