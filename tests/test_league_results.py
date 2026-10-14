"""Played league-phase matches must line up with the fixture list exactly."""

from __future__ import annotations

import pandas as pd
import pytest

from ucl.config import SEASONS
from ucl.data import fixtures as fx
from ucl.data import league_results, wikipedia
from ucl.models.elo import elo_as_of

SEASON = "2026-27"


@pytest.fixture(scope="module")
def played():
    return league_results.load_played(SEASON)


def test_played_matches_map_to_fixtures(played):
    fixtures = pd.read_csv(fx.fixtures_path(SEASON))
    assert len(played) > 0
    assert set(played.matchday) <= set(fixtures.matchday)
    pairs = list(zip(played.home_id, played.away_id))
    assert len(pairs) == len(set(pairs)), "a fixture appears twice in the results"


def test_every_played_match_has_a_score(played):
    assert played.hg.notna().all() and played.ag.notna().all()
    assert (played.hg >= 0).all() and (played.ag >= 0).all()
    assert set(played.status) <= {"played", "awarded"}


def test_completed_matchdays_counts_only_full_rounds(played):
    fixtures = pd.read_csv(fx.fixtures_path(SEASON))
    complete = league_results.completed_matchdays(played, fixtures)
    for matchday in complete:
        assert int((played.matchday == matchday).sum()) == \
               int((fixtures.matchday == matchday).sum())


def test_a_partial_matchday_is_not_reported_as_complete(played):
    fixtures = pd.read_csv(fx.fixtures_path(SEASON))
    trimmed = played.iloc[:-1]
    assert played.matchday.max() not in league_results.completed_matchdays(trimmed, fixtures)


def test_an_unmapped_result_raises(tmp_path, monkeypatch):
    """A match against a club that is in no fixture must not be absorbed."""
    page = league_results.live_page(SEASON)
    original = league_results.wikipedia.parse_matches

    def with_a_stranger(*args, **kwargs):
        frame = original(*args, **kwargs)
        extra = frame[frame.played].iloc[:1].copy()
        extra["home"] = "Some Club That Is Not Playing"
        return pd.concat([frame, extra], ignore_index=True)

    monkeypatch.setattr(league_results.wikipedia, "parse_matches", with_a_stranger)
    with pytest.raises(league_results.MappingError, match="map to no fixture"):
        league_results.load_played(SEASON)


def test_a_duplicated_fixture_raises(monkeypatch):
    original = league_results.wikipedia.parse_matches

    def with_a_replay(*args, **kwargs):
        frame = original(*args, **kwargs)
        return pd.concat([frame, frame[frame.played].iloc[:1]], ignore_index=True)

    monkeypatch.setattr(league_results.wikipedia, "parse_matches", with_a_replay)
    with pytest.raises(league_results.MappingError, match="more than once"):
        league_results.load_played(SEASON)


def test_the_live_page_does_not_leak_into_the_pre_tournament_ratings():
    """Matchday 1 is after the cutoff, so it must not reach the frozen forecast."""
    cfg = SEASONS[SEASON]
    ratings = elo_as_of(cfg.cutoff_date)
    assert ratings.pre_match.date.max().date() < cfg.cutoff_date


def test_the_live_page_does_reach_a_later_as_of_date(played):
    """And after matchday 1 it must, or the update mode learns nothing."""
    import datetime

    later = elo_as_of(datetime.date(2026, 10, 12))
    assert later.pre_match.date.max().date() >= played.date.min().date()
    assert len(later.pre_match) > len(elo_as_of(SEASONS[SEASON].cutoff_date).pre_match)


def test_the_information_report_names_every_source():
    info = league_results.information(SEASON)
    assert {"domestic/football_data", "european/openfootball",
            "european/wikipedia"} <= set(info.source)
    present = info[info.status == "present"]
    assert present.cached_to.notna().all()


def test_the_information_report_separates_what_is_cached_from_what_is_used():
    """The domestic cache runs past the pre-tournament cutoff; saying otherwise
    would misrepresent the information set."""
    import datetime

    info = league_results.information(SEASON, as_of=datetime.date(2026, 9, 7))
    domestic = info[info.source == "domestic/football_data"].iloc[0]
    assert domestic.cached_to > datetime.date(2026, 9, 7)
    assert domestic.used_to < datetime.date(2026, 9, 7)


def test_a_refreshed_page_must_parse_every_past_fixture(tmp_path, monkeypatch):
    """The matchday-detection path is where a dropped result does most harm.

    A finished round with one unreadable score looks partial, and the weekly
    update then refuses for a reason unconnected to the real cause.
    """
    from datetime import date

    page = league_results.live_page(SEASON)
    source = wikipedia.CACHE_DIR / page.file_name
    text = source.read_text(encoding="utf-8")
    # Make matchday 1 unreadable the way matchday 2 of 2026-27 was.
    broken = text.replace("|score      = 2–1", "|score      = 2¬1", 1)
    assert broken != text, "expected at least one en-dash score to mangle"
    (tmp_path / page.file_name).write_text(broken, encoding="utf-8")
    manifest = wikipedia.CACHE_DIR / "manifest.json"
    (tmp_path / "manifest.json").write_text(manifest.read_text(encoding="utf-8"),
                                            encoding="utf-8")

    with pytest.raises(ValueError, match="parsed no score"):
        league_results.load_played(SEASON, cache_dir=tmp_path, refresh=False,
                                   today=date(2026, 10, 14))
