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
    would misrepresent the information set.

    Read across the domestic directories, not inside one: the in-season refresh
    fetches the current season into ``football_data_current`` and the prune
    deletes the superseded copies, so which one runs past the cutoff moves.
    """
    import datetime

    info = league_results.information(SEASON, as_of=datetime.date(2026, 9, 7))
    domestic = info[info.source.str.startswith("domestic/")]
    cached = [d for d in domestic.cached_to if d is not None]
    used = [u for u in domestic.used_to if u is not None]
    assert max(cached) > datetime.date(2026, 9, 7)
    assert max(used) < datetime.date(2026, 9, 7)


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


# --- the domestic-coverage gate --------------------------------------------
# On 2026-10-20 a weekly update published a 50,000-iteration forecast whose
# domestic results stopped on 2026-05-24: the refresh had unzipped into
# "football_data_current " and the previous day's prune had already removed the
# 2026-27 files from the main cache. Everything else about the run was normal.

def domestic_frame(rows) -> pd.DataFrame:
    """A minimal football-data frame: date, season and source_file are all the
    coverage check reads."""
    return pd.DataFrame([
        {"date": pd.Timestamp(day), "season": season, "source_file": f"{code}_2627.csv"}
        for code, season, day in rows
    ])


def every_main_league(day="2026-10-18", season="2026-27") -> pd.DataFrame:
    from ucl.data.football_data import MAIN_LEAGUES

    return domestic_frame([(code, season, day) for code in MAIN_LEAGUES])


def test_a_healthy_domestic_cache_publishes():
    frame = every_main_league()
    assert league_results.domestic_publication_gaps(
        frame, season_label=SEASON, as_of=pd.Timestamp("2026-10-19").date()) == []


def test_a_main_league_with_no_current_season_results_refuses():
    frame = every_main_league()
    frame = frame[frame.source_file != "E0_2627.csv"]
    reasons = league_results.domestic_publication_gaps(
        frame, season_label=SEASON, as_of=pd.Timestamp("2026-10-19").date())
    assert len(reasons) == 1
    assert "E0 (Premier League)" in reasons[0]
    assert "1 of 11 main leagues" in reasons[0]


def test_last_season_s_results_do_not_count_as_this_season_s():
    """The exact shape of the failure: files present, none of them current."""
    frame = every_main_league(day="2026-05-24", season="2025-26")
    reasons = league_results.domestic_publication_gaps(
        frame, season_label=SEASON, as_of=pd.Timestamp("2026-10-19").date())
    assert len(reasons) == 2
    assert "11 of 11 main leagues" in reasons[0]
    assert "148 days before the as-of date" in reasons[1]


def test_results_ending_just_inside_the_bar_still_publish():
    as_of = pd.Timestamp("2026-10-19").date()
    inside = every_main_league(day="2026-10-09")        # 10 days
    outside = every_main_league(day="2026-10-08")       # 11
    assert league_results.domestic_publication_gaps(
        inside, season_label=SEASON, as_of=as_of) == []
    reasons = league_results.domestic_publication_gaps(
        outside, season_label=SEASON, as_of=as_of)
    assert len(reasons) == 1 and "11 days" in reasons[0]


def test_an_empty_cache_refuses_rather_than_dividing_by_nothing():
    reasons = league_results.domestic_publication_gaps(
        domestic_frame([]).assign(date=pd.Series(dtype="datetime64[ns]")),
        season_label=SEASON, as_of=pd.Timestamp("2026-10-19").date())
    assert len(reasons) == 1 and "no results at all" in reasons[0]


def test_the_warning_fires_before_the_refusal():
    """A forecast is published with a note between the two thresholds, and
    refused past the second. The other order would mean refusing without ever
    having warned."""
    assert league_results.DOMESTIC_STALE_DAYS < league_results.DOMESTIC_PUBLISH_DAYS


def test_the_coverage_table_shows_the_hole_as_a_row():
    frame = every_main_league()
    frame = frame[frame.source_file != "SC0_2627.csv"]
    coverage = league_results.domestic_coverage(frame, SEASON)
    assert len(coverage) == 11                      # every league, hole included
    empty = coverage[coverage.matches == 0]
    assert list(empty.code) == ["SC0"]
    assert empty.iloc[0].last_result is None
    assert (coverage[coverage.code == "E0"].iloc[0].last_result
            == pd.Timestamp("2026-10-18").date())


def test_the_real_cache_that_produced_the_published_forecast_would_publish():
    """What is published came from a cache fit to publish from.

    Evaluated at the published as-of date, so it does not drift: domestic
    results only ever grow, and the bar is relative to that fixed date. It
    fails if the current-season cache goes missing again - which is what
    happened, and what nothing noticed.
    """
    from ucl.data import football_data
    from ucl.webapp import loaders, paths

    status = loaders.load_status(paths.SEASON)
    if not status or status.get("state") != "ok":
        pytest.skip("no successful update has run")
    as_of = pd.Timestamp(status["as_of"]).date()
    results = football_data.load_results(as_of)
    assert league_results.domestic_publication_gaps(
        results, season_label=paths.SEASON, as_of=as_of) == []
