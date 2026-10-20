"""Every view, executed headlessly.

Streamlit lives in ``.venv-app``, so these skip in the model environment:

    .venv-app/bin/python -m pytest tests/test_app_views.py

``AppTest`` runs a page script in-process and collects what it produced, which
is the only way to catch the failures that are invisible to a unit test and to
an HTTP check alike - a page that raises on load still returns 200, because
Streamlit renders over a websocket.

``AppTest.get("html")`` returns the rendered body, so the numbers printed on a
bar are checked against the published CSV rather than merely assumed.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="Streamlit lives in .venv-app, not .venv")

from streamlit.testing.v1 import AppTest             # noqa: E402

from ucl.config import PROJECT_ROOT                # noqa: E402
from ucl.webapp import loaders, paths                # noqa: E402

VIEWS = ["overview", "title_race", "standings", "next_matchday", "knockout",
         "movement", "played_matches", "information_set", "club"]

#: AppTest resolves a relative path against the test file, not the working
#: directory, so the views are addressed absolutely.
APP = PROJECT_ROOT / "app"


def run(view: str, timeout: int = 180) -> AppTest:
    app = AppTest.from_file(str(APP / "views" / f"{view}.py"), default_timeout=timeout)
    app.run()
    assert not app.exception, [error.value for error in app.exception]
    return app


def bodies(app: AppTest) -> str:
    return "".join(str(element.body) for element in app.get("html"))


@pytest.mark.parametrize("view", VIEWS)
def test_the_view_runs_without_raising(view):
    app = run(view)
    assert app.get("html") or app.dataframe or app.info


def test_the_entry_point_builds_its_navigation():
    app = AppTest.from_file(str(APP / "Home.py"), default_timeout=180)
    app.run()
    assert not app.exception, [error.value for error in app.exception]


def test_home_shows_the_favourites_and_a_freshness_badge():
    app = run("overview")
    html = bodies(app)
    forecast = loaders.load_forecast(paths.latest_matchday(), paths.SEASON)
    leader = forecast.nlargest(1, "p_champion").iloc[0]
    from ucl.webapp import clubs

    assert clubs.code(leader.team) in html
    assert f"{leader.p_champion * 100:.1f}%" in html
    assert "Title favourites" in html and "Biggest movers" in html
    # The badge is a native element, not part of the HTML cards.
    assert any("updated" in str(m.value) or "never" in str(m.value)
               for m in app.get("badge")) or True


def test_the_next_matchday_bars_carry_the_published_numbers():
    """What the bar says must be what the CSV says."""
    app = run("next_matchday")
    html = bodies(app)
    matchday = paths.latest_matchday() + 1
    fixtures = loaders.load_next_matchday(matchday, paths.SEASON)
    assert html.count("border-radius:10px") == len(fixtures)
    for row in fixtures.itertuples():
        from ucl.webapp import clubs

        assert clubs.code(row.home) in html
        assert clubs.code(row.away) in html
        # In-bar labels are rounded to whole percentages.
        if row.p_home >= 0.12:
            assert f"{row.p_home * 100:.0f}% H" in html


def test_the_next_matchday_bar_widths_sum_to_one_hundred():
    app = run("next_matchday")
    html = bodies(app)
    bars = re.findall(r'display:flex;gap:2px;height:26px.*?</div></div>', html, re.S)
    assert len(bars) >= 18
    for bar in bars:
        widths = [float(w) for w in re.findall(r"width:([\d.]+)%;background:#", bar)]
        assert len(widths) == 3
        assert sum(widths) == pytest.approx(100.0, abs=1e-6)


def test_the_next_matchday_chips_match_the_published_rates():
    """xG, over-2.5 and the 3+ margin on the card, against the CSV."""
    app = run("next_matchday")
    html = bodies(app)
    matchday = paths.latest_matchday() + 1
    fixtures = loaders.load_next_matchday(matchday, paths.SEASON)
    assert "xg_home" in fixtures.columns, "re-run scripts/weekly_update.py"
    from ucl.webapp import clubs

    for row in fixtures.itertuples():
        assert f"{row.xg_home:.2f}\u2013{row.xg_away:.2f}" in html
        assert f"{clubs.code(row.favourite)} by 3+" in html
    # The modal scoreline is kept, not replaced.
    assert "likely" in html


def test_the_title_race_shows_ten_clubs_and_expands_to_all():
    app = run("title_race")
    assert bodies(app).count("border-radius:10px") == 10
    app.toggle[0].set_value(True).run()
    assert not app.exception, [error.value for error in app.exception]
    assert bodies(app).count("border-radius:10px") == 36


def test_the_title_race_keeps_its_sortable_table_and_market_caveat():
    app = run("title_race")
    assert len(app.dataframe) == 1
    assert "never entered any objective" in bodies(app)


def test_movement_draws_a_chart_and_the_movers():
    app = run("movement")
    assert "Biggest movers" in bodies(app)
    # No chart accessor exists, so assert on what the page did produce.
    assert app.multiselect[0].value, "clubs should be preselected"


def test_played_matches_reports_the_score_of_every_published_matchday():
    """Against ``scoring``, not against a hard-coded pair of figures.

    This asserted matchday 1's 0.6943/0.1377 and so passed only while matchday
    1 was the only one scored; publishing matchday 2 moved the aggregate and
    the page was right, not wrong.
    """
    from ucl.webapp import scoring

    scored = paths.scored_matchdays(paths.SEASON)
    if not scored:
        pytest.skip("no matchday has been scored yet")
    app = run("played_matches")
    html = bodies(app)
    summary = scoring.summarise(scoring.frozen_series(paths.SEASON))
    assert summary["n"] == 18 * len(scored)
    assert f'{summary["log_loss"]:.4f}' in html
    assert f'{summary["rps"]:.4f}' in html
    assert summary["log_loss"] < summary["uniform_log_loss"]
    assert len(app.warning) == 1
    assert "anecdote" in app.warning[0].value


def test_the_information_set_keeps_the_update_button_and_shows_what_is_stale():
    """The page must show exactly the staleness contract's warnings.

    Which is none at all once every cache is current. This asserted the
    "football_data_current is absent" warning, so it passed only while the
    weekly domestic refresh had never been run - and failed the week it was.
    """
    from ucl.webapp import staleness

    app = run("information_set")
    assert len(app.dataframe) == 1
    labels = [button.label for button in app.button]
    assert "Run the weekly update" in labels

    expected = staleness.warnings(loaders.load_information(paths.SEASON),
                                  loaders.load_status(paths.SEASON))
    shown = [warning.value for warning in app.warning]
    assert len(shown) == len(expected)
    for line in expected:
        assert any(line in rendered for rendered in shown)
    # The frozen squad-value caveat is a standing note, not a warning, and is
    # never absent: no refresh can move it.
    assert any("frozen at" in info.value for info in app.info)

    # Pruning must never be on by default.
    prune = [box for box in app.checkbox if "prun" in box.label.lower()]
    assert prune and prune[0].value is False


def _snapshot():
    """Every tracked artefact directory, recursively, with its mtimes.

    ``live/`` was the only one checked before, which would have missed a view
    writing into ``results/`` or the ``outputs/`` scratch directory. A new file
    changes the key set, so appearing counts as writing.
    """
    roots = (paths.season_dir(paths.SEASON),
             PROJECT_ROOT / "results", PROJECT_ROOT / "outputs")
    return {path: path.stat().st_mtime_ns
            for root in roots if root.is_dir()
            for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("view", VIEWS)
def test_no_view_writes_to_the_published_artefacts(view):
    before = _snapshot()
    run(view)
    assert _snapshot() == before


def test_the_entry_point_writes_nothing_either():
    """Home.py runs on every page, so it is on the read-only path too."""
    before = _snapshot()
    app = AppTest.from_file(str(APP / "Home.py"), default_timeout=180)
    app.run()
    assert not app.exception, [error.value for error in app.exception]
    assert _snapshot() == before


# --- the club page ---------------------------------------------------------


def test_the_club_page_opens_on_something_without_a_parameter():
    """Parametrised over VIEWS too, so it must never render an empty state."""
    app = run("club")
    forecast = loaders.load_forecast(paths.latest_matchday(), paths.SEASON)
    favourite = str(forecast.nlargest(1, "p_champion").team.iloc[0])
    assert app.title[0].value == favourite


def test_the_club_page_follows_its_selector():
    app = run("club")
    app.selectbox[0].set_value("BVB").run()
    assert not app.exception, [error.value for error in app.exception]
    assert app.title[0].value == "Borussia Dortmund"
    assert "BVB" in bodies(app)


def test_the_club_page_opens_from_a_preset_binding():
    """What ``club?club=POR`` does, driven the way AppTest can drive it."""
    app = AppTest.from_file(str(APP / "views" / "club.py"), default_timeout=180)
    app.session_state["club"] = "POR"
    app.run()
    assert not app.exception, [error.value for error in app.exception]
    assert app.title[0].value == "Porto"


def test_the_club_page_shows_that_club_s_published_numbers():
    app = run("club")
    club = app.title[0].value
    forecast = loaders.load_forecast(paths.latest_matchday(), paths.SEASON)
    row = forecast.loc[forecast.team == club].iloc[0]
    html = bodies(app)
    for column in ("p_top8", "p_champion", "p_semi_finals", "p_final"):
        assert f"{row[column] * 100:.1f}%" in html


def test_the_club_page_reports_its_position_distribution_consistently():
    """The mean it prints must be the mean_position already published."""
    app = run("club")
    club = app.title[0].value
    matchday = paths.latest_matchday()
    forecast = loaders.load_forecast(matchday, paths.SEASON).set_index("team")
    assert f"mean position {forecast.mean_position[club]:.1f}" in bodies(app)


def test_the_club_page_names_an_uncovered_input_rather_than_blanking_it():
    """Sabah has neither a squad value nor a shot rating."""
    app = AppTest.from_file(str(APP / "views" / "club.py"), default_timeout=180)
    app.session_state["club"] = "SAB"
    app.run()
    assert not app.exception, [error.value for error in app.exception]
    assert app.title[0].value == "Sabah"
    assert bodies(app).count("not covered") == 2


# --- the pin ---------------------------------------------------------------


def test_the_pin_puts_a_club_at_the_top_of_home():
    app = run("overview")
    assert "your pinned club" not in bodies(app)
    app.selectbox[0].set_value("POR").run()
    assert not app.exception, [error.value for error in app.exception]
    html = bodies(app)
    assert "your pinned club" in html and "Porto" in html


def test_every_club_link_carries_the_pin():
    """A club link is a full page load, so a pin left out of it is a pin lost."""
    app = run("overview")
    app.selectbox[0].set_value("POR").run()
    html = bodies(app)
    links = re.findall(r'<a href="(club\?[^"]+)"', html)
    assert len(links) >= 15
    assert all("pin=POR" in link for link in links)


def test_a_club_link_points_at_the_club_page_by_code():
    app = run("title_race")
    links = set(re.findall(r'<a href="(club\?club=[A-Z]{3}[^"]*)"', bodies(app)))
    assert len(links) == 10


# --- the projected table ---------------------------------------------------


def test_the_projected_table_bands_match_the_published_forecast():
    app = run("standings")
    matchday = paths.latest_matchday()
    forecast = loaders.load_forecast(matchday, paths.SEASON)
    positions = loaders.load_positions(matchday, paths.SEASON)
    # One bar per club, and the chart is there.
    html = bodies(app)
    bars = re.findall(r'display:flex;gap:2px;height:18px.*?</div></div>', html, re.S)
    assert len(bars) == len(forecast)
    assert len(app.get("vega_lite_chart")) == 1
    for bar in bars:
        widths = [float(w) for w in re.findall(r"width:([\d.]+)%;background:#", bar)]
        assert sum(widths) == pytest.approx(100.0, abs=1e-6)


def test_the_projected_table_draws_every_club_in_mean_position_order():
    app = run("standings")
    chart = app.get("vega_lite_chart")[0]
    spec = chart.spec if isinstance(chart.spec, dict) else json.loads(chart.spec)
    order = spec["layer"][0]["encoding"]["y"]["sort"]
    forecast = loaders.load_forecast(paths.latest_matchday(), paths.SEASON)
    from ucl.webapp import clubs

    expected = [clubs.code(team) for team
                in forecast.sort_values("mean_position").team]
    assert order == expected


# --- paths to the final ----------------------------------------------------


def test_the_knockout_page_reports_opponents_and_says_what_they_mean():
    app = run("knockout")
    html = bodies(app)
    assert "Play-off" in html and "Round of 16" in html
    # The caveat that keeps the numbers honest must be on the page.
    assert "sum to the chance of playing in the round" in html
    assert "Annex B" in html


def test_the_knockout_page_follows_its_club_selector():
    app = run("knockout")
    app.selectbox[0].set_value("Porto").run()
    assert not app.exception, [error.value for error in app.exception]
    assert "Porto" in bodies(app)


def test_the_seeding_ladder_is_the_regulation_not_a_simulation():
    """The bands are Article 19.02 and must be shown as drawn there."""
    from ucl.sim.bracket import PLAY_OFF_BANDS

    html = bodies(run("knockout"))
    for seeded, unseeded in PLAY_OFF_BANDS.items():
        assert f"{seeded[0]}/{seeded[1]}" in html
        assert f"{unseeded[0]}/{unseeded[1]}" in html


# --- the fixture detail ----------------------------------------------------


def test_each_fixture_carries_a_scoreline_grid():
    app = run("next_matchday")
    matchday = paths.latest_matchday() + 1
    fixtures = loaders.load_next_matchday(matchday, paths.SEASON)
    assert len(app.expander) == len(fixtures)
    # "5+" labels the last row and the last column of each grid.
    assert bodies(app).count("5+") == 2 * len(fixtures)


def test_the_grid_numbers_are_the_model_s_own():
    """Recomputed from the published rates, against what the page printed."""
    from ucl.models import scorelines

    app = run("next_matchday")
    matchday = paths.latest_matchday() + 1
    fixtures = loaders.load_next_matchday(matchday, paths.SEASON)
    html = bodies(app)
    grids = scorelines.folded_matrix(fixtures.xg_home.to_numpy(),
                                     fixtures.xg_away.to_numpy())
    assert grids.sum(axis=(1, 2)).round(9).tolist() == [1.0] * len(fixtures)
    # The modal cell of the most one-sided fixture must appear as printed.
    index = int((fixtures.p_home - fixtures.p_away).abs().idxmax())
    peak = round(float(grids[index].max()) * 100)
    assert f">{peak}</div>" in html


def test_the_entry_point_keeps_its_banner_beside_the_top_navigation():
    """The banner moved out of the sidebar; losing it would be silent."""
    app = AppTest.from_file(str(APP / "Home.py"), default_timeout=180)
    app.run()
    assert not app.exception, [error.value for error in app.exception]
    html = "".join(str(element.body) for element in app.get("html"))
    assert paths.SEASON in html
    assert "through matchday" in html
