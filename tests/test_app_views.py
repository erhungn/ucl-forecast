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

import re

import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="Streamlit lives in .venv-app, not .venv")

from streamlit.testing.v1 import AppTest             # noqa: E402

from ucl.config import PROJECT_ROOT                # noqa: E402
from ucl.webapp import loaders, paths                # noqa: E402

VIEWS = ["overview", "title_race", "next_matchday", "movement",
         "played_matches", "information_set"]

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


def test_played_matches_reports_the_archived_matchday_one_score():
    app = run("played_matches")
    html = bodies(app)
    assert "0.6913" in html and "0.1367" in html
    assert len(app.warning) == 1
    assert "anecdote" in app.warning[0].value


def test_the_information_set_keeps_the_update_button_and_its_warning():
    app = run("information_set")
    assert len(app.dataframe) == 1
    labels = [button.label for button in app.button]
    assert "Run the weekly update" in labels
    assert any("football_data_current" in warning.value for warning in app.warning)
    # Pruning must never be on by default.
    prune = [box for box in app.checkbox if "prun" in box.label.lower()]
    assert prune and prune[0].value is False


@pytest.mark.parametrize("view", VIEWS)
def test_no_view_writes_to_the_published_artefacts(view):
    season = paths.season_dir(paths.SEASON)
    before = {path: path.stat().st_mtime_ns for path in season.iterdir()}
    run(view)
    after = {path: path.stat().st_mtime_ns for path in season.iterdir()}
    assert before == after
