"""The presentation layer: colours, badges, bars and chart specs.

All pure functions, so none of this needs Streamlit installed. The point of
testing markup at all is that the parts which can go quietly wrong - an
unreadable badge, two clubs that look identical, a bar whose segments do not
add up - are exactly the parts nobody notices by eye.
"""

from __future__ import annotations

import itertools
import re
import tomllib
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ucl.config import PROJECT_ROOT
from ucl.webapp import cards, clubs, freshness, palette

CONFIG = PROJECT_ROOT / ".streamlit" / "config.toml"


# --- colour arithmetic ----------------------------------------------------

def test_contrast_ratio_against_known_values():
    assert palette.contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert palette.contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0)
    # Mid grey on white, a published WCAG example.
    assert palette.contrast_ratio("#767676", "#ffffff") == pytest.approx(4.54, abs=0.02)


def test_enforce_contrast_moves_only_as_far_as_it_must():
    already = palette.enforce_contrast("#000000", "#ffffff")
    assert already == "#000000"
    fixed = palette.enforce_contrast("#ed1c24", "#ffffff")
    assert palette.contrast_ratio(fixed, "#ffffff") >= palette.MIN_CONTRAST
    # Recognisably the same red, not a different colour.
    assert palette.colour_distance("#ed1c24", fixed) < 6


def test_colour_distance_separates_hues_and_not_shades():
    assert palette.colour_distance("#ffe500", "#ffe500") == 0
    assert palette.colour_distance("#2dd4bf", "#fb923c") > 50


# --- club badges ----------------------------------------------------------

def test_every_club_has_a_distinct_three_letter_code():
    teams = set(pd.read_csv(PROJECT_ROOT / "data" / "raw" / "teams_2026_27.csv",
                            keep_default_na=False).team)
    assert set(clubs.CODES) == teams
    assert len(set(clubs.CODES.values())) == 36
    assert all(len(c) == 3 and c.isalpha() and c.isupper()
               for c in clubs.CODES.values())
    assert clubs.CODES["Borussia Dortmund"] == "BVB"


def test_every_club_declares_colours():
    assert set(clubs.COLOURS) == set(clubs.CODES)


@pytest.mark.parametrize("team", sorted(clubs.CODES))
def test_every_badge_is_legible(team):
    """WCAG AA for normal text: the badges set 12-13px, so 4.5:1 applies."""
    ground, text = clubs.badge(team)
    assert palette.contrast_ratio(ground, text) >= palette.MIN_CONTRAST


@pytest.mark.parametrize("team", sorted(clubs.CODES))
def test_every_outline_badge_is_legible_on_a_card(team):
    """A navy club would otherwise vanish when drawn as an outline."""
    assert palette.contrast_ratio(clubs.outline(team), palette.SURFACE) \
        >= palette.MIN_CONTRAST


def test_two_badges_in_one_card_are_always_distinguishable():
    """Either the grounds differ, or the away badge switches to an outline."""
    for home, away in itertools.permutations(sorted(clubs.CODES), 2):
        home_style, away_style = clubs.badge_pair_styles(home, away)
        distance = palette.colour_distance(clubs.badge(home)[0],
                                           clubs.badge(away)[0])
        assert home_style == "solid"
        assert away_style == "outline" or distance >= clubs.MIN_DISTANCE


def test_the_same_coloured_pairs_really_do_trigger_the_outline():
    # Bodo/Glimt and Lens both play in the same yellow.
    assert clubs.badge_pair_styles("Bodo/Glimt", "Lens") == ("solid", "outline")
    assert clubs.badge_pair_styles("Borussia Dortmund", "Bodo/Glimt")[1] == "outline"
    # Arsenal red and Manchester City sky blue need no help.
    assert clubs.badge_pair_styles("Arsenal", "Manchester City") == ("solid", "solid")


def test_an_unknown_club_still_renders():
    assert clubs.code("Some New Club") == "SOM"
    ground, text = clubs.badge("Some New Club")
    assert palette.contrast_ratio(ground, text) >= palette.MIN_CONTRAST
    assert "SOM" in cards.badge_html("Some New Club")


# --- bars and cards -------------------------------------------------------

def _widths(html: str) -> list[float]:
    return [float(w) for w in re.findall(r"width:([\d.]+)%;background:#", html)]


def test_bar_segments_sum_to_the_full_width():
    html = cards.stacked_bar_html(0.494685, 0.247367, 0.257947)
    widths = _widths(html)
    assert len(widths) == 3
    assert sum(widths) == pytest.approx(100.0, abs=1e-6)


def test_bar_segments_are_normalised_when_the_inputs_are_not():
    widths = _widths(cards.stacked_bar_html(1.0, 1.0, 2.0))
    assert widths == pytest.approx([25.0, 25.0, 50.0])


def test_a_wide_segment_is_labelled_inside_the_bar():
    html = cards.stacked_bar_html(0.494685, 0.247367, 0.257947)
    assert "49% H" in html and "25% D" in html and "26% A" in html
    assert cards.narrow_outcomes(0.494685, 0.247367, 0.257947) == ""


def test_a_narrow_segment_moves_its_label_below_the_bar():
    html = cards.stacked_bar_html(0.870, 0.091, 0.038)
    assert "87% H" in html
    assert "4% A" not in html                   # too thin to hold a label
    below = cards.narrow_outcomes(0.870, 0.091, 0.038)
    assert below == "draw 9.1% · away 3.8%"


def test_an_all_zero_bar_does_not_divide_by_zero():
    assert _widths(cards.stacked_bar_html(0.0, 0.0, 0.0)) == [0.0, 0.0, 0.0]


def test_delta_markers_carry_an_arrow_as_well_as_a_colour():
    assert "▲" in cards.delta_html(0.1795) and palette.UP in cards.delta_html(0.1795)
    assert "▼" in cards.delta_html(-0.1643) and palette.DOWN in cards.delta_html(-0.1643)
    assert "–" in cards.delta_html(0.0)
    assert cards.delta_html(None) == ""
    assert cards.delta_html(float("nan")) == ""


def test_a_club_name_cannot_inject_markup():
    html = cards.compact_rows_html(
        [{"team": "<script>alert(1)</script>", "value": 0.5}])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_fixture_cards_render_one_card_per_row():
    rows = [
        {"home": "Manchester City", "away": "Paris Saint-Germain", "date": "14 Oct",
         "p_home": 0.494685, "p_draw": 0.247367, "p_away": 0.257947,
         "pick": "home", "modal_score": "1-1", "p_modal": 0.117509},
        {"home": "Bodo/Glimt", "away": "Lens", "date": "13 Oct",
         "p_home": 0.4, "p_draw": 0.26, "p_away": 0.34},
    ]
    html = cards.fixture_cards_html(rows)
    assert html.count("border-radius:10px") == 2
    assert "MCI" in html and "PSG" in html and "BOD" in html and "LEN" in html
    assert "likely 1-1 (11.8%)" in html
    # The same-colour fixture outlines its away badge.
    assert "background:transparent" in html


def test_fixture_cards_can_show_the_actual_result():
    html = cards.fixture_cards_html([
        {"home": "AEK Athens", "away": "LASK", "p_home": 0.4925, "p_draw": 0.2479,
         "p_away": 0.2596, "score": "1-0", "result": "home", "p_actual": 0.4925}],
        show_result=True)
    assert "1-0 AEK Athens" in html
    assert "forecast 49.2%" in html


def test_empty_lists_say_so_rather_than_rendering_nothing():
    assert "No fixtures" in cards.fixture_cards_html([])
    assert "No clubs" in cards.club_cards_html([])
    assert "Nothing published" in cards.compact_rows_html([])
    assert "No movement" in cards.mover_rows_html([])


# --- freshness ------------------------------------------------------------

def _status(hours: float, **extra) -> dict:
    finished = datetime.now(timezone.utc) - timedelta(hours=hours)
    return {"state": "ok", "finished_at_utc": finished.isoformat(), **extra}


def test_freshness_levels_at_their_boundaries():
    assert freshness.level(_status(1))[0] == "fresh"
    assert freshness.level(_status(47))[0] == "fresh"
    assert freshness.level(_status(49))[0] == "ageing"
    assert freshness.level(_status(24 * 9))[0] == "stale"


def test_freshness_reports_a_running_or_refused_run():
    assert freshness.level({"state": "running"})[0] == "ageing"
    assert freshness.level({"state": "refused", "message": "mid-round"})[0] == "stale"
    assert freshness.level({})[0] == "stale"


def test_a_recent_run_over_a_stale_cache_is_not_fresh():
    """A forecast computed an hour ago from a fortnight-old cache is not current."""
    information = pd.DataFrame([
        {"source": "domestic/football_data", "status": "present", "files": 1,
         "cached_to": "2026-09-21", "used_to": "2026-09-21", "fetched": None}])
    level, _, _ = freshness.level(_status(1, as_of="2026-10-12"), information)
    assert level == "ageing"


def test_the_age_label_reads_naturally():
    assert freshness.level(_status(2))[1] == "updated today"
    assert freshness.level(_status(30))[1] == "updated yesterday"
    assert freshness.level(_status(24 * 4))[1] == "updated 4 days ago"


# --- the theme matches the palette ---------------------------------------

def test_the_streamlit_theme_and_the_palette_agree():
    """Two definitions of one colour is one too many; this pins them together."""
    with open(CONFIG, "rb") as handle:
        theme = tomllib.load(handle)["theme"]
    assert theme["base"] == "dark"
    assert theme["backgroundColor"].lower() == palette.INK
    assert theme["secondaryBackgroundColor"].lower() == palette.SURFACE
    assert theme["textColor"].lower() == palette.TEXT
    assert theme["primaryColor"].lower() == palette.ACCENT
    assert theme["borderColor"].lower() == palette.BORDER
    assert [c.lower() for c in theme["chartCategoricalColors"]] == palette.CATEGORICAL
