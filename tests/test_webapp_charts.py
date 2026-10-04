"""Altair chart specifications.

Altair is installed only in the app environment (``.venv-app``), so these skip
when the suite runs in the model environment - which is the point of keeping the
two separate. Nothing here needs a browser: an Altair chart is a dict, and a
dict can be asserted on.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("altair", reason="Altair lives in .venv-app, not .venv")

from ucl.webapp import charts, clubs, palette  # noqa: E402


def _series() -> pd.DataFrame:
    rows = []
    for matchday in (0, 1, 2):
        for team, base in (("Arsenal", 0.2), ("Bayern Munich", 0.18),
                           ("Como", 0.01), ("Sabah", 0.001)):
            rows.append({"matchday": matchday, "team": team,
                         "p_champion": base + matchday * 0.01,
                         "p_top8": base * 3, "p_play_off": 0.3,
                         "p_eliminated": 0.1, "mean_position": 10 - matchday})
    return pd.DataFrame(rows)


def test_the_trend_chart_layers_selected_clubs_over_faint_context():
    spec = charts.trend(_series(), "p_champion", ["Arsenal", "Bayern Munich"]).to_dict()
    assert len(spec["layer"]) == 2
    highlight = spec["layer"][1]["encoding"]
    assert highlight["color"]["scale"]["range"][:2] == palette.CATEGORICAL[:2]
    assert highlight["color"]["scale"]["domain"] == ["ARS", "BAY"]
    assert spec["layer"][0]["mark"]["color"] == palette.BORDER


def test_the_trend_chart_caps_the_series_it_colours():
    many = sorted(clubs.CODES)[:12]
    series = pd.concat([_series()] * 1)
    spec = charts.trend(series, "p_champion", many).to_dict()
    assert len(spec["layer"][1]["encoding"]["color"]["scale"]["domain"]) \
        <= charts.MAX_SERIES


def test_mean_position_is_drawn_with_a_reversed_axis():
    """Position 1 is the best, so it belongs at the top."""
    spec = charts.trend(_series(), "mean_position", ["Arsenal"]).to_dict()
    assert spec["layer"][1]["encoding"]["y"]["scale"]["reverse"] is True


def test_small_multiples_facet_by_club_code():
    spec = charts.small_multiples(_series(), "p_top8", ["Arsenal", "Como"]).to_dict()
    assert spec["facet"]["field"] == "code"
    assert spec["columns"] == 2


def test_the_running_score_chart_dashes_the_uniform_baseline():
    frame = pd.DataFrame({
        "matchday": [1, 2, 1, 2], "series": ["uniform", "uniform",
                                             "frozen at cutoff", "frozen at cutoff"],
        "log_loss": [1.0986, 1.0986, 0.69, 0.70], "rps": [0.26, 0.26, 0.13, 0.14]})
    spec = charts.running_score(frame, "log_loss").to_dict()
    dashes = dict(zip(spec["encoding"]["strokeDash"]["scale"]["domain"],
                      spec["encoding"]["strokeDash"]["scale"]["range"]))
    assert dashes["uniform"] == [4, 3]
    assert dashes["frozen at cutoff"] == [1, 0]


