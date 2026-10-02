"""The Wikipedia parser fills gaps openfootball lacks, so its edge cases matter.

Each case below is a real markup variant that silently dropped matches before it
was handled: an unlinked second-leg team name, {{flagicon}} instead of
{{fbaicon}}, and a nested {{plainlist}} that ends a line with "}}".
"""

import pandas as pd
import pytest

from ucl.data import wikipedia

BOX = """
==Knockout phase play-offs==
===Matches===
{{Football box
|date       = {{Start date|2025|2|20|df=y}}
|team1      = [[FC Bodo/Glimt|Bodo/Glimt]] {{fbaicon|NOR}}
|score      = 2–1
|aet        = yes
|team2      = {{fbaicon|NED}} [[FC Twente|Twente]]
|goals1     =
*[[A Player|Player]] {{goal|12}}
|goals2     =
*[[B Player|Other]] {{goal|75}}
|stadium    = {{plainlist|
* [[Aspmyra Stadion]], [[Bodo]]
}}
|penaltyscore = 4–3
}}
----
{{Football box
|date       = {{Start date|2025|2|27|df=y}}
|team1      = Twente {{flagicon|NED}}
|score      = 0–0
|team2      = {{fbaicon|NOR}} Bodo/Glimt
}}
"""


@pytest.fixture
def parsed():
    return wikipedia.parse_matches(BOX, season="2024-25", competition="UEL")


def test_both_boxes_parse(parsed):
    assert len(parsed) == 2


def test_extra_time_score_is_separated_from_ninety_minutes(parsed):
    row = parsed.iloc[0]
    # Wikipedia's score includes extra time; 90 minutes comes from goal minutes.
    assert (row.et_hg, row.et_ag) == (2, 1)
    assert (row.hg, row.ag) == (1, 1)
    assert row.aet


def test_penalty_shootout_is_read(parsed):
    assert (parsed.iloc[0].pens_h, parsed.iloc[0].pens_a) == (4, 3)


def test_nested_plainlist_does_not_truncate_the_box(parsed):
    # penaltyscore sits after the nested {{plainlist}}, so it is only found if
    # the block is extracted with balanced braces.
    assert pd.notna(parsed.iloc[0].pens_h)


def test_unlinked_second_leg_team_and_flagicon(parsed):
    row = parsed.iloc[1]
    assert (row.home, row.home_country) == ("Twente", "NED")
    assert (row.away, row.away_country) == ("Bodo/Glimt", "NOR")


def test_stage_comes_from_the_heading(parsed):
    assert parsed.iloc[0].stage == "Knockout phase play-offs"


def test_plain_score_is_the_ninety_minute_score(parsed):
    row = parsed.iloc[1]
    assert (row.hg, row.ag) == (0, 0)
    assert not row.aet and pd.isna(row.et_hg)


def test_validation_pages_are_excluded_from_the_data_by_default():
    from ucl.config import season

    cutoff = season("2026-27").cutoff_date
    gaps = wikipedia.load_results(cutoff)
    # 2024-25 exists only to validate the parser; loading it would double-count
    # what openfootball already provides.
    assert "2024-25" not in set(gaps.season)
    assert {"2025-26", "2026-27"} <= set(gaps.season)
    # The pre-2021 Europa League seasons are gap data too: openfootball has no
    # el.txt before 2020-21.
    assert {"2016-17", "2017-18", "2018-19", "2019-20"} <= set(gaps.season)
    assert set(gaps[gaps.season < "2020-21"].competition) == {"UEL"}
    assert {p.role for p in wikipedia.PAGES} == {"gap", "validation"}


def test_gap_data_respects_the_cutoff():
    from ucl.config import season

    cfg = season("2026-27")
    gaps = wikipedia.load_results(cfg.cutoff_date)
    assert gaps.date.max().date() < cfg.cutoff_date


def test_european_sources_do_not_overlap():
    from ucl.config import season
    from ucl.data.european import coverage, load_results

    frame = load_results(season("2026-27").cutoff_date)
    per_phase = coverage(frame).groupby(["season", "competition", "phase"]).source.nunique()
    assert per_phase.max() == 1
    assert len(frame) > 4500
