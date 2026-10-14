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


# --- dash variants in the score field --------------------------------------
# Matchday 2 of 2026-27 was written with MINUS SIGN where matchday 1 used an en
# dash. The pattern accepted the en dash and the hyphen only, so nine played
# matches were read as unplayed and nothing complained.

DASH_VARIANTS = [
    ("hyphen-minus", "-"),
    ("hyphen", "‐"),
    ("non-breaking hyphen", "‑"),
    ("figure dash", "‒"),
    ("en dash", "–"),
    ("em dash", "—"),
    ("horizontal bar", "―"),
    ("minus sign", "−"),
    ("small hyphen-minus", "﹣"),
    ("fullwidth hyphen-minus", "－"),
    ("two-em dash", "⸺"),
    ("wave dash", "〜"),
]


@pytest.mark.parametrize("name,dash", DASH_VARIANTS, ids=[n for n, _ in DASH_VARIANTS])
def test_a_score_parses_whatever_dash_it_is_written_with(name, dash):
    box = BOX.replace("2–1", f"3{dash}1").replace("|aet        = yes\n", "")
    frame = wikipedia.parse_matches(box, season="2024-25", competition="UEL")
    first = frame.iloc[0]
    assert (first.hg, first.ag) == (3, 1), f"{name} (U+{ord(dash):04X}) not read"
    assert bool(first.played)


@pytest.mark.parametrize("name,dash", DASH_VARIANTS, ids=[n for n, _ in DASH_VARIANTS])
def test_a_penalty_score_parses_whatever_dash_it_is_written_with(name, dash):
    box = BOX.replace("4–3", f"5{dash}4")
    frame = wikipedia.parse_matches(box, season="2024-25", competition="UEL")
    first = frame.iloc[0]
    assert (first.pens_h, first.pens_a) == (5, 4), f"{name} not read"


def test_spaces_around_the_dash_are_tolerated():
    for spelling in ("3 - 1", "3− 1", "3 –1", "3—1"):
        frame = wikipedia.parse_matches(
            BOX.replace("2–1", spelling).replace("|aet        = yes\n", ""),
            season="2024-25", competition="UEL")
        assert (frame.iloc[0].hg, frame.iloc[0].ag) == (3, 1), spelling


def test_the_dash_table_covers_every_unicode_dash():
    """The table is written out for import speed; this is what keeps it honest.

    Scanning Unicode for category Pd costs ~80ms, too much for a module the
    model pipeline imports, so the characters are listed literally. If a future
    Unicode release adds a dash, this fails instead of a season's results going
    quietly missing.
    """
    import sys
    import unicodedata

    dashes = {chr(point) for point in range(sys.maxunicode + 1)
              if unicodedata.category(chr(point)) == "Pd"}
    missing = sorted(dashes - set(wikipedia._DASHES))
    assert not missing, (
        "category Pd characters absent from _DASHES: "
        + ", ".join(f"U+{ord(c):04X} {unicodedata.name(c, '?')}" for c in missing))
    # MINUS SIGN is Sm, not Pd: the exact gap that caused the failure, so it is
    # asserted separately rather than left to the category scan.
    assert "−" in wikipedia._DASHES
    assert unicodedata.category("−") == "Sm"


def test_normalising_dashes_leaves_everything_else_alone():
    assert wikipedia.normalise_dashes("Bodo/Glimt 2–1") == "Bodo/Glimt 2-1"
    assert wikipedia.normalise_dashes("Atlético") == "Atlético"
    assert wikipedia.normalise_dashes("") == ""


# --- the guard: a past match with no score is a failure, not a fixture -----


def test_a_past_match_with_no_score_raises_instead_of_reading_as_unplayed():
    from datetime import date

    box = BOX.replace("|score      = 2–1", "|score      = 2?1")
    # Without the guard this is the silent failure: played=False, no complaint.
    quiet = wikipedia.parse_matches(box, season="2024-25", competition="UEL")
    assert not bool(quiet.iloc[0].played)

    with pytest.raises(ValueError, match="parsed no score") as raised:
        wikipedia.parse_matches(box, season="2024-25", competition="UEL",
                                source_file="page.wikitext",
                                require_scores_before=date(2025, 3, 1))
    message = str(raised.value)
    assert "2025-02-20" in message
    assert "Bodo/Glimt" in message and "Twente" in message
    assert "'2?1'" in message, "the raw score text must be in the message"
    assert "page.wikitext" in message


def test_the_guard_ignores_a_fixture_that_has_not_been_played_yet():
    from datetime import date

    box = BOX.replace("|score      = 2–1", "|score      =")
    frame = wikipedia.parse_matches(box, season="2024-25", competition="UEL",
                                    require_scores_before=date(2025, 2, 1))
    assert not bool(frame.iloc[0].played)


def test_the_guard_exempts_an_awarded_match():
    """An awarded match has no numeric score by definition."""
    from datetime import date

    box = BOX.replace("|score      = 2–1", "|score      = Awarded to Twente")
    frame = wikipedia.parse_matches(box, season="2024-25", competition="UEL",
                                    require_scores_before=date(2025, 3, 1))
    assert bool(frame.iloc[0].awarded)
    assert not bool(frame.iloc[0].played)


def test_the_guard_reports_every_offender_not_just_the_first():
    from datetime import date

    box = BOX.replace("2–1", "2?1").replace("0–0", "0?0")
    with pytest.raises(ValueError) as raised:
        wikipedia.parse_matches(box, season="2024-25", competition="UEL",
                                require_scores_before=date(2025, 3, 1))
    assert "2 match(es)" in str(raised.value)


def test_load_results_bars_only_the_window_it_uses():
    """min(cutoff, today), so a stale cache cannot break historical loads.

    A page whose latest matchday has no scores yet is normal mid-season. It
    must not make a backtest at an earlier cutoff fail, because those rows are
    filtered out anyway - but it must fail a load that would have used them.
    """
    from datetime import date

    from ucl.config import season

    cutoff = season("2026-27").cutoff_date
    # Every cached 2026-27 fixture is dated on or after the cutoff, so nothing
    # inside the window can be unscored however stale the cache is.
    frame = wikipedia.load_results(cutoff)
    assert frame.date.max().date() < cutoff


def test_an_unreadable_score_raises_even_on_a_tolerated_stale_read():
    """The two failure classes are not the same thing.

    A score the parser cannot read is a parser bug whatever the age of the
    page. An empty one only means the cache predates the result.
    """
    from datetime import date

    unreadable = BOX.replace("|score      = 2–1", "|score      = 2?1")
    with pytest.raises(ValueError, match="parsed no score"):
        wikipedia.parse_matches(unreadable, season="2024-25", competition="UEL",
                                require_scores_before=date(2025, 3, 1),
                                tolerate_missing=True)

    empty = BOX.replace("|score      = 2–1", "|score      =")
    frame = wikipedia.parse_matches(empty, season="2024-25", competition="UEL",
                                    require_scores_before=date(2025, 3, 1),
                                    tolerate_missing=True)
    assert not bool(frame.iloc[0].played)
    # ...and is still a failure when the page was just fetched.
    with pytest.raises(ValueError, match="parsed no score"):
        wikipedia.parse_matches(empty, season="2024-25", competition="UEL",
                                require_scores_before=date(2025, 3, 1),
                                tolerate_missing=False)
