"""The main-phase classifier, pinned to the competition format.

`phase` alone cannot do this job: the openfootball `world.more` files for
2021-22 to 2023-24 carry `phase == "mixed"` because one file holds the
qualifying rounds and the main draw together. The stage string decides there,
and the strings are treacherous - "Playoffs" means the qualifying play-off round
in one season and the new format's knockout play-off in another.

The counts asserted here come from the regulations (96 group-stage matches in a
32-team format, 144 league-phase matches in the 36-team one), not from a data
pull, so they do not go stale when the cache moves.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ucl.data import european

CUTOFF = date(2026, 9, 7)
#: 32 clubs in 8 groups of 4, home and away: 8 * 4 * 3 = 96.
GROUP_STAGE = 96
#: 36 clubs playing 8 each, in a single table: 36 * 8 / 2 = 144.
LEAGUE_PHASE = 144
#: Plus the knockout: 96 + 16 + 8 + 4 + 1 for the old format,
#: 144 + 16 + 16 + 8 + 4 + 1 for the new one.
MAIN_PHASE = {"old": 125, "new": 189}


@pytest.fixture(scope="module")
def results() -> pd.DataFrame:
    frame = european.load_results(CUTOFF)
    return frame.assign(main=european.is_main_phase(frame))


def test_the_classifier_finds_every_group_or_league_phase_match(results):
    """96 a season while the format had groups, 144 once it had a league."""
    ucl = results[results.competition.eq("UCL") & results.main]
    stage = ucl.stage.fillna("").str.strip()
    phase = ucl[stage.str.match(r"^(?:group|gruppe|league, matchday)", case=False)]
    for season, group in phase.groupby("season"):
        expected = LEAGUE_PHASE if season >= "2024-25" else GROUP_STAGE
        assert len(group) == expected, f"{season}: {len(group)} != {expected}"
    assert len(set(phase.season)) >= 8, "too few seasons to be a real check"


def test_the_main_phase_count_matches_the_format(results):
    """Group or league phase plus the knockout, per season.

    2019-20 is excluded: COVID cut its knockout to a single-leg tournament in
    Lisbon, so the format constant does not apply - but its group stage
    completed, which is why the test above still covers it.
    """
    ucl = results[results.competition.eq("UCL") & results.main]
    for season, group in ucl.groupby("season"):
        if season in ("2019-20", "2026-27"):
            continue
        expected = MAIN_PHASE["new"] if season >= "2024-25" else MAIN_PHASE["old"]
        assert len(group) == expected, f"{season}: {len(group)} != {expected}"


def test_the_qualifying_rounds_score_lower_than_the_draw_they_feed(results):
    """The reason the two are separated at all."""
    ucl = results[results.competition.eq("UCL")]
    goals = ucl.hg + ucl.ag
    main = goals[ucl.main].mean()
    qualifying = goals[~ucl.main].mean()
    assert main > qualifying + 0.2, (main, qualifying)


def test_the_ambiguous_playoffs_string_is_resolved_both_ways(results):
    """"Playoffs" is qualifying in one place and main phase in another.

    2024-25 carries both spellings of the same word, which is why `phase` is
    consulted first and the stage rule is confined to the mixed files.
    """
    ambiguous = results[results.stage.str.strip().str.fullmatch(
        r"play-?offs", case=False, na=False)]
    assert not ambiguous.empty
    by_phase = ambiguous.groupby("phase").main.nunique()
    assert by_phase.get("qualifying", 1) == 1
    assert not ambiguous[ambiguous.phase.eq("qualifying")].main.any()
    mixed = ambiguous[ambiguous.phase.eq("mixed")]
    assert not mixed.empty and not mixed.main.any(), (
        "the August qualifying play-off in a mixed file must not read as main phase")


def test_the_february_knockout_play_off_is_main_phase(results):
    """It shares a word with the qualifier and is the opposite of it."""
    knockout = results[results.stage.str.contains(
        "knockout", case=False, na=False)]
    assert not knockout.empty
    assert knockout.main.all()


def test_an_unknown_stage_in_a_mixed_file_raises(results):
    """A near miss is not a legitimate value - the cache-guard rule.

    Defaulting an unrecognised string to main phase would quietly move
    qualifying matches into a competition's own goal-scale term, and nothing
    downstream would look wrong.
    """
    frame = pd.DataFrame({
        "phase": ["mixed", "mixed"],
        "stage": ["Group A - 1", "Seventh Preliminary Qualifying Playoff Round"],
        "competition": ["UCL", "UCL"],
    })
    with pytest.raises(european.PhaseError) as caught:
        european.is_main_phase(frame)
    assert "Seventh Preliminary Qualifying Playoff Round" in str(caught.value)


def test_a_reliable_phase_never_consults_the_stage_string(results):
    """Where `phase` is authoritative it decides alone, whatever the stage says."""
    frame = pd.DataFrame({
        "phase": ["main", "qualifying"],
        "stage": ["1st Qualifying Round", "Group A - 1"],
        "competition": ["UCL", "UCL"],
    })
    assert list(european.is_main_phase(frame)) == [True, False]


def test_every_cached_stage_string_is_classifiable(results):
    """The whole cache passes, so the raise above is about future layouts."""
    assert results.main.notna().all()
    assert results.stage.fillna("").str.strip().ne("").all()
