"""Football.TXT parsing: extra time, penalties and the implicit year must be right."""

from datetime import date

import pandas as pd
import pytest

from ucl.data.openfootball import _parse_score, load_results, parse_file

SAMPLE = """= UEFA Champions League 2025/26

# Date       Tue Sep 16 2025 - Sat May 30 2026 (256d)

▪ League, Matchday 1
  Tue Sep 16 2025
    18:45  Athletic Club (ESP)     v Arsenal FC (ENG)         0-2 (0-0)
           SK Slavia Praha (CZE)   v FK Bodø/Glimt (NOR)      2-2 (1-0)
  Wed Sep 17
    21:00  Liverpool FC (ENG)      v Club Atlético de Madrid (ESP)  3-2 (2-1)

▪ Finals, Final
  Wed Jan 28
    21:00  Paris Saint-Germain FC (FRA) v Arsenal FC (ENG)    4-3 pen. 1-1 a.e.t. (1-1, 0-1)
    21:00  RB Leipzig (GER)        v Spartak Moskva (RUS)     [cancelled]
    21:00  Tottenham Hotspur (ENG) v Stade Rennais (FRA)      0-3    [awarded]
"""


@pytest.fixture
def parsed(tmp_path):
    path = tmp_path / "2025-26_cl.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    return parse_file(path, season="2025-26", competition="UCL")


def test_plain_result_uses_full_time_and_half_time(parsed):
    row = parsed.iloc[0]
    assert (row.home, row.away) == ("Athletic Club", "Arsenal FC")
    assert (row.hg, row.ag) == (0, 2)
    assert (row.ht_hg, row.ht_ag) == (0, 0)
    assert not row.aet and row.played


def test_country_codes_are_split_out(parsed):
    row = parsed.iloc[1]
    assert (row.home, row.home_country) == ("SK Slavia Praha", "CZE")
    assert (row.away, row.away_country) == ("FK Bodø/Glimt", "NOR")


def test_implicit_year_is_carried_then_rolled_over(parsed):
    # Sep dates inherit 2025; the January final must become 2026, not 2025.
    assert parsed.iloc[0].date.date() == date(2025, 9, 16)
    assert parsed.iloc[2].date.date() == date(2025, 9, 17)
    assert parsed.iloc[3].date.date() == date(2026, 1, 28)


def test_penalty_shootout_keeps_ninety_minute_score(parsed):
    row = parsed.iloc[3]
    # 4-3 pen. 1-1 a.e.t. (1-1, 0-1): the goal model must see the 90-minute 1-1.
    assert (row.hg, row.ag) == (1, 1)
    assert (row.et_hg, row.et_ag) == (1, 1)
    assert (row.pens_h, row.pens_a) == (4, 3)
    assert (row.ht_hg, row.ht_ag) == (0, 1)
    assert row.aet and row.played


def test_cancelled_and_awarded_are_not_playable_results(parsed):
    cancelled = parsed.iloc[4]
    awarded = parsed.iloc[5]
    assert not cancelled.played and pd.isna(cancelled.hg)
    # An awarded 3-0 is a disciplinary result, not evidence about scoring rates.
    assert awarded.awarded and not awarded.played and (awarded.hg, awarded.ag) == (0, 3)


def test_extra_time_without_half_time_bracket():
    score = _parse_score("1-0 a.e.t. (0-0)")
    assert (score["hg"], score["ag"]) == (0, 0)
    assert (score["et_hg"], score["et_ag"]) == (1, 0)
    assert score["aet"] and score["pens_h"] is None


def test_cached_european_results_respect_the_cutoff():
    from ucl.config import season

    cfg = season("2024-25")
    frame = load_results(cfg.cutoff_date)
    assert len(frame) > 1000
    assert frame.date.max().date() < cfg.cutoff_date
    assert set(frame.competition) <= {"UCL", "UEL", "UECL"}
