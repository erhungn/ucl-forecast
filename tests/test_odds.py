"""The outright benchmark, and the discipline it is held to.

The market never enters an objective, but a benchmark computed from prices the
forecast could not have seen would flatter or damn it for knowing something it
did not - so the odds are subject to the same cutoff rule as the model.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ucl.config import SEASONS
from ucl.data import names, odds
from ucl.data.cutoff import LookAheadError

SEASON = "2026-27"
CUTOFF = SEASONS[SEASON].cutoff_date


def test_both_markets_are_found():
    found = {path.name for path in odds.outright_paths(SEASON)}
    assert "outright_odds_2026-27.csv" in found
    assert "outright_odds_kalshi_2026-27.csv" in found


def test_every_market_de_vigs_to_one():
    frame = odds.load_all_outrights(SEASON, cutoff=CUTOFF)
    for market, group in frame.groupby("market"):
        assert group.p_market.sum() == pytest.approx(1.0), market


def test_a_price_quoted_only_as_a_bound_is_listed_but_not_priced():
    """Kalshi quoted 16 clubs as "<1c", which is an interval, not a price."""
    frame = odds.load_outright(SEASON, path=odds.DATA_RAW /
                               "outright_odds_kalshi_2026-27.csv")
    assert frame.attrs["listed"] == 29
    assert frame.attrs["priced"] == 13
    assert len(frame) == 13
    assert frame.decimal_odds.notna().all()


def test_the_overrounds_differ_as_an_exchange_and_a_bookmaker_should():
    frame = odds.load_all_outrights(SEASON, cutoff=CUTOFF)
    markets = frame.attrs["markets"]
    bookmaker = markets["maltafootball.com"]["overround"]
    exchange = markets["kalshi via oddsshopper.com"]["overround"]
    assert bookmaker == pytest.approx(1.3078, abs=5e-4)
    # Over the priced subset only; the 16 censored clubs would add at most 16c,
    # so the exchange's true figure is somewhere in 1.025 to 1.185.
    assert exchange == pytest.approx(1.0250, abs=5e-4)
    assert exchange < bookmaker


def test_a_post_cutoff_price_is_refused(tmp_path):
    """The benchmark is held to the model's own no-look-ahead rule."""
    path = tmp_path / "outright_odds_late_2026-27.csv"
    pd.DataFrame([{"market_name": "Arsenal", "team": "Arsenal",
                   "fractional_odds": "", "decimal_odds": 5.0,
                   "source": "test", "odds_date": "2026-09-08", "note": ""}]
                 ).to_csv(path, index=False)
    frame = odds.load_outright(SEASON, path=path)
    frame["market"] = "test"
    with pytest.raises(LookAheadError, match="on or after the cutoff"):
        from ucl.data.cutoff import assert_no_lookahead

        assert_no_lookahead(frame, CUTOFF, date_column="odds_date")


def test_every_quoted_club_is_one_of_the_thirty_six():
    """A market's own spelling must resolve, or a price goes to nobody."""
    teams = set(names.load_teams().team)
    for path in odds.outright_paths(SEASON):
        frame = pd.read_csv(path)
        unknown = sorted(set(frame.team) - teams)
        assert unknown == [], f"{path.name}: {unknown}"


def test_the_kalshi_decimal_odds_match_the_cents_they_were_read_from():
    """The note records the quoted cents; the odds must be 100 / cents."""
    frame = pd.read_csv(odds.DATA_RAW / "outright_odds_kalshi_2026-27.csv")
    priced = frame[frame.decimal_odds.notna()]
    assert len(priced) == 13
    for row in priced.itertuples():
        cents = float(str(row.note).split("c.")[0])
        assert row.decimal_odds == pytest.approx(100.0 / cents, abs=5e-4)


def test_the_prices_predate_the_cutoff():
    frame = odds.load_all_outrights(SEASON, cutoff=CUTOFF)
    assert pd.to_datetime(frame.odds_date).max().date() < CUTOFF
    # And the two markets were observed on different days, as recorded.
    assert set(frame.odds_date) == {"2026-08-18", "2026-08-13"}


def test_the_markets_agree_on_favourites_and_diverge_on_longshots():
    """Favourite-longshot bias, measured rather than assumed.

    The bookmaker's margin is concentrated in the longshots, so its raw prices
    there are about twice the exchange's while the two agree within a few
    percent at the top. Proportional de-vigging assumes the margin is spread
    evenly, which is why the de-vigged bookmaker numbers understate the
    favourites.
    """
    frame = odds.load_all_outrights(SEASON, cutoff=CUTOFF)
    wide = frame.pivot_table(index="team", columns="market", values="p_raw").dropna()
    exchange, bookmaker = ("kalshi via oddsshopper.com", "maltafootball.com")
    assert len(wide) == 13
    ratio = wide[exchange] / wide[bookmaker]
    favourites = ratio.loc[["Paris Saint-Germain", "Real Madrid", "Manchester City"]]
    longshots = ratio.loc[["Aston Villa", "Napoli", "Borussia Dortmund"]]
    assert favourites.min() > 0.95
    assert longshots.max() < 0.70
