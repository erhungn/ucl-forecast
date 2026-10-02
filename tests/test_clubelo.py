"""ClubElo is optional: an outage must be reported, never raised, never faked."""

import urllib.error
from datetime import date

from ucl.data import clubelo

DAY = date(2026, 9, 7)


def _fail(*_args, **_kwargs):
    raise urllib.error.HTTPError("http://api.clubelo.com/x", 502, "Bad Gateway", {}, None)


def test_outage_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _fail)
    slept: list[float] = []
    status = clubelo.fetch_snapshot(
        DAY, cache_dir=tmp_path, attempts=3, base_delay=1.0, sleep=slept.append
    )
    assert not status.available
    assert status.attempts == 3
    assert "502" in status.detail
    assert "UNAVAILABLE" in status.summary()
    # Exponential backoff, and no sleep after the final attempt.
    assert slept == [1.0, 2.0]


def test_missing_snapshot_loads_as_none(tmp_path):
    assert clubelo.load_snapshot(DAY, cache_dir=tmp_path) is None


def test_successful_fetch_is_cached_and_hashed(tmp_path, monkeypatch):
    class Response:
        def read(self):
            return b"Rank,Club,Country,Level,Elo\n1,Arsenal,ENG,1,2050.3\n"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(clubelo.urllib.request, "urlopen", lambda *a, **k: Response())
    status = clubelo.fetch_snapshot(DAY, cache_dir=tmp_path)
    assert status.available and status.attempts == 1

    frame = clubelo.load_snapshot(DAY, cache_dir=tmp_path)
    assert list(frame.columns) == ["rank", "club", "country", "level", "elo"]

    # A second call must reuse the cache rather than hit the network again.
    monkeypatch.setattr(clubelo.urllib.request, "urlopen", _fail)
    again = clubelo.fetch_snapshot(DAY, cache_dir=tmp_path)
    assert again.available and again.detail == "already cached"
