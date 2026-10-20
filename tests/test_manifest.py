"""Cache integrity: a raw file that changed under us must never be loaded silently."""

import json

import pytest

from ucl.data.manifest import (
    CacheIntegrityError,
    CacheNameError,
    canonical_cache_name,
    entry_for,
    near_misses,
    require_exact_cache_names,
    read_manifest,
    untracked_files,
    verify_cache,
    write_manifest,
)


@pytest.fixture
def cache(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("Date,HomeTeam\n01/01/2026,Arsenal\n")
    write_manifest(tmp_path, [entry_for(path, "https://example.test/a.csv")])
    return tmp_path


def test_clean_cache_verifies(cache):
    verified = verify_cache(cache)
    assert list(verified) == ["a.csv"]
    assert verified["a.csv"].source_url == "https://example.test/a.csv"


def test_tampered_file_is_rejected(cache):
    (cache / "a.csv").write_text("Date,HomeTeam\n01/01/2026,Chelsea\n")
    with pytest.raises(CacheIntegrityError, match="sha256 mismatch"):
        verify_cache(cache)


def test_missing_file_is_rejected(cache):
    (cache / "a.csv").unlink()
    with pytest.raises(CacheIntegrityError, match="missing on disk"):
        verify_cache(cache)


def test_unrecorded_file_is_rejected_when_requested(cache):
    (cache / "b.csv").write_text("x\n")
    assert untracked_files(cache) == ["b.csv"]
    with pytest.raises(CacheIntegrityError, match="not in manifest.json"):
        verify_cache(cache, files=["b.csv"])


def test_absent_manifest_is_rejected(tmp_path):
    with pytest.raises(CacheIntegrityError, match="no manifest.json"):
        verify_cache(tmp_path)


def test_manifest_roundtrip_keeps_provenance(cache):
    entry = read_manifest(cache)["a.csv"]
    assert entry.bytes == (cache / "a.csv").stat().st_size
    assert len(entry.sha256) == 64
    assert json.loads((cache / "manifest.json").read_text())[0]["file"] == "a.csv"


# --- near-miss cache directory names ---------------------------------------
# A fetch unzipped into "football_data_current " - one trailing space, from the
# zip's own file name - and the loader read the cache as absent, which is a
# legitimate state for that one. A forecast was published from domestic results
# five months old and nothing in a terminal listing showed why.

#: Every way the same folder has been or could be misnamed. Each must reduce to
#: the canonical name, or the loader goes on reading the real cache as absent.
NEAR_MISSES = [
    "football_data_current ",            # the one that happened
    " football_data_current",
    "football_data_current  ",
    "\tfootball_data_current",
    "football_data_current\u00a0",       # no-break space
    "football_data_current\u2007",       # figure space
    "football_data_current\u200b",       # zero-width space
    "football_data_current\ufeff",       # byte-order mark
    "football_data_current.",            # Windows drops a trailing dot
    "Football_Data_Current",             # case
    "FOOTBALL_DATA_CURRENT",
    "football_data_current 2",           # a second copy
    "football_data_current (2)",
    "football_data_current(3)",
    "football_data_current copy",
    "football_data_current_copy",
    "football_data_current-copy",
    "football_data_current copy 2",
    "football_data_current 2 copy",
    "football_data_current Kopie",
    "football-data-current",             # separator
    "football data current",
]


@pytest.mark.parametrize("name", NEAR_MISSES)
def test_every_near_miss_name_reduces_to_the_real_one(name):
    assert canonical_cache_name(name) == "football_data_current"


@pytest.mark.parametrize("name", [
    "football_data", "football_data_history", "football_data_2025",
    "football_data_current_2025", "openfootball", "wikipedia",
])
def test_a_genuinely_different_cache_is_not_a_near_miss(name):
    """``_2025`` is how a separate cache would be named, not a typo for one.

    Flagging it would refuse every load for a directory nobody got wrong.
    """
    assert canonical_cache_name(name) != "football_data_current"


def test_the_folder_that_was_read_as_absent_is_found_and_named(tmp_path):
    (tmp_path / "football_data_current ").mkdir()
    expected = tmp_path / "football_data_current"
    assert near_misses(expected) == [tmp_path / "football_data_current "]
    with pytest.raises(CacheNameError) as error:
        require_exact_cache_names(expected)
    message = str(error.value)
    # The folder, quoted so the trailing space is visible; why; and the fix.
    assert "'football_data_current '" in message
    assert "trailing whitespace" in message
    assert "absent" in message and "mv " in message


def test_a_stray_copy_beside_the_real_cache_is_also_a_fault(tmp_path):
    """The loaders read one and ignore the other, with nothing to say which."""
    (tmp_path / "football_data").mkdir()
    (tmp_path / "football_data copy").mkdir()
    with pytest.raises(CacheNameError, match="also exists"):
        require_exact_cache_names(tmp_path / "football_data")


def test_every_near_miss_is_listed_in_one_go(tmp_path):
    (tmp_path / "wikipedia ").mkdir()
    (tmp_path / "transfermarkt 2").mkdir()
    with pytest.raises(CacheNameError) as error:
        require_exact_cache_names(tmp_path / "wikipedia", tmp_path / "transfermarkt",
                                  tmp_path / "openfootball")
    message = str(error.value)
    assert "2 cache directory name(s)" in message
    assert "'wikipedia '" in message and "'transfermarkt 2'" in message


def test_an_exact_name_and_a_missing_parent_are_both_silent(tmp_path):
    (tmp_path / "football_data_current").mkdir()
    require_exact_cache_names(tmp_path / "football_data_current")
    require_exact_cache_names(tmp_path / "nowhere" / "football_data_current")
    assert near_misses(tmp_path / "nowhere" / "football_data") == []


def test_a_file_with_a_near_miss_name_is_not_a_cache(tmp_path):
    """Only directories count: ``football_data_current.zip`` is the download."""
    (tmp_path / "football_data_current ").write_text("not a directory")
    assert near_misses(tmp_path / "football_data_current") == []
