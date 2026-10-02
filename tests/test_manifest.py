"""Cache integrity: a raw file that changed under us must never be loaded silently."""

import json

import pytest

from ucl.data.manifest import (
    CacheIntegrityError,
    entry_for,
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
