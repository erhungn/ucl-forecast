"""Provenance and integrity for cached raw downloads.

Some sources cannot be reached from every machine (football-data.co.uk is
DNS-blocked on the author's network), so raw files are fetched once, by hand or
by script, and recorded in a ``manifest.json`` next to them: source URL, UTC
timestamp, SHA-256 and size per file. Loaders verify the hashes on every load
and never re-fetch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["CacheIntegrityError", "ManifestEntry", "read_manifest", "verify_cache", "sha256_of", "write_manifest"]

MANIFEST_NAME = "manifest.json"


class CacheIntegrityError(RuntimeError):
    """Raised when a cached file is missing, unrecorded, or hash-mismatched."""


@dataclass(frozen=True)
class ManifestEntry:
    file: str
    source_url: str
    downloaded_at_utc: str
    sha256: str
    bytes: int


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(cache_dir: Path) -> dict[str, ManifestEntry]:
    path = Path(cache_dir) / MANIFEST_NAME
    if not path.exists():
        raise CacheIntegrityError(
            f"no {MANIFEST_NAME} in {cache_dir}; raw files must be recorded before use"
        )
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return {
        row["file"]: ManifestEntry(
            file=row["file"],
            source_url=row["source_url"],
            downloaded_at_utc=row["downloaded_at_utc"],
            sha256=row["sha256"],
            bytes=int(row["bytes"]),
        )
        for row in rows
    }


def write_manifest(cache_dir: Path, entries: list[ManifestEntry]) -> Path:
    path = Path(cache_dir) / MANIFEST_NAME
    payload = [
        {
            "file": e.file,
            "source_url": e.source_url,
            "downloaded_at_utc": e.downloaded_at_utc,
            "sha256": e.sha256,
            "bytes": e.bytes,
        }
        for e in sorted(entries, key=lambda e: e.file)
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def entry_for(path: Path, source_url: str) -> ManifestEntry:
    """Build a manifest entry for a file that was just downloaded."""
    path = Path(path)
    return ManifestEntry(
        file=path.name,
        source_url=source_url,
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
        sha256=sha256_of(path),
        bytes=path.stat().st_size,
    )


def verify_cache(cache_dir: Path, *, files: list[str] | None = None) -> dict[str, ManifestEntry]:
    """Re-hash cached files against the manifest and return the verified entries.

    ``files`` limits the check to specific names; by default every manifest
    entry is checked. Raises :class:`CacheIntegrityError` on the first problem,
    listing all of them rather than only the first.
    """
    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    wanted = list(manifest) if files is None else list(files)

    problems: list[str] = []
    for name in wanted:
        entry = manifest.get(name)
        if entry is None:
            problems.append(f"{name}: not in {MANIFEST_NAME}")
            continue
        path = cache_dir / name
        if not path.exists():
            problems.append(f"{name}: recorded in {MANIFEST_NAME} but missing on disk")
            continue
        actual = sha256_of(path)
        if actual != entry.sha256:
            problems.append(
                f"{name}: sha256 mismatch (manifest {entry.sha256[:12]}…, "
                f"on disk {actual[:12]}…)"
            )
        elif path.stat().st_size != entry.bytes:
            problems.append(f"{name}: size mismatch")

    if problems:
        raise CacheIntegrityError(
            f"{cache_dir}: {len(problems)} cache problem(s):\n  " + "\n  ".join(problems)
        )
    return {name: manifest[name] for name in wanted}


def untracked_files(cache_dir: Path, suffixes: tuple[str, ...] = (".csv", ".txt")) -> list[str]:
    """Files present in the cache but absent from the manifest."""
    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    return sorted(
        p.name
        for p in cache_dir.iterdir()
        if p.is_file() and p.suffix in suffixes and p.name not in manifest
    )
