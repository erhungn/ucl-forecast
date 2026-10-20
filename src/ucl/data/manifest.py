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
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "CacheIntegrityError", "CacheNameError", "ManifestEntry", "canonical_cache_name",
    "difference_from", "near_misses", "read_manifest", "require_exact_cache_names",
    "verify_cache", "sha256_of", "write_manifest",
]

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


#: Characters that can sit in a folder name without showing in a listing: the
#: zero-width family, the soft hyphen and the byte-order mark. Every Unicode
#: space is handled separately, by :meth:`str.isspace`.
#: Written as escapes on purpose: a source line about characters that
#: cannot be seen must not itself contain any.
_INVISIBLE = {ord(ch): None for ch in (
    "\u200b"      # zero-width space
    "\u200c"      # zero-width non-joiner
    "\u200d"      # zero-width joiner
    "\u2060"      # word joiner
    "\ufeff"      # byte-order mark
    "\u00ad"      # soft hyphen
)}

#: A trailing duplicate marker, as a zip tool or a Finder copy leaves it: " 2",
#: " (2)", " copy", "_copy", " copy 2", " Kopie". Deliberately **not**
#: ``_2025``: an underscore and digits are how a legitimately separate cache
#: would be named, and flagging ``football_data_2025`` as a misspelling of
#: ``football_data`` would block every load for no reason.
_DUPLICATE = re.compile(
    r"(?:\s*\(\s*\d+\s*\)|\s+\d+|[\s_-]*cop(?:y|ie)(?:\s*\d+)?|[\s_-]*kopie(?:\s*\d+)?)$",
    re.IGNORECASE,
)


class CacheNameError(RuntimeError):
    """Raised when a cache directory is on disk under a near-miss name.

    This is the loud end of a failure that was silent once: a fetch unzipped
    into ``football_data_current `` - one trailing space, from the zip's own
    file name - so the loader saw no such directory, reported the cache as
    absent, and published a forecast with no current-season domestic results
    at all. Nothing in a terminal listing shows a trailing space.
    """


def canonical_cache_name(name: str) -> str:
    """A cache directory's name reduced to what the user meant by it.

    Case, every kind of whitespace, the invisible characters above, a trailing
    dot, a duplicate marker, and ``-``/space where ``_`` belongs all collapse,
    so two names that differ only in those ways compare equal.
    """
    text = unicodedata.normalize("NFC", str(name)).translate(_INVISIBLE)
    text = "".join(" " if ch.isspace() else ch for ch in text)
    text = re.sub(r"\s+", " ", text).strip(" .").casefold()
    while True:                             # " copy 2" needs two passes
        shorter = _DUPLICATE.sub("", text).strip(" .")
        if shorter == text or not shorter:
            break
        text = shorter
    return re.sub(r"[ \-]+", "_", text)


def difference_from(found: str, expected: str) -> str:
    """Why two near-miss names differ, in words, for the error message."""
    if found.strip() != found:
        where = ("leading and trailing" if found[:1].isspace() and found[-1:].isspace()
                 else "leading" if found[:1].isspace() else "trailing")
        return f"{where} whitespace"
    if found.casefold() == expected.casefold():
        return "letter case"
    if _DUPLICATE.search(found.strip(" .")):
        return "a duplicate suffix"
    separators = found.strip(" .").casefold().replace("-", "_").replace(" ", "_")
    if separators == expected.casefold():
        return "a separator"
    return "characters that do not show in a listing"


def near_misses(expected: Path) -> list[Path]:
    """Sibling directories whose name is a near miss of ``expected``'s."""
    expected = Path(expected)
    if not expected.parent.is_dir():
        return []
    target = canonical_cache_name(expected.name)
    return sorted(
        child for child in expected.parent.iterdir()
        if child.is_dir() and child.name != expected.name
        and canonical_cache_name(child.name) == target
    )


def require_exact_cache_names(*expected: Path) -> None:
    """Raise :class:`CacheNameError` for every near-miss cache directory found.

    Checked whether or not the real directory exists. Both cases are faults:
    without it the cache reads as absent, and beside it the loaders silently
    read one copy and ignore the other.
    """
    problems: list[str] = []
    for directory in expected:
        directory = Path(directory)
        for found in near_misses(directory):
            problems.append(
                f"{found.parent}/{found.name!r} differs from "
                f"{directory.name!r} only by {difference_from(found.name, directory.name)}"
                + (f", and {directory.name!r} also exists, so the loaders read "
                   f"that one and ignore this. Compare the two and delete the "
                   f"stray copy" if directory.is_dir()
                   else f", so the loaders report that cache as absent. Rename "
                        f'it:  mv "{found}" "{directory}"')
                + "."
            )
    if problems:
        raise CacheNameError(
            f"{len(problems)} cache directory name(s) are a near miss:\n  "
            + "\n  ".join(problems)
        )
