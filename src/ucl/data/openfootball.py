"""UEFA club competition results from openfootball (Football.TXT on GitHub).

football-data.co.uk carries no European club football, so our own Elo has
nothing to link national leagues with. openfootball supplies UCL/UEL/UECL
results in the Football.TXT format::

    ▪ League, Matchday 1
      Tue Sep 16 2025
        18:45  Athletic Club (ESP)     v Arsenal FC (ENG)         0-2 (0-0)

Scores carry extra-time and penalty detail, which must be separated so a
knockout leg is not fed to the goal model as a 90-minute result::

    4-3 pen. 1-1 a.e.t. (1-1, 0-1)   -> pens 4-3, after ET 1-1, FT 1-1, HT 0-1

Two repositories are used, each where it is the superset: ``world.more`` for
2021-22..2023-24 (its files include the qualifying rounds) and
``champions-league`` for 2024-25 onwards (separate main and qualifying files).
Known gaps are declared in :data:`MISSING`, never filled by guesswork.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ucl.config import DATA_RAW
from ucl.data.cutoff import filter_to_cutoff
from ucl.data.manifest import ManifestEntry, entry_for, verify_cache, write_manifest

__all__ = ["CACHE_DIR", "SOURCES", "MISSING", "fetch", "load_results", "parse_file"]

CACHE_DIR = DATA_RAW / "openfootball"
RAW_URL = "https://raw.githubusercontent.com/openfootball/{repo}/master/{path}"


@dataclass(frozen=True)
class Source:
    season: str
    competition: str
    repo: str
    path: str

    @property
    def file_name(self) -> str:
        return f"{self.season}_{Path(self.path).name}"

    @property
    def url(self) -> str:
        return RAW_URL.format(repo=self.repo, path=self.path)


def _sources() -> list[Source]:
    out: list[Source] = []
    # world.more bundles qualifying with the main draw for the older seasons.
    for season in ("2021-22", "2022-23", "2023-24"):
        for comp, stem in (("UCL", "uefa.cl"), ("UEL", "uefa.el"), ("UECL", "uefa.conf")):
            out.append(Source(season, comp, "world.more", f"{season}/{stem}.txt"))
    # champions-league splits main and qualifying; 2025-26 UEL/UECL main are absent.
    for season, stems in (
        ("2024-25", (("UCL", "cl"), ("UCL", "clq"), ("UEL", "el"), ("UEL", "elq"),
                     ("UECL", "conf"), ("UECL", "confq"))),
        ("2025-26", (("UCL", "cl"), ("UCL", "clq"), ("UEL", "elq"), ("UECL", "confq"))),
    ):
        for comp, stem in stems:
            out.append(Source(season, comp, "champions-league", f"{season}/{stem}.txt"))
    return out


SOURCES: list[Source] = _sources()

# Declared, not silently absorbed: openfootball publishes nothing for these.
MISSING: dict[tuple[str, str], str] = {
    ("2025-26", "UEL"): "main phase absent upstream (only elq.txt exists)",
    ("2025-26", "UECL"): "main phase absent upstream (only confq.txt exists)",
    ("2026-27", "UCL"): "no 2026-27 directory upstream",
    ("2026-27", "UEL"): "no 2026-27 directory upstream",
    ("2026-27", "UECL"): "no 2026-27 directory upstream",
}

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

_STAGE_RE = re.compile(r"^▪\s*(.+?)\s*$")
_DATE_RE = re.compile(r"^\s{1,4}[A-Z][a-z]{2}\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
_MATCH_RE = re.compile(
    r"^\s+(?:(?P<time>\d{1,2}:\d{2})\s+)?"
    r"(?P<home>.+?)\s+v\s+(?P<away>.+?)"
    r"(?:\s{2,}(?P<result>.*?))?\s*$"
)
_CLUB_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<country>[A-Z]{3})\)$")
_SCORE_RE = re.compile(
    r"^(?:(?P<ph>\d+)-(?P<pa>\d+)\s+pen\.\s+)?"
    r"(?P<a>\d+)-(?P<b>\d+)"
    r"(?P<aet>\s+a\.e\.t\.)?"
    r"(?:\s*\((?P<p1h>\d+)-(?P<p1a>\d+)(?:,\s*(?P<p2h>\d+)-(?P<p2a>\d+))?\))?"
    r"(?P<flags>.*)$"
)


def fetch(cache_dir: Path | None = None, *, sources: list[Source] | None = None) -> list[str]:
    """Download the declared sources into the cache and record a manifest.

    Returns the names of files that could not be fetched, so a disappearing
    upstream file surfaces instead of silently shrinking the data.
    """
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ManifestEntry] = []
    failed: list[str] = []
    for source in sources or SOURCES:
        target = cache_dir / source.file_name
        try:
            with urllib.request.urlopen(source.url, timeout=30) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            failed.append(f"{source.file_name}: {error}")
            continue
        target.write_bytes(payload)
        entries.append(entry_for(target, source.url))
    if entries:
        write_manifest(cache_dir, entries)
    return failed


def _parse_score(text: str) -> dict[str, object]:
    """Split a Football.TXT result into 90-minute, extra-time and penalty parts."""
    text = (text or "").strip()
    blank = {
        "hg": None, "ag": None, "ht_hg": None, "ht_ag": None,
        "aet": False, "et_hg": None, "et_ag": None,
        "pens_h": None, "pens_a": None, "awarded": False, "played": False,
    }
    if not text or text.startswith("["):
        return blank
    match = _SCORE_RE.match(text)
    if not match:
        return blank
    g = match.groupdict()
    awarded = "awarded" in (g["flags"] or "")
    if g["aet"]:
        # Leading pair is the extra-time score; the first bracket is full time.
        result = {
            "hg": int(g["p1h"]) if g["p1h"] is not None else None,
            "ag": int(g["p1a"]) if g["p1a"] is not None else None,
            "ht_hg": int(g["p2h"]) if g["p2h"] is not None else None,
            "ht_ag": int(g["p2a"]) if g["p2a"] is not None else None,
            "aet": True,
            "et_hg": int(g["a"]), "et_ag": int(g["b"]),
        }
    else:
        result = {
            "hg": int(g["a"]), "ag": int(g["b"]),
            "ht_hg": int(g["p1h"]) if g["p1h"] is not None else None,
            "ht_ag": int(g["p1a"]) if g["p1a"] is not None else None,
            "aet": False, "et_hg": None, "et_ag": None,
        }
    result["pens_h"] = int(g["ph"]) if g["ph"] is not None else None
    result["pens_a"] = int(g["pa"]) if g["pa"] is not None else None
    result["awarded"] = awarded
    result["played"] = result["hg"] is not None and not awarded
    return result


def parse_file(path: Path, *, season: str, competition: str) -> pd.DataFrame:
    """Parse one Football.TXT file into match rows."""
    path = Path(path)
    stage = ""
    year: int | None = None
    current: date | None = None
    rows: list[dict[str, object]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("="):
            continue
        if line.lstrip().startswith("#"):
            # Header comment; the first one carries the season's opening year.
            found = re.search(r"\b(\d{4})\b", line)
            if year is None and found:
                year = int(found.group(1))
            continue

        stage_match = _STAGE_RE.match(line)
        if stage_match:
            stage = stage_match.group(1)
            continue

        date_match = _DATE_RE.match(line)
        if date_match:
            month_name, day, explicit_year = date_match.groups()
            if explicit_year:
                year = int(explicit_year)
            month = _MONTHS[month_name]
            if year is None:
                raise ValueError(f"{path.name}: date before any year was seen: {line!r}")
            candidate = date(year, month, int(day))
            # Only the first date of a new year carries it; roll over otherwise.
            if current is not None and candidate < current:
                year += 1
                candidate = date(year, month, int(day))
            current = candidate
            continue

        if " v " not in line:
            continue
        match = _MATCH_RE.match(line)
        if not match or current is None:
            continue
        home_raw, away_raw = match.group("home").strip(), match.group("away").strip()
        home_club, away_club = _CLUB_RE.match(home_raw), _CLUB_RE.match(away_raw)
        if not (home_club and away_club):
            continue
        row: dict[str, object] = {
            "date": pd.Timestamp(current),
            "home": home_club.group("name"),
            "away": away_club.group("name"),
            "home_country": home_club.group("country"),
            "away_country": away_club.group("country"),
            "country": "EUR",
            "league": competition,
            "season": season,
            "competition": competition,
            "stage": stage,
            "source_file": path.name,
        }
        row.update(_parse_score(match.group("result") or ""))
        rows.append(row)

    columns = [
        "date", "home", "away", "home_country", "away_country", "country", "league",
        "season", "competition", "stage", "source_file", "hg", "ag", "ht_hg", "ht_ag",
        "aet", "et_hg", "et_ag", "pens_h", "pens_a", "awarded", "played",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    # Fix dtypes here so files without any extra-time match concatenate cleanly.
    for column in ("hg", "ag", "ht_hg", "ht_ag", "et_hg", "et_ag", "pens_h", "pens_a"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("aet", "awarded", "played"):
        frame[column] = frame[column].fillna(False).astype(bool)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_results(
    cutoff: date,
    *,
    cache_dir: Path | None = None,
    competitions: list[str] | None = None,
    played_only: bool = True,
    verify: bool = True,
) -> pd.DataFrame:
    """Load cached European results dated strictly before ``cutoff``.

    ``played_only`` drops cancelled and awarded matches, which carry no
    footballing information about scoring rates.
    """
    cache_dir = Path(cache_dir or CACHE_DIR)
    chosen = [s for s in SOURCES if competitions is None or s.competition in competitions]
    present = [s for s in chosen if (cache_dir / s.file_name).exists()]
    if not present:
        raise FileNotFoundError(
            f"no cached openfootball files in {cache_dir}; run scripts/fetch_data.py"
        )
    if verify:
        verify_cache(cache_dir, files=[s.file_name for s in present])

    frames = [
        parse_file(cache_dir / s.file_name, season=s.season, competition=s.competition)
        for s in present
    ]
    combined = pd.concat(frames, ignore_index=True)
    if played_only:
        combined = combined[combined["played"]].copy()
        combined["hg"] = combined["hg"].astype(int)
        combined["ag"] = combined["ag"].astype(int)
    combined = filter_to_cutoff(
        combined, cutoff, source=f"openfootball cache ({len(present)} files)"
    )
    return combined.sort_values(["date", "competition", "home"]).reset_index(drop=True)


def missing_report() -> pd.DataFrame:
    """The declared upstream gaps, as rows, for the coverage report."""
    return pd.DataFrame(
        [{"season": s, "competition": c, "gap": why} for (s, c), why in MISSING.items()]
    )
