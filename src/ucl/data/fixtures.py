"""League-phase fixture lists, and the structural checks the format guarantees.

A fixtures file is a *schedule*, so unlike every other loader it may contain
dates after the cutoff — that is the point of it. What it must never carry is
results: the backtest seasons are built from openfootball files that do contain
scores, and those columns are dropped here so a fixture list can never leak an
outcome into fitting.

Sources, in the order the project uses them:

* 2024-25 and 2025-26: the cached openfootball ``cl.txt``, League stage only.
* 2026-27: the Wikipedia league-phase article's wikitext, which openfootball
  does not yet publish, cross-checked against uefa.com.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from ucl.config import DATA_RAW
from ucl.data.manifest import entry_for, read_manifest, verify_cache, write_manifest
from ucl.data.names import association, normalize_club_name
from ucl.data.openfootball import CACHE_DIR as OF_CACHE, parse_file

__all__ = [
    "FixtureValidationError", "WIKI_CACHE", "COLUMNS",
    "from_openfootball", "from_wikitext", "pots_from_wikitext",
    "fetch_wikitext", "validate", "fixtures_path", "pots_path",
]

WIKI_CACHE = DATA_RAW / "wikipedia"
COLUMNS = ["season", "matchday", "date", "home", "away", "home_country", "away_country"]

WIKI_URL = "https://en.wikipedia.org/w/index.php?title={title}&action=raw"
# Wikimedia rejects the default urllib agent with 403; their policy asks for a
# descriptive one naming the tool.
USER_AGENT = "ucl-forecast/0.1 (research project; contact via repository)"
WIKI_TITLES = {
    "2024-25": "2024%E2%80%9325_UEFA_Champions_League_league_phase",
    "2025-26": "2025%E2%80%9326_UEFA_Champions_League_league_phase",
    "2026-27": "2026%E2%80%9327_UEFA_Champions_League_league_phase",
}


class FixtureValidationError(AssertionError):
    """Raised when a fixture list violates the published league-phase format."""


def fixtures_path(season_label: str) -> Path:
    return DATA_RAW / f"fixtures_{season_label}.csv"


def pots_path(season_label: str) -> Path:
    return DATA_RAW / f"pots_{season_label}.csv"


# --- sources ---------------------------------------------------------------

def fetch_wikitext(
    season_label: str, *, cache_dir: Path | None = None, timeout: float = 25.0
) -> tuple[Path | None, str]:
    """Cache the league-phase article's wikitext.

    Returns ``(path, "")`` on success and ``(None, reason)`` on failure, so an
    unreachable source is reported rather than mistaken for an empty article.
    """
    cache_dir = Path(cache_dir or WIKI_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"league_phase_{season_label}.wikitext"
    if target.exists():
        return target, ""
    url = WIKI_URL.format(title=WIKI_TITLES[season_label])
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return None, f"{type(error).__name__}: {error}"
    target.write_bytes(payload)
    existing = []
    if (cache_dir / "manifest.json").exists():
        existing = [e for e in read_manifest(cache_dir).values() if e.file != target.name]
    write_manifest(cache_dir, [*existing, entry_for(target, url)])
    return target, ""


def from_openfootball(
    season_label: str, *, cache_dir: Path | None = None
) -> pd.DataFrame:
    """League-phase fixtures from the cached ``cl.txt``, with results stripped."""
    path = Path(cache_dir or OF_CACHE) / f"{season_label}_cl.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not cached; run scripts/fetch_data.py")
    verify_cache(path.parent, files=[path.name])
    frame = parse_file(path, season=season_label, competition="UCL")
    league = frame[frame.stage.str.startswith("League")].copy()
    league["matchday"] = (
        league.stage.str.extract(r"Matchday\s+(\d+)", expand=False).astype(int)
    )
    # Drop every score column: a schedule must not carry outcomes.
    return league[COLUMNS].sort_values(["matchday", "date", "home"]).reset_index(drop=True)


_WIKI_LINK = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")
_FBAICON = re.compile(r"fbaicon\|([A-Z]{3})")
_START_DATE = re.compile(r"\{\{Start date\|(\d{4})\|(\d{1,2})\|(\d{1,2})")


def _wiki_team(field: str) -> tuple[str, str] | None:
    """``{{fbaicon|ENG}} [[Arsenal F.C.|Arsenal]]`` -> ``('Arsenal', 'ENG')``."""
    name = _WIKI_LINK.search(field)
    country = _FBAICON.search(field)
    if not (name and country):
        return None
    return name.group(1).strip(), country.group(1)


def from_wikitext(text: str, season_label: str) -> pd.DataFrame:
    """Parse the ``Football box`` blocks of a league-phase article."""
    rows = []
    matchday = 0
    for chunk in re.split(r"(?m)^===\s*Matchday\s+(\d+)\s*===\s*$", text)[1:]:
        if chunk.strip().isdigit():
            matchday = int(chunk.strip())
            continue
        for block in chunk.split("{{#invoke:Football box|main")[1:]:
            block = block.split("\n}}")[0]
            fields = dict(
                (key.strip(), value.strip())
                for key, _, value in (
                    line.partition("=") for line in block.splitlines() if line.startswith("|")
                )
                for key in [key.lstrip("|")]
            )
            date = _START_DATE.search(fields.get("date", ""))
            home = _wiki_team(fields.get("team1", ""))
            away = _wiki_team(fields.get("team2", ""))
            if not (date and home and away):
                continue
            rows.append(
                {
                    "season": season_label,
                    "matchday": matchday,
                    "date": pd.Timestamp(int(date.group(1)), int(date.group(2)), int(date.group(3))),
                    "home": home[0],
                    "away": away[0],
                    "home_country": home[1],
                    "away_country": away[1],
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["matchday", "date", "home"]
    ).reset_index(drop=True)


def pots_from_wikitext(text: str, season_label: str) -> pd.DataFrame:
    """Parse the four seeding-pot tables."""
    rows = []
    for block in re.split(r"\|\+\s*Pot\s+(\d)", text)[1:]:
        if block.strip().isdigit():
            pot = int(block.strip())
            continue
        for line in block.split("\n|}")[0].splitlines():
            if "fbaicon" not in line:
                continue
            team = _wiki_team(line)
            if team:
                rows.append({"season": season_label, "team": team[0],
                             "country": association(team[1]), "pot": pot})
    return pd.DataFrame(rows, columns=["season", "team", "country", "pot"])


# --- validation ------------------------------------------------------------

def validate(
    fixtures: pd.DataFrame, pots: pd.DataFrame | None = None, *, season_label: str = ""
) -> list[str]:
    """Return every structural violation of the league-phase format, as strings."""
    label = season_label or (fixtures.season.iloc[0] if len(fixtures) else "?")
    problems: list[str] = []

    if len(fixtures) != 144:
        problems.append(f"{label}: {len(fixtures)} matches, expected 144")

    teams = sorted(set(fixtures.home) | set(fixtures.away))
    if len(teams) != 36:
        problems.append(f"{label}: {len(teams)} distinct teams, expected 36")

    country = {}
    for row in fixtures.itertuples():
        country.setdefault(row.home, association(row.home_country))
        country.setdefault(row.away, association(row.away_country))

    for team in teams:
        home = int((fixtures.home == team).sum())
        away = int((fixtures.away == team).sum())
        if (home, away) != (4, 4):
            problems.append(f"{label}: {team} plays {home} home / {away} away, expected 4/4")

    for row in fixtures.itertuples():
        if association(row.home_country) == association(row.away_country):
            problems.append(
                f"{label}: same-association pairing {row.home} v {row.away} "
                f"({association(row.home_country)})"
            )

    pairs = fixtures.apply(lambda r: tuple(sorted((r.home, r.away))), axis=1)
    for pair, count in pairs.value_counts().items():
        if count > 1:
            problems.append(f"{label}: {pair[0]} v {pair[1]} appears {count} times")

    if pots is not None and len(pots):
        problems.extend(_validate_pots(fixtures, pots, teams, country, label))
    return problems


def _pot_lookup(pots: pd.DataFrame, teams: list[str], country: dict[str, str],
                label: str) -> tuple[dict[str, int], list[str]]:
    """Join pots to fixture team names by exact match after normalisation."""
    problems: list[str] = []
    index: dict[tuple[str, str], int] = {}
    for row in pots.itertuples():
        index[(row.country, normalize_club_name(row.team))] = row.pot
    lookup: dict[str, int] = {}
    for team in teams:
        key = (country[team], normalize_club_name(team))
        if key in index:
            lookup[team] = index[key]
        else:
            problems.append(
                f"{label}: no pot for {team} ({country[team]}); "
                f"normalised {normalize_club_name(team)!r} not in the pot table"
            )
    return lookup, problems


def _validate_pots(fixtures, pots, teams, country, label) -> list[str]:
    lookup, problems = _pot_lookup(pots, teams, country, label)
    if problems:
        return problems

    sizes = pd.Series(list(lookup.values())).value_counts().to_dict()
    for pot in (1, 2, 3, 4):
        if sizes.get(pot) != 9:
            problems.append(f"{label}: pot {pot} has {sizes.get(pot, 0)} teams, expected 9")

    for team in teams:
        opponents = [
            (row.away, "H") if row.home == team else (row.home, "A")
            for row in fixtures.itertuples()
            if team in (row.home, row.away)
        ]
        by_pot: dict[int, list[str]] = {}
        for opponent, venue in opponents:
            by_pot.setdefault(lookup[opponent], []).append(venue)
        for pot in (1, 2, 3, 4):
            venues = by_pot.get(pot, [])
            if len(venues) != 2:
                problems.append(
                    f"{label}: {team} faces {len(venues)} team(s) from pot {pot}, expected 2"
                )
            elif sorted(venues) != ["A", "H"]:
                problems.append(
                    f"{label}: {team} faces pot {pot} {venues.count('H')} times at home "
                    f"and {venues.count('A')} away, expected one of each"
                )
    return problems


# --- bridging two sources' names by schedule -------------------------------

def bridge_names(
    source: pd.DataFrame, reference: pd.DataFrame
) -> tuple[dict[str, str], list[str]]:
    """Map ``source`` team names onto ``reference`` names using the schedule.

    Both frames describe the same 144 matches, so the correspondence is a fact
    about the data rather than a guess: seed it with exact matches after
    normalisation, then propagate through each team's (matchday, venue,
    opponent) fixtures until nothing new resolves. Anything left unresolved or
    ambiguous is reported, never guessed.
    """
    def country_of(frame: pd.DataFrame) -> dict[str, str]:
        out: dict[str, str] = {}
        for row in frame.itertuples():
            out.setdefault(row.home, association(row.home_country))
            out.setdefault(row.away, association(row.away_country))
        return out

    def fixtures_of(frame: pd.DataFrame) -> dict[str, set[tuple[int, str, str]]]:
        out: dict[str, set[tuple[int, str, str]]] = {}
        for row in frame.itertuples():
            out.setdefault(row.home, set()).add((row.matchday, "H", row.away))
            out.setdefault(row.away, set()).add((row.matchday, "A", row.home))
        return out

    src_country, ref_country = country_of(source), country_of(reference)
    src_fixtures, ref_fixtures = fixtures_of(source), fixtures_of(reference)

    ref_by_key: dict[tuple[str, str], list[str]] = {}
    for name, country in ref_country.items():
        ref_by_key.setdefault((country, normalize_club_name(name)), []).append(name)

    resolved: dict[str, str] = {}
    problems: list[str] = []
    for name, country in src_country.items():
        candidates = ref_by_key.get((country, normalize_club_name(name)), [])
        if len(candidates) == 1:
            resolved[name] = candidates[0]

    progress = True
    while progress:
        progress = False
        for name, country in src_country.items():
            if name in resolved:
                continue
            known = {
                (md, venue, resolved[opponent])
                for md, venue, opponent in src_fixtures[name]
                if opponent in resolved
            }
            if not known:
                continue
            taken = set(resolved.values())
            candidates = [
                ref_name
                for ref_name, ref_country_code in ref_country.items()
                if ref_country_code == country
                and ref_name not in taken
                and known <= ref_fixtures[ref_name]
            ]
            if len(candidates) == 1:
                resolved[name] = candidates[0]
                progress = True

    for name, country in sorted(src_country.items()):
        if name not in resolved:
            problems.append(f"could not bridge {name!r} ({country}) to the reference source")
    return resolved, problems
