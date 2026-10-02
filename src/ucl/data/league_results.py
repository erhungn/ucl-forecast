"""League-phase matches that have already been played.

The source is the season's own Wikipedia article, parsed by
:func:`ucl.data.wikipedia.parse_matches` - the parser checked at 342/342 against
openfootball - and cached with the same manifest discipline as everything else,
so the fetch timestamp and hash are part of a forecast's provenance.

Every played match must map to exactly one fixture. A match that maps to none,
or to two, means the schedule moved or the page changed shape, and that is
raised rather than absorbed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ucl.data import fixtures as fx
from ucl.data import names, wikipedia

__all__ = ["MappingError", "live_page", "load_played", "information"]

COLUMNS = ["matchday", "date", "home_id", "away_id", "hg", "ag", "status"]


class MappingError(RuntimeError):
    """Raised when played matches do not line up with the fixture list."""


def live_page(season_label: str) -> wikipedia.Page:
    for page in wikipedia.PAGES:
        if page.season == season_label and page.kind == "league_phase":
            return page
    raise KeyError(f"no league-phase page configured for {season_label}")


def load_played(
    season_label: str = "2026-27", *, cache_dir: Path | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Played league-phase matches, mapped onto the fixture list.

    ``status`` is "played" for a normal result and "awarded" for a forfeit: an
    awarded score counts in the standings but carries no footballing
    information, so the caller can fix it in the table while leaving it out of
    anything that learns from scorelines.
    """
    page = live_page(season_label)
    path, reason = wikipedia.fetch_page(page, cache_dir=cache_dir, refresh=refresh)
    if path is None:
        raise MappingError(f"could not fetch {page.title}: {reason}")

    parsed = wikipedia.parse_matches(
        path.read_text(encoding="utf-8"), season=season_label,
        competition="UCL", source_file=path.name,
    )
    decided = parsed[parsed.played | parsed.awarded].copy()

    lookup = names.alias_to_canonical("wikipedia")
    decided["home_id"] = decided.home.map(lookup).fillna(decided.home)
    decided["away_id"] = decided.away.map(lookup).fillna(decided.away)

    fixtures = pd.read_csv(fx.fixtures_path(season_label), parse_dates=["date"])
    fixtures["home_id"] = fixtures.home.map(lookup).fillna(fixtures.home)
    fixtures["away_id"] = fixtures.away.map(lookup).fillna(fixtures.away)
    index = {(r.home_id, r.away_id): r.matchday for r in fixtures.itertuples()}

    matchdays, unmapped = [], []
    for row in decided.itertuples():
        matchday = index.get((row.home_id, row.away_id))
        if matchday is None:
            unmapped.append(f"{row.home_id} v {row.away_id} on {row.date.date()}")
        matchdays.append(matchday)
    if unmapped:
        raise MappingError(
            f"{len(unmapped)} played match(es) map to no fixture: " + "; ".join(unmapped[:5])
        )

    decided["matchday"] = matchdays
    pairs = list(zip(decided.home_id, decided.away_id))
    duplicated = {p for p in pairs if pairs.count(p) > 1}
    if duplicated:
        raise MappingError(f"fixture(s) played more than once: {sorted(duplicated)[:5]}")

    decided["status"] = decided.awarded.map({True: "awarded", False: "played"})
    decided["hg"] = decided.hg.astype(int)
    decided["ag"] = decided.ag.astype(int)
    return decided[COLUMNS].sort_values(["matchday", "date"]).reset_index(drop=True)


def completed_matchdays(played: pd.DataFrame, fixtures: pd.DataFrame) -> list[int]:
    """Matchdays whose every fixture has a result."""
    expected = fixtures.groupby("matchday").size()
    actual = played.groupby("matchday").size()
    return sorted(md for md in expected.index if actual.get(md, 0) == expected[md])


def information(season_label: str = "2026-27", as_of=None) -> pd.DataFrame:
    """What every source knows, and what an as-of date actually uses.

    ``cached_to`` is the last match in the cache; ``used_to`` is the last one
    before ``as_of``. They differ whenever the cache runs past the information
    set, which is exactly the case a forecast must not misrepresent.
    """
    from ucl.data import european, football_data
    from ucl.data.manifest import read_manifest

    horizon = pd.Timestamp("2030-01-01").date()
    limit = pd.Timestamp(as_of) if as_of is not None else None

    def used_to(frame):
        if limit is None:
            return frame.date.max().date()
        before = frame[frame.date < limit]
        return before.date.max().date() if len(before) else None

    rows = []
    for directory in football_data.CACHE_DIRS:
        if not directory.is_dir():
            rows.append({"source": f"domestic/{directory.name}", "status": "absent",
                         "files": 0, "cached_to": None, "used_to": None, "fetched": None})
            continue
        frame = football_data.load_results(horizon, cache_dirs=(directory,))
        manifest = read_manifest(directory)
        rows.append({
            "source": f"domestic/{directory.name}", "status": "present",
            "files": len(manifest), "cached_to": frame.date.max().date(),
            "used_to": used_to(frame),
            "fetched": max(e.downloaded_at_utc for e in manifest.values())[:19],
        })
    euro = european.load_results(horizon)
    for source, group in euro.groupby("source"):
        rows.append({"source": f"european/{source}", "status": "present",
                     "files": group.source_file.nunique(),
                     "cached_to": group.date.max().date(), "used_to": used_to(group),
                     "fetched": None})

    # The squad-value prior does not move with the as-of date: the source has
    # been paused since 2026-07-06, so ``used_to`` is the season's frozen as-of
    # whatever date the forecast is run for. A weekly update refreshes the
    # ratings and not this.
    from ucl.data import transfermarkt

    if transfermarkt.CACHE_DIR.is_dir():
        valuations, appearances = transfermarkt._tables()
        rows.append({
            "source": "squad_value/transfermarkt",
            "status": "frozen (source paused)",
            "files": len(read_manifest(transfermarkt.CACHE_DIR)),
            "cached_to": max(valuations.date.max(), appearances.date.max()).date(),
            "used_to": transfermarkt.as_of_for_season(season_label),
            "fetched": None,
        })
    return pd.DataFrame(rows)
