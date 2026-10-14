"""European club competition results parsed from Wikipedia wikitext.

openfootball does not publish the 2025-26 Europa and Conference League main
phases, nor anything for 2026-27, and our own Elo needs those matches to link
national leagues. Wikipedia's competition articles carry the same results in
``{{#invoke:Football box|main}}`` blocks, which parse deterministically::

    |date         = {{Start date|2024|10|2|df=y}}
    |team1        = [[İstanbul Başakşehir F.K.|İstanbul Başakşehir]] {{#invoke:flag|fbaicon|TUR}}
    |score        = 1-2
    |team2        = {{#invoke:flag|fbaicon|AUT}} [[SK Rapid Wien|Rapid Wien]]
    |aet          = yes
    |penaltyscore = 3-2

One semantic difference from openfootball matters: when ``aet`` is set, Wikipedia's
``score`` is the score *after* extra time, not at 90 minutes. The 90-minute score
is recovered by counting goals whose minute is 90 or earlier, and that derivation
is checked against openfootball before it is trusted — see scripts/validate_wikipedia.py.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ucl.config import DATA_RAW
from ucl.data.cutoff import filter_to_cutoff
from ucl.data.manifest import entry_for, read_manifest, verify_cache, write_manifest

__all__ = [
    "CACHE_DIR", "PAGES", "Page", "USER_AGENT",
    "fetch_page", "fetch_all", "parse_matches", "load_results", "COLUMNS",
    "normalise_dashes", "find_score",
]

CACHE_DIR = DATA_RAW / "wikipedia"
RAW_URL = "https://en.wikipedia.org/w/index.php?title={title}&action=raw"
# Wikimedia rejects the default urllib agent with 403.
USER_AGENT = "ucl-forecast/0.1 (research project; contact via repository)"

COLUMNS = [
    "date", "home", "away", "home_country", "away_country", "country", "league",
    "season", "competition", "stage", "source_file", "hg", "ag", "ht_hg", "ht_ag",
    "aet", "et_hg", "et_ag", "pens_h", "pens_a", "awarded", "played",
]


@dataclass(frozen=True)
class Page:
    season: str
    competition: str
    kind: str
    title: str
    #: "gap" pages are real data; "validation" pages duplicate openfootball and
    #: exist only so the parser can be checked against it.
    role: str = "gap"

    @property
    def file_name(self) -> str:
        slug = self.title.replace("–", "-").replace(" ", "_")
        return f"{slug}.wikitext"


def _pages() -> list[Page]:
    pages: list[Page] = []
    # Validation: seasons openfootball also covers, so the parser can be checked.
    for season, dash in (("2024-25", "2024–25"),):
        for competition, name in (("UEL", "Europa League"), ("UECL", "Conference League")):
            pages.append(Page(season, competition, "league_phase",
                              f"{dash} UEFA {name} league phase", "validation"))
            pages.append(Page(season, competition, "knockout",
                              f"{dash} UEFA {name} knockout phase", "validation"))
            pages.append(Page(season, competition, "final",
                              f"2025 UEFA {name} final", "validation"))
    # Gap 1: the 2025-26 Europa and Conference League main phases. The final is
    # on neither the league-phase nor the knockout article; it has its own page.
    for competition, name in (("UEL", "Europa League"), ("UECL", "Conference League")):
        pages.append(Page("2025-26", competition, "league_phase",
                          f"2025–26 UEFA {name} league phase"))
        pages.append(Page("2025-26", competition, "knockout",
                          f"2025–26 UEFA {name} knockout phase"))
        pages.append(Page("2025-26", competition, "final", f"2026 UEFA {name} final"))
    # The 2026-27 league phase itself, as a results source: it is the only way
    # an in-season update sees matches that have already been played. Before the
    # 2026-09-07 cutoff it contributes nothing, so the pre-tournament forecast is
    # untouched.
    pages.append(Page("2026-27", "UCL", "league_phase",
                      "2026–27 UEFA Champions League league phase"))

    # Gap 3: the Europa League before 2020-21, which openfootball does not
    # publish. These seasons used the group-stage format, so the pages are
    # "group stage" rather than "league phase".
    for dash, final_year in (("2016–17", 2017), ("2017–18", 2018),
                             ("2018–19", 2019), ("2019–20", 2020)):
        season = dash.replace("–", "-")
        pages.append(Page(season, "UEL", "league_phase",
                          f"{dash} UEFA Europa League group stage"))
        pages.append(Page(season, "UEL", "knockout",
                          f"{dash} UEFA Europa League knockout phase"))
        pages.append(Page(season, "UEL", "final",
                          f"{final_year} UEFA Europa League final"))

    # Gap 2: 2026-27 qualifying and play-offs, played before the cutoff.
    for competition, name in (("UCL", "Champions League"), ("UEL", "Europa League")):
        pages.append(Page("2026-27", competition, "qualifying",
                          f"2026–27 UEFA {name} qualifying"))
    # Conference League qualifying has too many matches for one article, so the
    # index page carries only summaries and the boxes live on two sub-pages.
    for suffix in ("first and second round matches", "third and play-off round matches"):
        pages.append(Page("2026-27", "UECL", "qualifying",
                          f"2026–27 UEFA Conference League qualifying ({suffix})"))
    return pages


PAGES: list[Page] = _pages()


# --- fetching --------------------------------------------------------------

def _raw(title: str, timeout: float) -> str:
    url = RAW_URL.format(title=urllib.parse.quote(title.replace(" ", "_")))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


_REDIRECT = re.compile(r"^\s*#REDIRECT\s*\[\[([^\]]+)\]\]", re.IGNORECASE)


def fetch_page(
    page: Page, *, cache_dir: Path | None = None, timeout: float = 25.0,
    refresh: bool = False,
) -> tuple[Path | None, str]:
    """Cache one article's wikitext, following a single redirect.

    Returns ``(path, "")`` or ``(None, reason)``; an unreachable page is
    reported rather than mistaken for an empty article.
    """
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / page.file_name
    if target.exists() and not refresh:
        return target, ""
    try:
        text = _raw(page.title, timeout)
        redirect = _REDIRECT.match(text)
        if redirect:
            text = _raw(redirect.group(1), timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return None, f"{type(error).__name__}: {error}"
    if not re.search(r"football box", text, re.IGNORECASE):
        return None, f"no Football box blocks in {page.title!r} ({len(text)} bytes)"
    target.write_text(text, encoding="utf-8")
    existing = []
    if (cache_dir / "manifest.json").exists():
        existing = [e for e in read_manifest(cache_dir).values() if e.file != target.name]
    write_manifest(
        cache_dir,
        [*existing, entry_for(target, RAW_URL.format(
            title=urllib.parse.quote(page.title.replace(" ", "_"))))],
    )
    return target, ""


def fetch_all(
    pages: list[Page] | None = None, *, cache_dir: Path | None = None, timeout: float = 25.0,
    refresh: bool = False,
) -> list[str]:
    """Fetch every page; returns the failures so a missing one is visible."""
    failed = []
    for page in pages or PAGES:
        _, reason = fetch_page(page, cache_dir=cache_dir, timeout=timeout, refresh=refresh)
        if reason:
            failed.append(f"{page.title}: {reason}")
    return failed


# --- parsing ---------------------------------------------------------------

_HEADING = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", re.MULTILINE)
_WIKI_LINK = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")
# Most boxes use {{fbaicon|GER}}, but editors sometimes write {{flagicon|GER}};
# both spellings appear in the 2025-26 Europa League semi-finals.
_FBAICON = re.compile(r"(?:fba|flag)icon\|([A-Z]{3})\b")
_START_DATE = re.compile(r"\{\{Start date\|(\d{4})\|(\d{1,2})\|(\d{1,2})")
#: Every hyphen, dash and minus a score field might be written with, each
#: normalised to a plain hyphen before matching.
#:
#: Unicode category Pd covers the hyphens and dashes, but MINUS SIGN is
#: category **Sm** and so is not in Pd - and that is exactly the character an
#: editor used for matchday 2 of 2026-27, where a pattern accepting only the en
#: dash and the hyphen read all nine played matches as unplayed.
#:
#: Written out rather than derived at import because scanning Unicode for Pd
#: costs ~80ms on every import of this module, and the model pipeline imports
#: it. ``test_the_dash_table_covers_every_unicode_dash`` does that scan
#: instead, so a character added by a future Unicode release fails a test
#: rather than a season.
_DASHES = (
    "-"      # HYPHEN-MINUS (Pd)
    "֊"      # ARMENIAN HYPHEN (Pd)
    "־"      # HEBREW PUNCTUATION MAQAF (Pd)
    "᐀"      # CANADIAN SYLLABICS HYPHEN (Pd)
    "᠆"      # MONGOLIAN TODO SOFT HYPHEN (Pd)
    "‐"      # HYPHEN (Pd)
    "‑"      # NON-BREAKING HYPHEN (Pd)
    "‒"      # FIGURE DASH (Pd)
    "–"      # EN DASH (Pd)
    "—"      # EM DASH (Pd)
    "―"      # HORIZONTAL BAR (Pd)
    "⸗"      # DOUBLE OBLIQUE HYPHEN (Pd)
    "⸚"      # HYPHEN WITH DIAERESIS (Pd)
    "⸺"      # TWO-EM DASH (Pd)
    "⸻"      # THREE-EM DASH (Pd)
    "⹀"      # DOUBLE HYPHEN (Pd)
    "⹝"      # OBLIQUE HYPHEN (Pd)
    "〜"      # WAVE DASH (Pd)
    "〰"      # WAVY DASH (Pd)
    "゠"      # KATAKANA-HIRAGANA DOUBLE HYPHEN (Pd)
    "︱"      # PRESENTATION FORM FOR VERTICAL EM DASH (Pd)
    "︲"      # PRESENTATION FORM FOR VERTICAL EN DASH (Pd)
    "﹘"      # SMALL EM DASH (Pd)
    "﹣"      # SMALL HYPHEN-MINUS (Pd)
    "－"      # FULLWIDTH HYPHEN-MINUS (Pd)
    "\U00010ead"  # YEZIDI HYPHENATION MARK (Pd)
    "−"      # MINUS SIGN (Sm) - the 2026-27 matchday 2 character
    "⁻"      # SUPERSCRIPT MINUS (Sm)
    "₋"      # SUBSCRIPT MINUS (Sm)
    "˗"      # MODIFIER LETTER MINUS SIGN (Sk)
    "➖"      # HEAVY MINUS SIGN (So)
    "⧿"      # MINY (Sm)
)

_DASH_TABLE = {ord(character): "-" for character in _DASHES}

# Applied to the already-normalised string, so one spelling covers all 32.
_SCORE = re.compile(r"(\d+)\s*-\s*(\d+)")
# The same box appears in three spellings: "{{#invoke:Football box|main" on the
# league-phase pages, "{{Football box" on the knockout pages and
# "{{#invoke:football box|main" on the qualifying pages.
_BOX = re.compile(r"\{\{(?:#invoke:)?\s*football box(?:\s*\|\s*main)?", re.IGNORECASE)
_GOAL = re.compile(r"\{\{goal\|([^}]*)\}\}")
_MINUTE = re.compile(r"(\d{1,3})(?:\+\d+)?")


def normalise_dashes(text: str) -> str:
    """Every dash, hyphen and minus variant as a plain hyphen.

    Score fields are written with whichever dash the editor's keyboard or
    copy-paste produced; normalising first means the score pattern has one
    spelling to match instead of a character class that someone has to
    remember to extend.
    """
    return text.translate(_DASH_TABLE)


def find_score(field: str) -> re.Match[str] | None:
    """The ``h-a`` score in a field, whatever dash it was written with."""
    return _SCORE.search(normalise_dashes(field))


def _strip_markup(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", text, flags=re.DOTALL)
    return re.sub(r"\x27{2,3}", "", text).strip()


def parse_team(field: str) -> tuple[str, str] | None:
    """Read a team field, linked or not.

    Second legs name the club as plain text because the first leg already
    linked it: ``team1 = Viktoria Plzen {{fbaicon|CZE}}``. Requiring a wiki
    link silently drops every second leg.
    """
    country = _FBAICON.search(field)
    if not country:
        return None
    link = _WIKI_LINK.search(field)
    name = link.group(1).strip() if link else _strip_markup(field)
    return (name, country.group(1)) if name else None


def _goals_by_ninety(field: str) -> int | None:
    """Count goals scored at 90 minutes or earlier, from the goal templates."""
    if field is None:
        return None
    total = 0
    for args in _GOAL.findall(field):
        for part in args.split("|"):
            match = _MINUTE.match(part.strip())
            if match and int(match.group(1)) <= 90:
                total += 1
    return total


def _block_at(text: str, start: int) -> str:
    """The whole ``{{...}}`` block beginning at ``start``, braces balanced.

    Naive scanning to the first "\n}}" truncates boxes that nest a template such
    as {{plainlist|...}}, which silently drops half the matches on the knockout
    and qualifying pages.
    """
    depth, index = 0, start
    while index < len(text) - 1:
        pair = text[index:index + 2]
        if pair == "{{":
            depth += 1
            index += 2
        elif pair == "}}":
            depth -= 1
            index += 2
            if depth == 0:
                # Exclude the closing braces: otherwise a field that is the last
                # line of the box swallows them into its value.
                return text[start:index - 2]
        else:
            index += 1
    return text[start:]


def _fields(block: str) -> dict[str, str]:
    """Top-level ``|key = value`` fields, ignoring those inside nested templates."""
    out: dict[str, str] = {}
    key = None
    depth = 0
    for line in block.splitlines():
        at_top = depth <= 1
        if at_top and line.startswith("|"):
            key, _, value = line[1:].partition("=")
            key = key.strip()
            out[key] = value.strip()
        elif key is not None:
            out[key] = f"{out[key]}\n{line}"
        depth += line.count("{{") - line.count("}}")
    return out


def parse_matches(
    text: str, *, season: str, competition: str, source_file: str = "",
    require_scores_before: date | None = None,
    tolerate_missing: bool = False,
) -> pd.DataFrame:
    """Parse every Football box in an article into match rows.

    ``require_scores_before`` turns the quiet failure into a loud one. A box
    dated before it that yields no score is a parse failure, not an unplayed
    fixture - the match has happened - so it raises with the raw score text
    rather than returning ``played=False``. That is the guard the 2026-27
    matchday 2 dash change needed: nine played matches were silently dropped
    and only a hand-count noticed. Awarded matches are exempt, having no
    numeric score by definition.

    ``tolerate_missing`` splits the two ways that can happen. A score field the
    parser **could not read** is always a bug in the parser, whatever the age
    of the page. A field that is simply **empty** means the cached copy predates
    the result, which is a legitimate state for a deliberately stale read and a
    failure only when the page was just fetched. Set it when reading a cache
    that is allowed to lag.
    """
    headings = [(m.start(), len(m.group(1)), m.group(2)) for m in _HEADING.finditer(text)]

    def stage_at(position: int) -> str:
        path: dict[int, str] = {}
        for start, level, title in headings:
            if start > position:
                break
            path[level] = title
            for deeper in [k for k in path if k > level]:
                del path[deeper]
        parts = [path[level] for level in sorted(path) if level >= 2]
        return ", ".join(p for p in parts if p not in {"Matches", "Summary"})

    rows = []
    unscored: list[str] = []
    for match in _BOX.finditer(text):
        fields = _fields(_block_at(text, match.start()))
        when = _START_DATE.search(fields.get("date", ""))
        home = parse_team(fields.get("team1", ""))
        away = parse_team(fields.get("team2", ""))
        if not (when and home and away):
            continue
        raw_score = fields.get("score", "")
        score = find_score(raw_score)
        aet = fields.get("aet", "").strip().lower() in {"y", "yes", "true", "1"}
        pens = find_score(fields.get("penaltyscore", ""))
        awarded = "awarded" in raw_score.lower()

        played_on = date(int(when.group(1)), int(when.group(2)), int(when.group(3)))
        if (require_scores_before is not None and score is None and not awarded
                and played_on < require_scores_before
                and (raw_score.strip() or not tolerate_missing)):
            unscored.append(
                f"{played_on} {home[0]} v {away[0]} "
                f"[{stage_at(match.start()) or competition}] "
                f"raw score {raw_score.strip()!r}")

        hg = ag = et_hg = et_ag = None
        if score:
            if aet:
                # Wikipedia's score includes extra time; recover 90 minutes from
                # the goal minutes.
                et_hg, et_ag = int(score.group(1)), int(score.group(2))
                hg = _goals_by_ninety(fields.get("goals1"))
                ag = _goals_by_ninety(fields.get("goals2"))
            else:
                hg, ag = int(score.group(1)), int(score.group(2))

        rows.append(
            {
                "date": pd.Timestamp(int(when.group(1)), int(when.group(2)), int(when.group(3))),
                "home": home[0], "away": away[0],
                "home_country": home[1], "away_country": away[1],
                "country": "EUR", "league": competition,
                "season": season, "competition": competition,
                "stage": stage_at(match.start()), "source_file": source_file,
                "hg": hg, "ag": ag, "ht_hg": None, "ht_ag": None,
                "aet": aet, "et_hg": et_hg, "et_ag": et_ag,
                "pens_h": int(pens.group(1)) if pens else None,
                "pens_a": int(pens.group(2)) if pens else None,
                "awarded": awarded,
                "played": score is not None and not awarded,
            }
        )

    if unscored:
        raise ValueError(
            f"{len(unscored)} match(es) in {source_file or competition} are dated "
            f"before {require_scores_before} but parsed no score, so they would "
            f"have been treated as unplayed:\n  "
            + "\n  ".join(unscored)
            + "\n\nAn empty raw score means the cached page predates the result: "
              "re-fetch it with refresh=True. A non-empty one the parser could "
              "not read means the page's score format changed: widen "
              "ucl.data.wikipedia._DASHES or _SCORE."
        )

    frame = pd.DataFrame(rows, columns=COLUMNS)
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
    pages: list[Page] | None = None,
    role: str | None = "gap",
    played_only: bool = True,
    verify: bool = True,
    require_scores: bool = True,
    tolerate_missing: bool = True,
    today: date | None = None,
) -> pd.DataFrame:
    """Load cached Wikipedia European results dated strictly before ``cutoff``.

    By default only the gap pages are returned: the validation pages duplicate
    matches openfootball already provides, and loading both would double-count.

    ``require_scores`` refuses to return a frame in which a match this load
    would have used has happened but carries a score the parser could not
    read. The bar is ``min(cutoff, today)``, not ``today`` alone: a fixture
    dated after the cutoff is filtered out anyway, so demanding a score for it
    would make every historical load fail on a cache that merely predates the
    current matchday.

    ``tolerate_missing`` defaults to true here, unlike on a freshly fetched
    page: this function reads a cache that is **allowed** to lag, and it cannot
    tell a page that predates a result from one that is missing it. An empty
    score is therefore left to the staleness report, while a score that is
    present and unreadable still raises, because that is a parser bug at any
    age.
    """
    cache_dir = Path(cache_dir or CACHE_DIR)
    candidates = [p for p in (pages or PAGES) if role is None or p.role == role]
    chosen = [p for p in candidates if (cache_dir / p.file_name).exists()]
    if not chosen:
        raise FileNotFoundError(f"no cached Wikipedia pages in {cache_dir}")
    if verify:
        verify_cache(cache_dir, files=[p.file_name for p in chosen])
    limit = min(cutoff, today or date.today()) if require_scores else None
    frames = [
        parse_matches(
            (cache_dir / page.file_name).read_text(encoding="utf-8"),
            season=page.season, competition=page.competition, source_file=page.file_name,
            require_scores_before=limit, tolerate_missing=tolerate_missing,
        ).assign(phase="qualifying" if page.kind == "qualifying" else "main")
        for page in chosen
    ]
    combined = pd.concat(frames, ignore_index=True)
    if played_only:
        combined = combined[combined.played].copy()
    combined = filter_to_cutoff(
        combined, cutoff, source=f"wikipedia cache ({len(chosen)} pages)"
    )
    return combined.sort_values(["date", "competition", "home"]).reset_index(drop=True)
