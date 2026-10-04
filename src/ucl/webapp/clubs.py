"""Three-letter codes and kit colours for the 36 league-phase clubs.

Presentation only: nothing here reaches a model, and no crests or logos are
used anywhere in the app. The codes are hand-written because the obvious
algorithm collides (Dortmund and Bodo/Glimt both reduce to ``BOD``) and reads
badly for the rest - ``MAC`` and ``MAU`` for the Manchester clubs, ``REM`` and
``REB`` for the Real ones.

Each club declares a first and second kit colour. The badge uses the first as
its ground, but the second is used for the text only when it is actually
legible on it; :func:`badge` falls back to ink or white otherwise, because an
unreadable badge is worse than an off-brand one.

Many of these clubs play in near-identical reds, so two badges in one card can
look the same at a glance. :func:`badge_pair_styles` detects that and renders
the away badge as an outline instead, which stays distinct whatever the hue.
"""

from __future__ import annotations

from ucl.webapp.palette import (
    MIN_CONTRAST, SURFACE, colour_distance, enforce_contrast, readable_text,
)

__all__ = [
    "CODES", "COLOURS", "code", "badge", "outline", "badge_pair_styles",
    "unknown", "MIN_DISTANCE",
]

#: Below this CIE76 distance two badge grounds are treated as the same colour.
MIN_DISTANCE = 25.0

#: club -> three-letter code. BVB for Borussia Dortmund, by request.
CODES: dict[str, str] = {
    "AEK Athens": "AEK",
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Atletico Madrid": "ATM",
    "Barcelona": "BAR",
    "Bayern Munich": "BAY",
    "Bodo/Glimt": "BOD",
    "Borussia Dortmund": "BVB",
    "Club Brugge": "BRU",
    "Como": "COM",
    "Fenerbahce": "FEN",
    "Feyenoord": "FEY",
    "Galatasaray": "GAL",
    "Inter": "INT",
    "LASK": "LSK",
    "Lens": "LEN",
    "Lille": "LIL",
    "Liverpool": "LIV",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Napoli": "NAP",
    "PSV Eindhoven": "PSV",
    "Paris Saint-Germain": "PSG",
    "Porto": "POR",
    "RB Leipzig": "RBL",
    "Real Betis": "BET",
    "Real Madrid": "RMA",
    "Roma": "ROM",
    "Sabah": "SAB",
    "Shakhtar Donetsk": "SHA",
    "Slavia Praha": "SLA",
    "Slovan Bratislava": "SLB",
    "Sporting CP": "SPO",
    "VfB Stuttgart": "STU",
    "Viking": "VIK",
    "Villarreal": "VIL",
}

#: club -> (first kit colour, second kit colour).
COLOURS: dict[str, tuple[str, str]] = {
    "AEK Athens": ("#f3d417", "#111111"),
    "Arsenal": ("#ef0107", "#ffffff"),
    "Aston Villa": ("#670e36", "#95bfe5"),
    "Atletico Madrid": ("#cb3524", "#ffffff"),
    "Barcelona": ("#004d98", "#edbb00"),
    "Bayern Munich": ("#dc052d", "#ffffff"),
    "Bodo/Glimt": ("#ffe500", "#111111"),
    "Borussia Dortmund": ("#fde100", "#111111"),
    "Club Brugge": ("#005ca9", "#111111"),
    "Como": ("#0b4ea2", "#ffffff"),
    "Fenerbahce": ("#0b1f4b", "#ffed00"),
    "Feyenoord": ("#cc0000", "#ffffff"),
    "Galatasaray": ("#a90432", "#fdb912"),
    "Inter": ("#0b1f8f", "#ffffff"),
    "LASK": ("#1a1a1a", "#ffffff"),
    "Lens": ("#ffe500", "#e1001a"),
    "Lille": ("#e01e13", "#0c2340"),
    "Liverpool": ("#c8102e", "#ffffff"),
    "Manchester City": ("#6cabdd", "#0b1220"),
    "Manchester United": ("#da291c", "#fbe122"),
    "Napoli": ("#12a0d7", "#ffffff"),
    "PSV Eindhoven": ("#ed1c24", "#ffffff"),
    "Paris Saint-Germain": ("#004170", "#da291c"),
    "Porto": ("#00428c", "#ffffff"),
    "RB Leipzig": ("#dd0741", "#ffffff"),
    "Real Betis": ("#00954c", "#ffffff"),
    "Real Madrid": ("#f0f0f0", "#00529f"),
    "Roma": ("#8e1f2f", "#f0bc42"),
    "Sabah": ("#00a3e0", "#ffffff"),
    "Shakhtar Donetsk": ("#ff6600", "#111111"),
    "Slavia Praha": ("#d7141a", "#ffffff"),
    "Slovan Bratislava": ("#0f52ba", "#ffffff"),
    "Sporting CP": ("#008057", "#ffffff"),
    "VfB Stuttgart": ("#e32219", "#ffffff"),
    "Viking": ("#002f6c", "#ffffff"),
    "Villarreal": ("#ffe667", "#005187"),
}

#: For a club the app has never heard of - a schedule change, a renamed club.
unknown = ("#243041", "#e5e7eb")


def code(team: str) -> str:
    """The club's three-letter code, or the first three letters upper-cased."""
    known = CODES.get(team)
    if known:
        return known
    letters = "".join(character for character in team if character.isalpha())
    return (letters[:3] or "???").upper()


def badge(team: str) -> tuple[str, str]:
    """``(ground, text)`` for the club's badge, guaranteed legible.

    The declared second colour is used when it reads on the first; otherwise
    ink or white, whichever has more contrast. Where neither clears 4.5:1 -
    true of a vivid mid-tone red - the ground itself deepens by the least
    amount that does.
    """
    declared, preferred = COLOURS.get(team, unknown)
    text = readable_text(declared, preferred)
    ground = enforce_contrast(declared, text)
    # Deepening the ground can flip which text colour is better; re-check once.
    return ground, readable_text(ground, text)


def outline(team: str) -> str:
    """The club's colour, legible as an outline on a card.

    A navy or near-black club colour vanishes against the card ground, so it
    lightens until it reads - otherwise the away badge in a same-colour
    fixture would be an invisible box.
    """
    ground, _ = COLOURS.get(team, unknown)
    return enforce_contrast(ground, SURFACE, minimum=MIN_CONTRAST)


def badge_pair_styles(home: str, away: str) -> tuple[str, str]:
    """Which style each badge takes in a card showing both clubs.

    ``("solid", "solid")`` normally; ``("solid", "outline")`` when the two
    grounds are too close to tell apart, which is common - a dozen of these
    clubs play in much the same red.
    """
    home_ground, _ = badge(home)
    away_ground, _ = badge(away)
    if colour_distance(home_ground, away_ground) < MIN_DISTANCE:
        return "solid", "outline"
    return "solid", "solid"
