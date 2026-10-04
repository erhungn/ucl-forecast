"""The cards, as HTML strings.

Everything here is a pure function returning markup, which the Streamlit layer
passes to ``st.html``. Three reasons it is built this way rather than from
Streamlit widgets:

* a phone. ``st.metric`` cards are about half a screen tall each, mostly empty;
  these are 70-90px and stack without ceremony.
* testability. A string can be asserted on directly, and ``AppTest`` can read
  the rendered body back, so the numbers on a bar are checked against the CSV.
* one element per list. Eighteen fixtures render as a single ``st.html`` call
  instead of eighteen, which keeps the spacing even and the page quick.

Styles are inline because that is the technique that survived the sanitiser
check on a real phone: ``st.markdown`` dropped the in-bar labels, and ``st.html``
with inline styles rendered exactly as drawn.
"""

from __future__ import annotations

from html import escape

from ucl.webapp import clubs
from ucl.webapp.palette import (
    AWAY, BORDER, DOWN, DRAW, FAINT, FLAT, HOME, MUTED, SURFACE,
    SURFACE_RAISED, TEXT, UP,
)

__all__ = [
    "LABEL_MINIMUM", "FONT", "badge_html", "badge_pair_html", "delta_html",
    "goal_chips_html",
    "stacked_bar_html", "narrow_outcomes", "bar_row_html", "fixture_cards_html",
    "club_cards_html", "compact_rows_html", "mover_rows_html", "card",
]

#: A label only fits inside a segment from about this share upwards; below it
#: the percentage moves to the line under the bar.
LABEL_MINIMUM = 0.12

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif"

_OUTCOMES = (("home", HOME, "H"), ("draw", DRAW, "D"), ("away", AWAY, "A"))


def card(body: str, *, raised: bool = False) -> str:
    """One bordered card. The grid that holds them is the caller's business."""
    ground = SURFACE_RAISED if raised else SURFACE
    return (
        f'<div style="background:{ground};border:1px solid {BORDER};'
        f'border-radius:10px;padding:11px 12px;margin:0 0 8px 0">{body}</div>'
    )


def badge_html(team: str, *, style: str = "solid", size: int = 12) -> str:
    """The club's three-letter code as a badge in its own colours.

    ``outline`` keeps the code readable while looking clearly unlike a solid
    badge, which is what makes two same-coloured clubs tellable apart.
    """
    label = escape(clubs.code(team))
    shared = (f"display:inline-block;min-width:38px;text-align:center;"
              f"padding:2px 6px;border-radius:5px;font:700 {size}px/1.35 {FONT};"
              f"letter-spacing:0.3px")
    if style == "outline":
        colour = clubs.outline(team)
        return (f'<span style="{shared};color:{colour};background:transparent;'
                f'border:1.5px solid {colour}">{label}</span>')
    ground, text = clubs.badge(team)
    return (f'<span style="{shared};color:{text};background:{ground};'
            f'border:1px solid rgba(255,255,255,0.14)">{label}</span>')


def badge_pair_html(home: str, away: str, *, size: int = 12) -> tuple[str, str]:
    """Both badges for one fixture, the away one outlined if they clash."""
    home_style, away_style = clubs.badge_pair_styles(home, away)
    return (badge_html(home, style=home_style, size=size),
            badge_html(away, style=away_style, size=size))


def delta_html(value: float | None, *, points: bool = True,
               size: int = 12) -> str:
    """An up/down marker. Colour never carries the meaning on its own."""
    if value is None or value != value:                     # None or NaN
        return ""
    if abs(value) < 0.0005:
        arrow, colour = "–", FLAT
    elif value > 0:
        arrow, colour = "▲", UP
    else:
        arrow, colour = "▼", DOWN
    shown = f"{abs(value) * 100:.1f}" if points else f"{abs(value):.2f}"
    text = f"{arrow} {shown}" if arrow != "–" else arrow
    return (f'<span style="color:{colour};font:600 {size}px/1.3 {FONT};'
            f'white-space:nowrap">{text}</span>')


def stacked_bar_html(home: float, draw: float, away: float, *,
                     height: int = 26) -> str:
    """One horizontal bar, home then draw then away, labelled in place."""
    total = home + draw + away
    shares = [home / total, draw / total, away / total] if total else [0.0, 0.0, 0.0]
    # Rounding each width independently leaves the bar a fraction short, so the
    # last segment takes whatever remains and the three always sum to 100%.
    widths = [round(share * 100, 4) for share in shares[:2]]
    widths.append(round(100.0 - sum(widths), 4) if total else 0.0)
    segments = ""
    for share, width, (_, colour, letter) in zip(shares, widths, _OUTCOMES):
        inside = f"{share * 100:.0f}% {letter}" if share >= LABEL_MINIMUM else ""
        segments += (
            f'<div style="width:{width:.4f}%;background:{colour};'
            f'color:#0b1220;font:700 12px/{height}px {FONT};text-align:center;'
            f'white-space:nowrap;overflow:hidden">{inside}</div>'
        )
    return (f'<div style="display:flex;gap:2px;height:{height}px;'
            f'border-radius:6px;overflow:hidden">{segments}</div>')


def narrow_outcomes(home: float, draw: float, away: float) -> str:
    """The outcomes too slim to label inside the bar, for the line below it."""
    total = home + draw + away
    shares = [home / total, draw / total, away / total] if total else [0, 0, 0]
    return " · ".join(
        f"{name} {share * 100:.1f}%"
        for share, (name, _, _) in zip(shares, _OUTCOMES)
        if share < LABEL_MINIMUM
    )


def bar_row_html(label: str, value: float, colour: str, *,
                 delta: float | None = None, width: int = 68) -> str:
    """A labelled single-colour bar: the title race's three rows."""
    marker = delta_html(delta) if delta is not None else ""
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin-top:5px">'
        f'<span style="color:{MUTED};font:500 11px/1.3 {FONT};'
        f'width:56px;flex:0 0 56px">{escape(label)}</span>'
        f'<span style="flex:1 1 auto;background:rgba(148,163,184,0.16);'
        f'height:9px;border-radius:5px;overflow:hidden;display:block">'
        f'<span style="display:block;width:{max(value, 0) * 100:.3f}%;'
        f'height:100%;background:{colour};border-radius:5px"></span></span>'
        f'<span style="color:{TEXT};font:700 12px/1.3 {FONT};width:48px;'
        f'flex:0 0 48px;text-align:right">{value * 100:.1f}%</span>'
        f'<span style="width:{width}px;flex:0 0 {width}px;text-align:right">'
        f'{marker}</span></div>'
    )


def goal_chips_html(row) -> str:
    """Expected goals and the two tail probabilities, as a line of chips.

    The modal scoreline is 1-0, 1-1 or 2-0 for most fixtures however lopsided
    they are, so on its own it hides the difference between a close game and a
    rout. These three numbers are what separate them, and they come from the
    same rates the bar does.

    Returns an empty string when the columns are absent - the frozen evaluation
    tables carry probabilities but not rates.
    """
    if row.get("xg_home") is None or row.get("xg_away") is None:
        return ""
    if row["xg_home"] != row["xg_home"]:                    # NaN
        return ""

    def chip(label: str, value: str) -> str:
        return (f'<span style="background:rgba(148,163,184,0.12);'
                f'border-radius:5px;padding:2px 7px;white-space:nowrap">'
                f'<span style="color:{FAINT}">{label}</span> '
                f'<span style="color:{TEXT};font-weight:700">{value}</span></span>')

    chips = [chip("xG", f'{float(row["xg_home"]):.2f}–{float(row["xg_away"]):.2f}')]
    if row.get("p_over_2_5") is not None:
        chips.append(chip("over 2.5", f'{float(row["p_over_2_5"]) * 100:.0f}%'))
    if row.get("favourite") and row.get("p_favourite_by_3") is not None:
        chips.append(chip(f'{escape(clubs.code(str(row["favourite"])))} by 3+',
                          f'{float(row["p_favourite_by_3"]) * 100:.0f}%'))
    return (f'<div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:7px;'
            f'font:500 11px/1.6 {FONT}">{"".join(chips)}</div>')


def _title_line(home: str, away: str, *, date: str = "") -> str:
    home_badge, away_badge = badge_pair_html(home, away)
    stamp = (f'<span style="color:{FAINT};font:500 11px/1.3 {FONT}">'
             f'{escape(date)}</span>' if date else "")
    return (
        f'<div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap;'
        f'margin-bottom:9px">{home_badge}'
        f'<span style="color:{TEXT};font:600 14px/1.3 {FONT}">{escape(home)}</span>'
        f'<span style="color:{FAINT};font:500 12px/1.3 {FONT}">v</span>'
        f'{away_badge}'
        f'<span style="color:{TEXT};font:600 14px/1.3 {FONT}">{escape(away)}</span>'
        f'<span style="margin-left:auto">{stamp}</span></div>'
    )


def fixture_cards_html(rows, *, show_result: bool = False) -> str:
    """One card per fixture: the bar, the pick, the modal scoreline.

    ``rows`` are mappings with ``home``, ``away``, ``p_home``, ``p_draw``,
    ``p_away`` and optionally ``date``, ``pick``, ``modal_score``, ``p_modal``,
    ``score`` and ``result``.
    """
    out = []
    for row in rows:
        home, away = row["home"], row["away"]
        probabilities = (float(row["p_home"]), float(row["p_draw"]),
                         float(row["p_away"]))
        footer = []
        if show_result and row.get("score"):
            outcome = str(row.get("result", ""))
            colour = {"home": HOME, "draw": DRAW, "away": AWAY}.get(outcome, MUTED)
            actual = {"home": home, "draw": "draw", "away": away}.get(outcome, outcome)
            footer.append(
                f'<span style="color:{colour};font:700 12px/1.3 {FONT}">'
                f'{escape(str(row["score"]))} {escape(actual)}</span>')
            if row.get("p_actual") is not None:
                footer.append(f'forecast {float(row["p_actual"]) * 100:.1f}%')
        if row.get("pick"):
            footer.append(f'pick {escape(str(row["pick"]))}')
        if row.get("modal_score"):
            footer.append(
                f'likely {escape(str(row["modal_score"]))} '
                f'({float(row.get("p_modal", 0)) * 100:.1f}%)')
        narrow = narrow_outcomes(*probabilities)
        if narrow:
            footer.append(narrow)
        body = (
            _title_line(home, away, date=str(row.get("date", "")))
            + stacked_bar_html(*probabilities)
            + (f'<div style="color:{MUTED};font:500 11.5px/1.45 {FONT};'
               f'margin-top:8px">{" · ".join(footer)}</div>' if footer else "")
            + goal_chips_html(row)
        )
        out.append(card(body))
    return "".join(out) or _empty("No fixtures to show.")


def club_cards_html(rows) -> str:
    """One card per club: three bars with their change since last matchday."""
    out = []
    for row in rows:
        team = row["team"]
        head = (
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<span style="color:{FAINT};font:700 12px/1.3 {FONT};'
            f'min-width:20px">{int(row["position"])}</span>'
            f'{badge_html(team)}'
            f'<span style="color:{TEXT};font:600 14px/1.3 {FONT}">'
            f'{escape(team)}</span>'
            f'<span style="margin-left:auto">'
            f'{delta_html(row.get("d_p_champion"))}</span></div>'
        )
        bars = (
            bar_row_html("top 8", float(row["p_top8"]), HOME,
                         delta=row.get("d_p_top8"))
            + bar_row_html("champion", float(row["p_champion"]), "#a78bfa",
                           delta=row.get("d_p_champion"))
            + bar_row_html("out", float(row["p_eliminated"]), AWAY,
                           delta=row.get("d_p_eliminated"))
        )
        out.append(card(head + bars))
    return "".join(out) or _empty("No clubs to show.")


def compact_rows_html(rows) -> str:
    """The home screen's favourites: one tight line each, no wasted height."""
    out = []
    for index, row in enumerate(rows, 1):
        team = row["team"]
        value = float(row["value"])
        out.append(
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'padding:7px 0;border-bottom:1px solid {BORDER}">'
            f'<span style="color:{FAINT};font:700 12px/1.3 {FONT};'
            f'min-width:16px">{index}</span>'
            f'{badge_html(team)}'
            f'<span style="color:{TEXT};font:600 13.5px/1.3 {FONT};'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
            f'{escape(team)}</span>'
            f'<span style="flex:1 1 auto;min-width:24px;background:'
            f'rgba(148,163,184,0.16);height:8px;border-radius:4px;'
            f'overflow:hidden;display:block;margin:0 2px">'
            f'<span style="display:block;width:{value * 100:.3f}%;height:100%;'
            f'background:{row.get("colour", HOME)};border-radius:4px"></span>'
            f'</span>'
            f'<span style="color:{TEXT};font:700 13px/1.3 {FONT};'
            f'width:46px;text-align:right">{value * 100:.1f}%</span>'
            f'<span style="width:62px;text-align:right">'
            f'{delta_html(row.get("delta"))}</span></div>'
        )
    return "".join(out) or _empty("Nothing published yet.")


def mover_rows_html(rows) -> str:
    """Biggest movers: from, to, and the change, on one line each."""
    out = []
    for row in rows:
        team = row["team"]
        out.append(
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'padding:7px 0;border-bottom:1px solid {BORDER}">'
            f'{badge_html(team)}'
            f'<span style="color:{TEXT};font:600 13.5px/1.3 {FONT};'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
            f'flex:1 1 auto">{escape(team)}</span>'
            f'<span style="color:{MUTED};font:500 12px/1.3 {FONT};'
            f'white-space:nowrap">{escape(str(row["label"]))} '
            f'{float(row["before"]) * 100:.0f}→{float(row["after"]) * 100:.0f}'
            f'</span>'
            f'<span style="width:62px;text-align:right">'
            f'{delta_html(row.get("delta"))}</span></div>'
        )
    return "".join(out) or _empty("No movement to show yet.")


def _empty(message: str) -> str:
    return (f'<div style="color:{MUTED};font:500 13px/1.4 {FONT};'
            f'padding:10px 0">{escape(message)}</div>')
