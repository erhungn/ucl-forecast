"""The app's colours, and the arithmetic that keeps them legible.

One definition. ``.streamlit/config.toml`` carries the same values for the
chrome Streamlit draws itself, and a test asserts the two agree, so the theme
and the hand-drawn cards cannot drift apart.

Home/draw/away deliberately avoid red against green: teal, slate and amber stay
distinguishable under the common forms of colour blindness. The up/down markers
do use green and red, but never alone - they always carry an arrow as well.
"""

from __future__ import annotations

__all__ = [
    "INK", "SURFACE", "SURFACE_RAISED", "BORDER", "TEXT", "MUTED", "FAINT",
    "HOME", "DRAW", "AWAY", "UP", "DOWN", "FLAT", "ACCENT", "CATEGORICAL",
    "FRESH", "AGEING", "STALE", "MIN_CONTRAST", "relative_luminance",
    "contrast_ratio", "readable_text", "enforce_contrast", "to_lab",
    "colour_distance",
]

#: WCAG AA for normal-size text. The badges set 12-13px bold, which is well
#: below the large-text exemption, so 4.5:1 is the bar that actually applies.
MIN_CONTRAST = 4.5

#: Page and card grounds, dark by design (see .streamlit/config.toml).
INK = "#0b1220"
SURFACE = "#111827"
SURFACE_RAISED = "#1a2232"
BORDER = "#243041"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"
FAINT = "#64748b"

#: The three outcomes, in the order the ranked probability score assumes.
HOME = "#2dd4bf"
DRAW = "#94a3b8"
AWAY = "#fb923c"

#: Movement. Always rendered with an arrow, never colour alone.
UP = "#4ade80"
DOWN = "#f87171"
FLAT = "#94a3b8"

ACCENT = "#2dd4bf"

#: Freshness of the published forecast.
FRESH = "#4ade80"
AGEING = "#fbbf24"
STALE = "#f87171"

#: Series colours for charts, also set as theme.chartCategoricalColors so that
#: Altair and Streamlit's own charts agree. Eight is the practical limit for
#: telling lines apart, which is why the movement page caps its selection.
CATEGORICAL = [
    "#2dd4bf", "#fb923c", "#a78bfa", "#60a5fa",
    "#f472b6", "#facc15", "#34d399", "#f87171",
]


def _channel(value: float) -> float:
    """sRGB channel to linear light, per WCAG."""
    value = value / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _rgb(colour: str) -> tuple[int, int, int]:
    colour = colour.lstrip("#")
    if len(colour) == 3:
        colour = "".join(channel * 2 for channel in colour)
    if len(colour) != 6:
        raise ValueError(f"{colour!r} is not a six-digit hex colour")
    return int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16)


def relative_luminance(colour: str) -> float:
    red, green, blue = (_channel(c) for c in _rgb(colour))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(one: str, other: str) -> float:
    """WCAG 2.1 contrast ratio, 1.0 (identical) to 21.0 (black on white)."""
    first, second = relative_luminance(one), relative_luminance(other)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def readable_text(background: str, preferred: str, *,
                  minimum: float = MIN_CONTRAST) -> str:
    """``preferred`` if it is legible on ``background``, else ink or white.

    Club badges declare their own text colour, but a club's second kit colour
    is not always readable against its first - yellow on white, for instance -
    and a badge nobody can read is worse than one off-brand.
    """
    if contrast_ratio(background, preferred) >= minimum:
        return preferred
    on_dark = contrast_ratio(background, INK)
    on_light = contrast_ratio(background, "#ffffff")
    return INK if on_dark >= on_light else "#ffffff"


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c))):02x}" for c in rgb)


def _mix(colour: str, target: str, amount: float) -> str:
    """Move ``colour`` ``amount`` of the way towards ``target``."""
    first, second = _rgb(colour), _rgb(target)
    return _hex(tuple(a + (b - a) * amount for a, b in zip(first, second)))


def enforce_contrast(colour: str, against: str, *, minimum: float = MIN_CONTRAST,
                     steps: int = 50) -> str:
    """Nudge ``colour`` until it reads against ``against``, keeping its hue.

    Two places need this and neither is optional. A vivid mid-tone red - PSV's
    and Arsenal's, say - reaches only ~4.4:1 against *both* white and ink, so
    no choice of text colour alone is legible; the ground has to deepen a
    little. And a club whose colour is navy would disappear entirely when its
    badge is drawn as an outline on a dark card, so it has to lighten.

    The shift is the smallest one that clears ``minimum``, so a badge stays as
    close to the club's own colour as legibility allows.
    """
    if contrast_ratio(colour, against) >= minimum:
        return colour
    # Away from the thing it has to contrast with: darken against a light
    # surface, lighten against a dark one.
    target = "#000000" if relative_luminance(against) > relative_luminance(colour) \
        else "#ffffff"
    best = colour
    for step in range(1, steps + 1):
        best = _mix(colour, target, step / steps)
        if contrast_ratio(best, against) >= minimum:
            return best
    return best


def to_lab(colour: str) -> tuple[float, float, float]:
    """CIELAB, for judging whether two colours look alike."""
    red, green, blue = (_channel(c) for c in _rgb(colour))
    # sRGB to XYZ (D65), then XYZ to Lab.
    x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883

    def f(value: float) -> float:
        return value ** (1 / 3) if value > 0.008856 else 7.787 * value + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def colour_distance(one: str, other: str) -> float:
    """CIE76 distance. Below ~25 two badges read as the same colour at a glance."""
    first, second = to_lab(one), to_lab(other)
    return sum((a - b) ** 2 for a, b in zip(first, second)) ** 0.5
