"""URLs and the club a URL names.

Tapping a club card is a plain ``<a href>`` inside an ``st.html`` blob, because
nothing else makes a whole card tappable: DOMPurify strips event handlers, and
Streamlit offers no way to set a widget value from inside injected HTML. The
cost is that a click is a **full browser navigation**, which starts a new
Streamlit session and empties ``st.session_state``. Anything that has to
survive the tap must therefore travel in the query string - which is why
:func:`club_href` takes the pinned club and writes it back out.

Clubs travel as their three-letter code, not their name: ``Bodo/Glimt``
contains a slash, and ``?club=BVB`` is a better thing to bookmark than
``?club=Borussia%20Dortmund``.

No Streamlit here. This module is imported by the pure card builders, and by
tests that run in the model virtualenv, where Streamlit is not installed.
"""

from __future__ import annotations

from urllib.parse import quote

from ucl.webapp import clubs

__all__ = ["CLUB_PAGE", "CLUB_PARAM", "PIN_PARAM", "href", "club_href",
           "home_href", "code_of", "resolve_club"]

#: The club page's ``url_path``. Relative, with no leading slash, so a link
#: keeps working under ``--server.baseUrlPath``.
CLUB_PAGE = "club"
CLUB_PARAM = "club"
PIN_PARAM = "pin"


def href(page: str, **params: str | None) -> str:
    """``page?key=value`` for the parameters that have a value."""
    query = "&".join(f"{key}={quote(str(value), safe='')}"
                     for key, value in sorted(params.items())
                     if value not in (None, ""))
    return f"{page}?{query}" if query else page


def code_of(value: str | None) -> str | None:
    """The three-letter code for a club named either way round, or ``None``.

    Accepts a club name or an already-coded value, because the pin travels as
    a code in the URL and as a name in the data, and both ends call this.
    """
    if not value:
        return None
    if value in clubs.CODES:
        return clubs.CODES[value]
    return clubs.CODES[named] if (named := clubs.from_code(value)) else None


def club_href(team: str, *, pin: str | None = None) -> str:
    """The link that opens ``team``'s page, carrying the pin through the reload.

    ``pin`` is the pinned club, as a name or a code, or ``None``. It is passed
    on because the navigation is a full page load: a parameter left out of the
    link is a parameter the new session will not have, so an unqualified link
    would silently unpin the club every time the user tapped a card.
    """
    return href(CLUB_PAGE, **{
        CLUB_PARAM: clubs.code(team),
        PIN_PARAM: code_of(pin),
    })


def home_href(*, pin: str | None = None) -> str:
    """The link to the default page, carrying a pin to set there.

    ``./`` rather than ``/``: resolved against ``/club`` it gives ``/``, and it
    keeps working under ``--server.baseUrlPath``, which an absolute path would
    not. ``st.page_link`` would be the native equivalent, but it raises unless
    the page was registered through ``st.navigation`` - which makes any view
    using it untestable in isolation, the pattern every view test here uses.
    """
    return href("./", **{PIN_PARAM: code_of(pin)})


def resolve_club(value: str | None, *, pin: str | None = None,
                 default: str | None = None) -> str | None:
    """Which club a URL means: its own parameter, else the pin, else a default.

    Returns ``None`` only when nothing resolves, so a page can say so rather
    than render someone else's numbers. An unknown code falls through to the
    pin exactly as an absent one does - a hand-edited URL should not be an
    error page.
    """
    for candidate in (value, pin):
        if candidate:
            named = clubs.from_code(candidate)
            if named:
                return named
            if candidate in clubs.CODES:
                return candidate
    return default
