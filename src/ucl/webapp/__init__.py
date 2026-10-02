"""Logic behind the local forecast app, kept out of the Streamlit pages.

Everything here is pure pandas and reads only what the weekly update published
under ``live/``. Streamlit is imported in exactly one module,
:mod:`ucl.webapp.st_helpers`, so the test suite never needs it installed and
these functions can be exercised directly.

The app is a **reader**. The only writer of ``live/`` is
``scripts/weekly_update.py``, and nothing here writes anywhere.
"""

from __future__ import annotations

from ucl.webapp import loaders, movement, paths, scoring, staleness, update

__all__ = ["loaders", "movement", "paths", "scoring", "staleness", "update"]
