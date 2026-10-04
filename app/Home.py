"""Entry point for the local forecast app.

    .venv-app/bin/streamlit run app/Home.py

Navigation is declared here rather than by a ``pages/`` directory, so this file
runs on every page and the banner is written once. The views live in
``app/views/`` - deliberately not ``app/pages/``, which Streamlit would also
auto-discover, giving two navigation menus.

The menu sits in the top bar (``position="top"``), not the sidebar: on a phone
the sidebar is behind a hamburger, so switching page cost two taps. Page titles
are short for the same reason - eight tabs have to share 390px. The club page
is ``visibility="hidden"``, so it never takes a tab but is still addressable at
``club?club=ARS``, which is what makes every club card a link.
"""

from __future__ import annotations

import streamlit as st

from ucl.webapp import nav, st_helpers

# "centered" rather than "wide": the cards are built for a phone, and a
# centred column keeps them a readable width on a desktop too.
st.set_page_config(page_title="UCL 2026-27", page_icon=":material/trophy:",
                   layout="centered", initial_sidebar_state="collapsed")

navigation = st.navigation([
    st.Page("views/overview.py", title="Home", icon=":material/home:",
            default=True),
    st.Page("views/title_race.py", title="Race",
            icon=":material/leaderboard:"),
    st.Page("views/standings.py", title="Table", icon=":material/table_rows:"),
    st.Page("views/next_matchday.py", title="Next",
            icon=":material/sports_soccer:"),
    st.Page("views/knockout.py", title="Paths",
            icon=":material/account_tree:"),
    st.Page("views/movement.py", title="Moves", icon=":material/trending_up:"),
    st.Page("views/played_matches.py", title="Played",
            icon=":material/scoreboard:"),
    st.Page("views/information_set.py", title="Info",
            icon=":material/inventory_2:"),
    st.Page("views/club.py", title="Club", icon=":material/shield:",
            url_path=nav.CLUB_PAGE, visibility="hidden"),
], position="top")

# Written here, so it is on every page exactly once - Home.py runs on each.
st_helpers.banner()
navigation.run()
