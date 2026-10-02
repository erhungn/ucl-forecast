"""Entry point for the local forecast app.

    .venv-app/bin/streamlit run app/Home.py

Navigation is declared here rather than by a ``pages/`` directory, so this file
runs on every page and the sidebar banner is written once. The views live in
``app/views/`` - deliberately not ``app/pages/``, which Streamlit would also
auto-discover, giving two navigation menus.
"""

from __future__ import annotations

import streamlit as st

from ucl.webapp import paths, st_helpers

st.set_page_config(page_title="UCL 2026-27 forecast", page_icon=":material/trophy:",
                   layout="wide")

st_helpers.sidebar(paths.SEASON)

navigation = st.navigation([
    st.Page("views/title_race.py", title="Title race",
            icon=":material/leaderboard:", default=True),
    st.Page("views/next_matchday.py", title="Next matchday",
            icon=":material/sports_soccer:"),
    st.Page("views/movement.py", title="Movement", icon=":material/trending_up:"),
    st.Page("views/played_matches.py", title="Played matches",
            icon=":material/scoreboard:"),
    st.Page("views/information_set.py", title="Information set",
            icon=":material/inventory_2:"),
])
navigation.run()
