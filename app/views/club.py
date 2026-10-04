"""One club, from every angle the published artefacts allow.

Reached by tapping a club anywhere in the app, which is a plain link to
``club?club=ARS``. That is a full page load - Streamlit offers no way to set a
widget from inside injected HTML - so the pinned club travels in the address
beside it, or tapping a card would quietly unpin.

Every section degrades on its own: an artefact that the weekly update has not
published yet removes its section and nothing else.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ucl.webapp import (
    cards, charts, clubs, loaders, movement, nav, opponents, paths, st_helpers,
)
from ucl.webapp.palette import HOME, MUTED

SEASON = paths.SEASON
matchday = paths.latest_matchday(SEASON)

try:
    forecast = loaders.load_forecast(matchday, SEASON)
except FileNotFoundError as error:
    st_helpers.missing(error)
    st.stop()

ranked = forecast.sort_values("p_champion", ascending=False).reset_index(drop=True)
ranked["position"] = ranked.index + 1
names = sorted(ranked.team)
pin = st_helpers.pinned(SEASON)

# The URL names the club; the pin is the fallback, then the title favourite, so
# the page always has something to show rather than an empty state.
wanted = nav.resolve_club(st.query_params.get(nav.CLUB_PARAM), pin=pin,
                          default=str(ranked.team.iloc[0]))
if wanted not in names:
    wanted = str(ranked.team.iloc[0])

choice = st.selectbox(
    "Club", [clubs.CODES[name] for name in names], key=nav.CLUB_PARAM,
    bind="query-params", persist_state="session", label_visibility="collapsed",
    format_func=lambda code: f"{clubs.from_code(code)} ({code})",
    index=names.index(wanted),
)
club = clubs.from_code(choice) or wanted

mine = ranked.loc[ranked.team == club]
row = mine.iloc[0]
st.title(club)
st.caption(f"{clubs.code(club)} · ranked {int(row.position)} of {len(ranked)} "
           f"by P(champion) · through matchday {matchday}")

# --- the three bands, and the pin ----------------------------------------
st.html(cards.club_cards_html(mine.to_dict("records"), link=False))

if pin == club:
    st_helpers.note("Pinned to the home screen.")
else:
    st.html(
        f'<a href="{nav.home_href(pin=club)}" style="display:inline-block;'
        f'text-decoration:none;color:#2dd4bf;border:1px solid #2dd4bf;'
        f'border-radius:8px;padding:6px 11px;font:600 12px/1.2 {cards.FONT}">'
        f'Pin {clubs.code(club)} to Home</a>')

# --- every stage ----------------------------------------------------------
st_helpers.section("Every stage", "probability of reaching each round")
STAGES = [("p_top8", "top 8"), ("p_play_off", "play-off"),
          ("p_round_of_16", "round of 16"), ("p_quarter_finals", "quarter-final"),
          ("p_semi_finals", "semi-final"), ("p_final", "final"),
          ("p_champion", "champion"), ("p_eliminated", "eliminated")]
st.html("".join(
    cards.bar_row_html(label, float(row[column]),
                       HOME if column != "p_eliminated" else "#f87171",
                       delta=row.get(f"d_{column}"))
    for column, label in STAGES if column in mine.columns))

# --- how it has moved -----------------------------------------------------
series = movement.forecast_series(SEASON)
if len(series) and series.matchday.nunique() > 1:
    st_helpers.section("Across the matchdays",
                       "this club in colour, the other 35 behind it")
    metric = st.segmented_control(
        "Metric", ["P(top 8)", "P(champion)"], default="P(top 8)",
        label_visibility="collapsed", key="club_metric")
    column = {"P(top 8)": "p_top8", "P(champion)": "p_champion"}[
        metric or "P(top 8)"]
    st.altair_chart(charts.trend(series, column, [club]), width="stretch")
else:
    st_helpers.section("Across the matchdays", "needs a second published matchday")

# --- where it finishes ----------------------------------------------------
try:
    positions = loaders.load_positions(matchday, SEASON)
except FileNotFoundError:
    positions = None
if positions is not None:
    st_helpers.section("Where it finishes",
                       f"{positions.attrs['iterations']:,} simulated seasons; "
                       f"dashed lines are the 8th and 24th places")
    st.altair_chart(charts.position_bars(positions, club),
                    width="stretch")
    mass = positions[positions.team == club].set_index("position").probability
    mean = float((mass.index.to_numpy() * mass.to_numpy()).sum())
    st_helpers.note(
        f"P(1-8) {mass.loc[1:8].sum():.1%} · P(9-24) {mass.loc[9:24].sum():.1%} "
        f"· P(25-36) {mass.loc[25:36].sum():.1%} · mean position {mean:.1f}")

# --- who it might meet ----------------------------------------------------
try:
    pairings = loaders.load_knockout_paths(matchday, SEASON)
except FileNotFoundError:
    pairings = None
if pairings is not None:
    st_helpers.section("Who it might meet",
                       "P(this tie happens at all), so each round sums to the "
                       "chance of playing in it")
    reached = opponents.reach(pairings, club).set_index("round")
    for round_name in opponents.ROUNDS:
        chance = float(reached.probability[round_name])
        if chance < 0.005:
            continue
        top = opponents.per_round(pairings, club, round_name, count=5)
        st_helpers.note(f"<b>{opponents.LABELS[round_name]}</b> - "
                        f"plays in it {chance:.1%} of the time")
        st.html(cards.compact_rows_html(
            [{"team": r.opponent, "value": r.probability, "colour": MUTED}
             for r in top.itertuples()], pin=pin))

# --- what the model knows about it ----------------------------------------
st_helpers.section("What the model knows about it", "its inputs, not outputs")
facts = []
try:
    ratings = loaders.load_ratings(matchday, SEASON).set_index("team")
    facts.append(("Elo", f"{ratings.elo[club]:,.0f}",
                  f"after matchday {matchday}"))
except (FileNotFoundError, KeyError):
    pass
try:
    inputs = loaders.load_club_inputs(SEASON).set_index("team")
    mine_inputs = inputs.loc[club]
    value = mine_inputs.get("squad_value_eur_m")
    facts.append((
        "Squad value",
        f"EUR {value:,.0f}m" if pd.notna(value) else "not covered",
        "frozen pre-season prior, as of 2026-07-06"))
    sot = mine_inputs.get("sot_attack")
    facts.append((
        "Shots on target vs Elo",
        f"{sot:+.2f}" if pd.notna(sot) else "not covered",
        "attacking shots on target beyond what this club's rating predicts, "
        "at the 2026-09-07 cutoff"))
except (FileNotFoundError, KeyError):
    pass

if facts:
    st.html("".join(
        f'<div style="display:flex;gap:8px;align-items:baseline;padding:6px 0;'
        f'border-bottom:1px solid #1e293b">'
        f'<span style="color:#e2e8f0;font:600 13px/1.3 {cards.FONT};'
        f'min-width:92px">{label}</span>'
        f'<span style="color:#2dd4bf;font:700 13px/1.3 {cards.FONT}">{value}</span>'
        f'<span style="color:#64748b;font:500 11px/1.4 {cards.FONT};'
        f'margin-left:auto;text-align:right">{note}</span></div>'
        for label, value, note in facts))
    st_helpers.note(
        "Four clubs have no squad value and seven no shot rating, because no "
        "free source covers their leagues; an uncovered club falls back on the "
        "rest of the model. That is the half-coverage the model was accepted "
        "at, not a gap in this page.")

# --- its remaining fixtures ----------------------------------------------
try:
    remaining = loaders.load_remaining_fixtures(matchday, SEASON)
except FileNotFoundError:
    remaining = None
if remaining is not None:
    ours = remaining[(remaining.home == club) | (remaining.away == club)]
    if len(ours):
        st_helpers.section(
            f"Remaining fixtures ({len(ours)})",
            "at today's ratings - not the simulator's own numbers")
        rows = ours.sort_values("date").assign(
            date=ours.sort_values("date").date.dt.strftime("%d %b")
        ).to_dict("records")
        st.html(cards.fixture_cards_html(rows))
        st_helpers.note(
            "Only the next matchday is a true pre-match forecast. Beyond it "
            "these are the model evaluated at <b>today's</b> ratings, which "
            "ignores the results in between - the forecast itself updates the "
            "ratings inside every simulated season, so its own numbers for "
            "matchday 8 differ from these."
        )

# --- the market, where there is one --------------------------------------
if "p_market" in mine.columns and pd.notna(row.get("p_market")):
    st_helpers.section("Against the market", "a diagnostic, never an objective")
    st_helpers.note(
        f"Model {row.p_champion:.1%} · bookmaker de-vigged {row.p_market:.1%} · "
        f"model is {row.model_over_market:.2f}x the market. One pre-season "
        f"snapshot with a 30.8% overround, de-vigged proportionally, which "
        f"ignores favourite-longshot bias.")
