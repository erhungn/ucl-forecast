"""Paths to the final: who meets whom, and how the bracket is seeded.

Deliberately **not** a filled-in bracket. The simulator draws the knockout
inside the Article 19 constraints once per iteration, so a bracket position is
a different fixture in each of the 50,000 seasons and "the most likely club in
slot 3" would be a number with no referent. What the simulation can say is
P(club A meets club B in round R), which is what this page shows.

The seeding ladder below it is not simulated at all - it is the regulation,
fixed before a ball is kicked - so it is the part that will still be true when
the real draw is made, and the part the real draw will slot into.
"""

from __future__ import annotations

import streamlit as st

from ucl.webapp import cards, loaders, opponents, paths, st_helpers
from ucl.webapp.palette import MUTED
from ucl.sim.bracket import PLAY_OFF_BANDS, R16_BANDS

SEASON = paths.SEASON
matchday = paths.latest_matchday(SEASON)

st.title("Paths to the final")

try:
    pairings = loaders.load_knockout_paths(matchday, SEASON)
    forecast = loaders.load_forecast(matchday, SEASON)
except FileNotFoundError as error:
    st_helpers.missing(error)
    st.stop()

iterations = opponents.iterations(pairings)
pin = st_helpers.pinned(SEASON)
st.caption(f"Through matchday {matchday}, over {iterations:,} simulated "
           f"seasons. The draw is re-made inside the regulation constraints in "
           f"every one of them.")

# --- one club's path ------------------------------------------------------
names = sorted(forecast.team)
default = pin if pin in names else str(
    forecast.nlargest(1, "p_champion").team.iloc[0])
club = st.selectbox("Club", names, index=names.index(default),
                    label_visibility="collapsed", key="knockout_club")

st_helpers.section(f"{club}'s likeliest opponents",
                   "P(the tie happens at all), round by round")
reached = opponents.reach(pairings, club).set_index("round")
for round_name in opponents.ROUNDS:
    chance = float(reached.probability[round_name])
    if chance < 0.005:
        st_helpers.note(f"<b>{opponents.LABELS[round_name]}</b> - reached in "
                        f"under 0.5% of seasons, so no opponent is worth "
                        f"naming")
        continue
    st_helpers.note(f"<b>{opponents.LABELS[round_name]}</b> - plays in it "
                    f"{chance:.1%} of the time")
    st.html(cards.compact_rows_html(
        [{"team": row.opponent, "value": row.probability, "colour": MUTED}
         for row in opponents.per_round(pairings, club, round_name,
                                        count=5).itertuples()], pin=pin))

st_helpers.note(
    "These are unconditional: they sum to the chance of playing in the round, "
    "not to one. A club with a 12% chance of a quarter-final cannot have a "
    "likely quarter-final opponent, and conditioning the numbers would hide "
    "exactly that."
)

# --- the likeliest ties in the competition -------------------------------
st_helpers.section("The likeliest ties", "across all 36 clubs")
round_choice = st.segmented_control(
    "Round", [opponents.LABELS[r] for r in opponents.ROUNDS],
    default=opponents.LABELS["round_of_16"], label_visibility="collapsed",
    key="knockout_round")
chosen = next((r for r in opponents.ROUNDS
               if opponents.LABELS[r] == (round_choice
                                          or opponents.LABELS["round_of_16"])),
              "round_of_16")
top = opponents.likeliest(pairings, chosen, count=8)
st.html("".join(
    f'<div style="display:flex;align-items:center;gap:6px;padding:6px 0;'
    f'border-bottom:1px solid #1e293b">'
    f'{cards.badge_html(row.club)}{cards.badge_html(row.opponent)}'
    f'<span style="color:#e2e8f0;font:500 12.5px/1.3 {cards.FONT};'
    f'flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;'
    f'white-space:nowrap">{row.tie}</span>'
    f'<span style="color:#2dd4bf;font:700 12.5px/1.3 {cards.FONT}">'
    f'{row.probability:.2%}</span></div>'
    for row in top.itertuples()) or "<div>Nothing to show.</div>")

# --- the seeding ladder, which is the regulation and not a simulation ----
st_helpers.section("How the bracket is seeded",
                   "Article 19, fixed before the season - not simulated")
st.html("".join(
    f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;'
    f'border-bottom:1px solid #1e293b;color:#e2e8f0;'
    f'font:500 12.5px/1.4 {cards.FONT}">'
    f'<span style="color:#2dd4bf;font-weight:700;min-width:54px">'
    f'{seeded[0]}/{seeded[1]}</span>'
    f'<span style="color:#64748b">play</span>'
    f'<span style="font-weight:600">{unseeded[0]}/{unseeded[1]}</span>'
    f'<span style="color:#64748b;margin-left:auto;font-size:11px">'
    f'seeded club hosts the second leg</span></div>'
    for seeded, unseeded in PLAY_OFF_BANDS.items()))
st_helpers.note(
    "Then in the round of 16: "
    + " · ".join(f"<b>{pair[0]}/{pair[1]}</b> meet a winner from "
                 f"{'/'.join(str(p) for p in band)}"
                 for pair, band in R16_BANDS.items())
    + ". The two clubs of a seeded pair are placed in opposite halves, so they "
      "can only meet in the final."
)
st_helpers.note(
    "<b>Where this structure came from.</b> Article 19.02 and 19.03 are "
    "readable text, but Annex B - which fixes exactly which play-off winner "
    "each seeded pair meets, and the quarter and semi tree - is a diagram in "
    "the PDF with no extractable text. The tree above was derived from the "
    "2024-25 and 2025-26 brackets, which agree exactly, and the tests assert "
    "every simulated draw is legal under it. When the real draw is made, it "
    "slots into this ladder and replaces the probabilities above."
)
