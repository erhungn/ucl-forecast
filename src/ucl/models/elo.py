"""Our own Elo, computed from match results, plus in-simulation rating updates.

ClubElo was the plan's rating source but its API is down and it has no mirror,
so ratings are computed here from domestic results (:mod:`ucl.data.football_data`)
linked across leagues by European results (:mod:`ucl.data.openfootball`).
:mod:`ucl.data.clubelo` stays available as an optional cross-check.
"""
