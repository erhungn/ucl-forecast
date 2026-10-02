"""Article 19 of the 2026/27 regulations, and Annex B's bracket.

19.02 bands the knockout play-off draw: "clubs 9 or 10 against clubs 23 or 24,
clubs 11 or 12 against clubs 21 or 22, clubs 13 or 14 against clubs 19 or 20,
and clubs 15 or 16 against clubs 17 or 18", and "in principle, the seeded clubs
play the return leg matches at home".

19.03 pairs clubs 1/2, 3/4, 5/6 and 7/8 and draws each pair "into one of two
positions in the round of 16 ... against the relevant winner of the knockout
phase play-offs".

19.04: "Seeded teams, i.e. teams ranked 1 to 4 after the league phase, play the
return leg at home in the quarter-finals, and teams ranked 1 and 2 also play the
return leg of the semi-finals at home. If a seeded team is beaten in any round,
the team that eliminates them takes over their seeding position in the bracket
path."

Annex B is a diagram with no extractable text, so the round-of-16 allocation and
the quarter/semi tree below were derived from the 2024-25 and 2025-26 brackets,
which agree exactly; those two seasons are asserted to be legal draws.
"""

from __future__ import annotations

import numpy as np
import pytest

from ucl.sim.bracket import (
    PLAY_OFF_BANDS,
    R16_BANDS,
    Bracket,
    draw_bracket,
)

TEAMS = {position: f"T{position:02d}" for position in range(1, 37)}


def _draw(seed: int) -> Bracket:
    return draw_bracket(TEAMS, rng=np.random.default_rng(seed))


@pytest.mark.parametrize("seed", range(25))
def test_play_off_ties_respect_the_article_19_02_bands(seed):
    bracket = _draw(seed)
    for tie in bracket.play_offs:
        band = next(b for b in PLAY_OFF_BANDS if tie.seed_high in b)
        assert tie.seed_low in PLAY_OFF_BANDS[band]


@pytest.mark.parametrize("seed", range(25))
def test_every_club_from_9_to_24_plays_exactly_one_play_off(seed):
    bracket = _draw(seed)
    seeds = sorted(s for tie in bracket.play_offs for s in (tie.seed_high, tie.seed_low))
    assert seeds == list(range(9, 25))


@pytest.mark.parametrize("seed", range(25))
def test_the_seeded_club_hosts_the_return_leg(seed):
    bracket = _draw(seed)
    for tie in bracket.play_offs:
        assert tie.second_leg_home == tie.team_high
        assert tie.first_leg_home == tie.team_low


@pytest.mark.parametrize("seed", range(25))
def test_round_of_16_pairs_each_seed_with_its_annex_b_band(seed):
    bracket = _draw(seed)
    for tie in bracket.round_of_16:
        band = R16_BANDS[next(b for b in R16_BANDS if tie.seed_high in b)]
        assert tie.seed_low in band, f"seed {tie.seed_high} drew outside its band"


@pytest.mark.parametrize("seed", range(25))
def test_a_seeded_pair_can_only_meet_in_the_final(seed):
    bracket = _draw(seed)
    for pair in ((1, 2), (3, 4), (5, 6), (7, 8)):
        halves = {bracket.half_of_seed[pair[0]], bracket.half_of_seed[pair[1]]}
        assert halves == {0, 1}, f"seeds {pair} share a half"


@pytest.mark.parametrize("seed", range(25))
def test_quarter_finals_pair_one_two_with_seven_eight_and_three_four_with_five_six(seed):
    bracket = _draw(seed)

    def group(position: int) -> tuple[int, int]:
        occupant = bracket.slots[position].seed_high
        return next(p for p in ((1, 2), (3, 4), (5, 6), (7, 8)) if occupant in p)

    for left, right in bracket.quarter_final_slots():
        assert {group(left), group(right)} in ({(1, 2), (7, 8)}, {(3, 4), (5, 6)})


@pytest.mark.parametrize("seed", range(25))
def test_lower_seed_number_hosts_the_second_leg_in_every_round(seed):
    bracket = _draw(seed)
    for tie in [*bracket.play_offs, *bracket.round_of_16]:
        assert tie.seed_high < tie.seed_low
        assert tie.second_leg_home == tie.team_high


def test_each_club_appears_exactly_once_per_round():
    bracket = _draw(0)
    play_off_clubs = [t for tie in bracket.play_offs for t in (tie.team_high, tie.team_low)]
    assert len(play_off_clubs) == len(set(play_off_clubs)) == 16
    r16_seeded = [tie.team_high for tie in bracket.round_of_16]
    assert len(r16_seeded) == len(set(r16_seeded)) == 8


def test_draws_are_deterministic_under_a_seed():
    first, second = _draw(11), _draw(11)
    assert [t.as_tuple() for t in first.play_offs] == [t.as_tuple() for t in second.play_offs]
    assert [t.as_tuple() for t in first.round_of_16] == [t.as_tuple() for t in second.round_of_16]


def test_draws_actually_vary_between_seeds():
    shapes = {tuple(t.as_tuple() for t in _draw(s).play_offs) for s in range(20)}
    assert len(shapes) > 1


@pytest.mark.parametrize("season_label", ["2024-25", "2025-26"])
def test_the_real_brackets_are_legal_draws(season_label):
    """The observed 2024-25 and 2025-26 brackets must be reachable draws."""
    from ucl.data.openfootball import CACHE_DIR, parse_file
    from ucl.sim.bracket import is_legal_play_off_pairing, is_legal_round_of_16_pairing
    from ucl.sim.tiebreakers import rank_table

    played = parse_file(CACHE_DIR / f"{season_label}_cl.txt",
                        season=season_label, competition="UCL")
    league = played[played.stage.str.startswith("League") & played.played]
    position = dict(zip(*rank_table(league)[["team", "position"]].values.T))

    play_offs = played[played.stage.str.contains("Playoff", case=False) & played.played]
    pairs = {tuple(sorted((position[r.home], position[r.away]))) for r in play_offs.itertuples()}
    assert len(pairs) == 8
    for seeded, unseeded in pairs:
        assert is_legal_play_off_pairing(seeded, unseeded)

    r16 = played[played.stage.str.contains("Round of 16", case=False) & played.played]
    r16_pairs = {tuple(sorted((position[r.home], position[r.away]))) for r in r16.itertuples()}
    for seed, qualifier in r16_pairs:
        assert seed <= 8, "the seeded side of a round-of-16 tie must be 1-8"
        assert is_legal_round_of_16_pairing(seed, qualifier)
