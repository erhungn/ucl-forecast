"""The knockout bracket, per Article 19 and Annex B of the 2026/27 regulations.

Article 19.02 bands the play-off draw by league position: 9/10 against 23/24,
11/12 against 21/22, 13/14 against 19/20, 15/16 against 17/18, with the seeded
club at home in the return leg. Article 19.03 pairs 1/2, 3/4, 5/6 and 7/8 and
draws each pair into one of two round-of-16 positions against the relevant
play-off winner.

Annex B, which fixes *which* play-off winner each seeded pair meets and the
quarter/semi tree, is a diagram in the PDF with no extractable text. The
structure below was therefore derived from the 2024-25 and 2025-26 brackets,
which agree exactly, and both are asserted to be legal draws in the tests:

    seeds 1,2 -> winners from the 15/16 v 17/18 band
    seeds 3,4 -> winners from the 13/14 v 19/20 band
    seeds 5,6 -> winners from the 11/12 v 21/22 band
    seeds 7,8 -> winners from the 9/10 v 23/24 band

Bracket positions 0..7 form the tree (0v1, 2v3, 4v5, 6v7 in the quarter-finals;
then 0..3 and 4..7 in the semi-finals), with the two clubs of a seeded pair
always placed in opposite halves, so they can only meet in the final.

Home advantage follows one rule in every two-legged round: the side carrying the
lower seeding position hosts the second leg. That reproduces 19.02 and 19.03
("the seeded clubs play the return leg matches at home") and 19.04 (seeds 1-4 at
home in the quarter-final return leg, seeds 1-2 in the semi-final return leg),
with seeding positions inherited by whoever eliminates a seeded club.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PLAY_OFF_BANDS", "R16_BANDS", "Tie", "Bracket", "draw_bracket",
    "is_legal_play_off_pairing", "is_legal_round_of_16_pairing",
]

#: Article 19.02: seeded pair -> the unseeded pair it is drawn against.
PLAY_OFF_BANDS: dict[tuple[int, int], tuple[int, int]] = {
    (9, 10): (23, 24),
    (11, 12): (21, 22),
    (13, 14): (19, 20),
    (15, 16): (17, 18),
}

#: Annex B: seeded pair -> the play-off band whose winners it meets.
R16_BANDS: dict[tuple[int, int], tuple[int, ...]] = {
    (1, 2): (15, 16, 17, 18),
    (3, 4): (13, 14, 19, 20),
    (5, 6): (11, 12, 21, 22),
    (7, 8): (9, 10, 23, 24),
}

#: Bracket position of each seeded pair's two clubs: (half 0 slot, half 1 slot).
#: 0v1 and 2v3 meet in one semi-final, 4v5 and 6v7 in the other.
_PAIR_SLOTS: dict[tuple[int, int], tuple[int, int]] = {
    (1, 2): (0, 4),
    (7, 8): (1, 5),
    (3, 4): (2, 6),
    (5, 6): (3, 7),
}


@dataclass(frozen=True)
class Tie:
    """A two-legged tie. The lower seeding position hosts the second leg."""

    team_high: str          # the better-seeded club
    team_low: str
    seed_high: int
    seed_low: int

    @property
    def first_leg_home(self) -> str:
        return self.team_low

    @property
    def second_leg_home(self) -> str:
        return self.team_high

    def as_tuple(self) -> tuple[str, str, int, int]:
        return (self.team_high, self.team_low, self.seed_high, self.seed_low)


def is_legal_play_off_pairing(seeded: int, unseeded: int) -> bool:
    band = next((v for k, v in PLAY_OFF_BANDS.items() if seeded in k), None)
    return band is not None and unseeded in band


def is_legal_round_of_16_pairing(seed: int, qualifier_position: int) -> bool:
    band = next((v for k, v in R16_BANDS.items() if seed in k), None)
    return band is not None and qualifier_position in band


@dataclass
class Bracket:
    play_offs: list[Tie]
    round_of_16: list[Tie]
    slot_of_seed: dict[int, int]
    half_of_seed: dict[int, int]
    #: Round-of-16 tie occupying each bracket position 0..7.
    slots: list[Tie]

    def quarter_final_slots(self) -> list[tuple[int, int]]:
        return [(0, 1), (2, 3), (4, 5), (6, 7)]

    def semi_final_slots(self) -> list[tuple[int, int]]:
        return [(0, 2), (4, 6)]


def draw_bracket(teams: dict[int, str], *, rng: np.random.Generator) -> Bracket:
    """Draw the whole knockout bracket from final league positions.

    ``teams`` maps league position (1-36) to club id. Every choice UEFA makes by
    draw is randomised here inside the regulation's constraints.
    """
    play_offs: list[Tie] = []
    winner_band: dict[tuple[int, int], list[int]] = {}
    for seeded_pair, unseeded_pair in PLAY_OFF_BANDS.items():
        opponents = list(unseeded_pair)
        if rng.random() < 0.5:                      # which seeded club meets which
            opponents.reverse()
        for seeded, unseeded in zip(seeded_pair, opponents):
            play_offs.append(Tie(teams[seeded], teams[unseeded], seeded, unseeded))
        winner_band[seeded_pair] = [*seeded_pair, *unseeded_pair]

    # Each seeded pair takes one bracket position in each half.
    slot_of_seed: dict[int, int] = {}
    half_of_seed: dict[int, int] = {}
    for pair, (slot_a, slot_b) in _PAIR_SLOTS.items():
        first, second = pair
        if rng.random() < 0.5:
            first, second = second, first
        slot_of_seed[first], slot_of_seed[second] = slot_a, slot_b
        half_of_seed[first], half_of_seed[second] = 0, 1

    # Each seeded pair meets the two winners out of its Annex B band. Which of
    # the two it faces is drawn; the qualifier is a placeholder until the
    # play-offs are actually played.
    round_of_16: list[Tie] = []
    slots: list[Tie | None] = [None] * 8
    for seeded_pair, band in R16_BANDS.items():
        source = next(k for k in PLAY_OFF_BANDS if k[0] in band)
        qualifiers = list(source)                   # the two seeded play-off slots
        if rng.random() < 0.5:
            qualifiers.reverse()
        for seed, qualifier_slot in zip(seeded_pair, qualifiers):
            tie = Tie(teams[seed], f"W{qualifier_slot}", seed, qualifier_slot)
            round_of_16.append(tie)
            slots[slot_of_seed[seed]] = tie

    return Bracket(
        play_offs=play_offs,
        round_of_16=round_of_16,
        slot_of_seed=slot_of_seed,
        half_of_seed=half_of_seed,
        slots=[t for t in slots if t is not None],
    )
