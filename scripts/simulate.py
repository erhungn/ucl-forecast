"""Monte Carlo forecast of a Champions League season from a frozen information set.

Each iteration plays the 144 league-phase fixtures with ratings updating as it
goes, ranks the table under Article 18, draws the knockout bracket inside the
Article 19 constraints, and plays it out to a champion.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import pandas as pd

from ucl import config
from ucl.config import OUTPUTS_DIR, season as season_config
from ucl.data import fixtures as fx
from ucl.data import names, odds, odds
from ucl.models.elo import elo_as_of
from ucl.models.poisson import fit_poisson
from ucl.sim.bracket import draw_bracket
from ucl.sim.knockout import run_knockout
from ucl.sim.league_phase import simulate_league_phase

STAGES = ["top8", "play_off", "eliminated", "round_of_16", "quarter_finals",
          "semi_finals", "final", "champion"]


def load_fixtures(season_label: str) -> pd.DataFrame:
    """Fixtures with club ids, so they join to the Elo ratings."""
    frame = pd.read_csv(fx.fixtures_path(season_label))
    lookup = names.alias_to_canonical("wikipedia")
    frame["home_id"] = frame.home.map(lookup).fillna(frame.home)
    frame["away_id"] = frame.away.map(lookup).fillna(frame.away)
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("-n", "--iterations", type=int, default=config.N_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = season_config(args.season)

    started = time.perf_counter()
    fixtures = load_fixtures(cfg.label)
    ratings = elo_as_of(cfg.cutoff_date)
    model = fit_poisson(ratings.pre_match)
    teams = sorted(set(fixtures.home_id) | set(fixtures.away_id))
    missing = [t for t in teams if t not in ratings.ratings]
    if missing:
        raise SystemExit(f"no rating for {missing}")
    base = {team: ratings.ratings[team] for team in teams}
    setup = time.perf_counter() - started

    print(f"{cfg.label}: {len(fixtures)} fixtures, {len(teams)} teams, "
          f"{args.iterations} iterations, seed {args.seed}")
    print(f"model a={model.a:.5f} h={model.h:.5f} b={model.b:.5f}")

    counts = {stage: Counter() for stage in STAGES}
    positions = Counter()
    unresolved_ties = 0
    rng = np.random.default_rng(args.seed)

    simulation_started = time.perf_counter()
    for _ in range(args.iterations):
        table, live = simulate_league_phase(fixtures, base, model, rng)
        unresolved_ties += int(table.unresolved.sum())

        by_position = dict(zip(table.position, table.team))
        # Every iteration must split 36 clubs into 8 / 16 / 12 exactly.
        top8 = [by_position[p] for p in range(1, 9)]
        play_off = [by_position[p] for p in range(9, 25)]
        eliminated = [by_position[p] for p in range(25, 37)]
        assert (len(top8), len(play_off), len(eliminated)) == (8, 16, 12)
        counts["top8"].update(top8)
        counts["play_off"].update(play_off)
        counts["eliminated"].update(eliminated)
        for position, team in by_position.items():
            positions[(team, position)] += 1

        bracket = draw_bracket(by_position, rng=rng)
        outcome = run_knockout(bracket, live, model, rng)
        counts["round_of_16"].update(outcome["round_of_16"])
        counts["quarter_finals"].update(outcome["quarter_finals"])
        counts["semi_finals"].update(outcome["semi_finals"])
        counts["final"].update(outcome["final"])
        counts["champion"].update([outcome["champion"]])
    elapsed = time.perf_counter() - simulation_started

    n = args.iterations
    frame = pd.DataFrame({"team": teams})
    for stage in STAGES:
        frame[f"p_{stage}"] = frame.team.map(lambda t, s=stage: counts[s][t] / n)
    frame["mean_position"] = frame.team.map(
        lambda t: sum(p * positions[(t, p)] for p in range(1, 37)) / n
    )
    frame = frame.sort_values("p_champion", ascending=False).reset_index(drop=True)

    # The market, as a diagnostic only - never a fitting target.
    try:
        market = odds.load_outright(cfg.label)
    except FileNotFoundError:
        market = None
    if market is not None:
        frame = frame.merge(
            market[["team", "decimal_odds", "p_market"]], on="team", how="left"
        )
        frame["model_minus_market"] = frame.p_champion - frame.p_market
        frame["model_over_market"] = frame.p_champion / frame.p_market

    # --- verification ---
    checks = {
        "P(champion) sums to 1": abs(frame.p_champion.sum() - 1.0) < 1e-9,
        "P(top8)+P(play-off)+P(eliminated) = 1 per club":
            bool(np.allclose(frame.p_top8 + frame.p_play_off + frame.p_eliminated, 1.0)),
        "8 clubs in the top-8 band per iteration": abs(frame.p_top8.sum() - 8.0) < 1e-9,
        "16 clubs in the play-off band per iteration":
            abs(frame.p_play_off.sum() - 16.0) < 1e-9,
        "12 clubs eliminated per iteration": abs(frame.p_eliminated.sum() - 12.0) < 1e-9,
        "16 clubs reach the round of 16": abs(frame.p_round_of_16.sum() - 16.0) < 1e-9,
        "8 reach the quarter-finals": abs(frame.p_quarter_finals.sum() - 8.0) < 1e-9,
        "4 reach the semi-finals": abs(frame.p_semi_finals.sum() - 4.0) < 1e-9,
        "2 reach the final": abs(frame.p_final.sum() - 2.0) < 1e-9,
        "stage probabilities are monotone":
            bool((frame.p_round_of_16 >= frame.p_quarter_finals - 1e-12).all()
                 and (frame.p_quarter_finals >= frame.p_semi_finals - 1e-12).all()
                 and (frame.p_semi_finals >= frame.p_final - 1e-12).all()
                 and (frame.p_final >= frame.p_champion - 1e-12).all()),
    }
    print("\n=== verification ===")
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"  unresolved league-phase ties: {unresolved_ties} "
          f"({unresolved_ties / n:.4f} per iteration)")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / (args.out or f"forecast_{cfg.label}.csv")
    frame.to_csv(out, index=False, float_format="%.6f")
    if market is not None:
        priced = frame.p_market.notna()
        unpriced_mass = float(frame.loc[~priced, "p_champion"].sum())
        print(f"\n=== model vs market ({int(priced.sum())} of 36 priced, overround "
              f"{market.attrs['overround']:.4f}, proportional de-vig) ===")
        print(f"  the {int((~priced).sum())} unpriced clubs hold {unpriced_mass:.3f} of model "
              f"probability; proportional de-vig gives that share to the priced clubs, so "
              f"every market number reads about {unpriced_mass:.1%} high")
        ranked = frame.loc[priced].reindex(
            frame.loc[priced].model_minus_market.abs().sort_values(ascending=False).index)
        print("  largest disagreements:")
        for row in ranked.head(10).itertuples():
            side = "model HIGH" if row.model_minus_market > 0 else "model LOW "
            print(f"    {row.team:22s} model {row.p_champion:.3f}  market {row.p_market:.3f}  "
                  f"{side} by {abs(row.model_minus_market):.3f}")

    print(f"\nsetup {setup:.1f}s, simulation {elapsed:.1f}s "
          f"({elapsed / n * 1000:.2f} ms/iteration), written to {out.name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
