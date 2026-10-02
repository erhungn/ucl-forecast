"""Monte Carlo forecast of a Champions League season from a frozen information set.

Each iteration plays the 144 league-phase fixtures with ratings updating as it
goes, ranks the table under Article 18, draws the knockout bracket inside the
Article 19 constraints, and plays it out to a champion.

``--model v1`` (the default) is the accepted model: Elo plus the pre-season
squad-value prior. ``--model elo`` is the frozen `elo-baseline-v0` model, kept
so the two can be compared and the archived baseline reproduced.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ucl import config
from ucl.config import OUTPUTS_DIR, season as season_config
from ucl.data import fixtures as fx
from ucl.data import league_results, names, odds
from ucl.data.cutoff import as_of_for_matchday
from ucl.models.covariates import fit_squad_value_model
from ucl.models.elo import elo_as_of
from ucl.models.poisson import fit_poisson
from ucl.sim.bracket import draw_bracket
from ucl.sim.knockout import run_knockout
from ucl.sim.league_phase import simulate_league_phase

STAGES = ["top8", "play_off", "eliminated", "round_of_16", "quarter_finals",
          "semi_finals", "final", "champion"]


def load_fixtures(season_label: str) -> pd.DataFrame:
    """Fixtures with club ids, so they join to the Elo ratings."""
    frame = pd.read_csv(fx.fixtures_path(season_label), parse_dates=["date"])
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
    parser.add_argument("--through-matchday", type=int, default=0,
                        help="fix the first k matchdays at their actual scores")
    parser.add_argument("--as-of", default=None,
                        help="information-set date (default: the day before the next matchday)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch the league-phase page before reading results")
    parser.add_argument("--model", choices=("v1", "elo"), default="v1",
                        help="v1: Elo plus the squad-value prior; elo: the frozen baseline")
    args = parser.parse_args(argv)
    cfg = season_config(args.season)
    if cfg.tournament_format != "league_phase":
        raise SystemExit(
            f"{cfg.label} used the {cfg.tournament_format} format (32 teams in 8 groups), "
            "which this simulator does not implement; it exists for the ratings backtest only"
        )

    started = time.perf_counter()
    fixtures = load_fixtures(cfg.label)
    matchday = args.through_matchday

    # Everything decided before the next matchday is in the information set.
    if args.as_of:
        as_of = pd.Timestamp(args.as_of).date()
    else:
        as_of = as_of_for_matchday(fixtures, matchday, cfg.cutoff_date)

    played = pd.DataFrame(columns=league_results.COLUMNS)
    if matchday > 0:
        played = league_results.load_played(cfg.label, refresh=args.refresh)
        complete = league_results.completed_matchdays(played, fixtures)
        for k in range(1, matchday + 1):
            if k not in complete:
                raise SystemExit(
                    f"matchday {k} is not complete ({int((played.matchday == k).sum())} of "
                    f"{int((fixtures.matchday == k).sum())} results); refusing to guess"
                )
        played = played[played.matchday <= matchday]
    remaining = fixtures[fixtures.matchday > matchday]

    print(f"\n=== information set (as of {as_of}) ===")
    info = league_results.information(cfg.label, as_of=as_of)
    print(info.to_string(index=False))
    if matchday > 0:
        print(f"  league phase: matchdays 1-{matchday} fixed at their actual scores "
              f"({len(played)} matches), {len(remaining)} fixtures simulated")
    stale = league_results.staleness(info, as_of)
    if stale:
        print(f"  NOTE: {stale}")

    ratings = elo_as_of(as_of)
    teams = sorted(set(fixtures.home_id) | set(fixtures.away_id))
    if args.model == "v1":
        from ucl.data import transfermarkt

        value_as_of = transfermarkt.as_of_for_season(cfg.label)
        values = transfermarkt.squad_values(value_as_of)
        model = fit_squad_value_model(ratings.pre_match, values, as_of=value_as_of)
    else:
        model = fit_poisson(ratings.pre_match)
    missing = [t for t in teams if t not in ratings.ratings]
    if missing:
        raise SystemExit(f"no rating for {missing}")
    base = {team: ratings.ratings[team] for team in teams}
    setup = time.perf_counter() - started

    print(f"{cfg.label}: {len(remaining)} fixtures to simulate, {len(teams)} teams, "
          f"{args.iterations} iterations, seed {args.seed}")
    print("model " + " ".join(f"{k}={v:.5f}" for k, v in model.as_dict().items())
          + f" [{args.model}]")
    if args.model == "v1":
        valued = [t for t in teams if t in model.values]
        share = model.coverage(fixtures.home_id.to_numpy(), fixtures.away_id.to_numpy())
        print(f"  squad value: {len(valued)}/36 clubs valued, both clubs valued in "
              f"{share:.1%} of the 144 fixtures; the rest fall back to Elo")
        print("  unvalued: " + ", ".join(sorted(set(teams) - set(valued))))

    counts = {stage: Counter() for stage in STAGES}
    positions = Counter()
    unresolved_ties = 0
    rng = np.random.default_rng(args.seed)

    simulation_started = time.perf_counter()
    for _ in range(args.iterations):
        table, live = simulate_league_phase(remaining, base, model, rng, played=played)
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
    suffix = f"_md{matchday}" if matchday > 0 else ""
    out = OUTPUTS_DIR / (args.out or f"forecast_{cfg.label}{suffix}.csv")
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

    # Movement since the previous matchday.
    previous_name = (f"forecast_{cfg.label}_md{matchday - 1}.csv" if matchday > 1
                     else f"forecast_{cfg.label}.csv")
    previous_path = OUTPUTS_DIR / previous_name
    if matchday > 0 and previous_path.exists():
        previous = pd.read_csv(previous_path)[["team", "p_champion", "p_top8", "p_eliminated"]]
        moved = frame.merge(previous, on="team", suffixes=("", "_prev"))
        for column in ("p_champion", "p_top8", "p_eliminated"):
            moved[f"d_{column}"] = moved[column] - moved[f"{column}_prev"]
        print(f"\n=== change since matchday {matchday - 1} ({previous_name}) ===")
        ranked = moved.reindex(moved.d_p_top8.abs().sort_values(ascending=False).index)
        print(f"  {'team':22s} {'P(top8)':>18s} {'P(champion)':>18s} {'P(out)':>18s}")
        for row in ranked.head(10).itertuples():
            print(f"  {row.team:22s} {row.p_top8_prev:.3f}->{row.p_top8:.3f} "
                  f"({row.d_p_top8:+.3f}) {row.p_champion_prev:.3f}->{row.p_champion:.3f} "
                  f"({row.d_p_champion:+.3f}) {row.p_eliminated_prev:.3f}->"
                  f"{row.p_eliminated:.3f} ({row.d_p_eliminated:+.3f})")
        frame = moved.drop(columns=[c for c in moved.columns if c.endswith("_prev")])
        frame.to_csv(out, index=False, float_format="%.6f")

    # The next matchday, from the frozen model at the current ratings. This
    # table needs no Monte Carlo - it is the fitted model evaluated at the
    # as-of ratings - so it is identical whatever --iterations says.
    next_matchday = fixtures[fixtures.matchday == matchday + 1]
    written = [out.name]
    if len(next_matchday):
        from ucl.backtest import probabilities_from_rates

        elo_home = next_matchday.home_id.map(base).to_numpy(dtype=float)
        elo_away = next_matchday.away_id.map(base).to_numpy(dtype=float)
        rate_home, rate_away = model.rates_for(
            next_matchday.home_id.to_numpy(), next_matchday.away_id.to_numpy(),
            elo_home, elo_away)
        probabilities = probabilities_from_rates(rate_home, rate_away)
        outcomes = ("home", "draw", "away")
        rows = []
        for i, row in enumerate(next_matchday.itertuples()):
            goals = np.arange(0, 11)
            joint = (poisson.pmf(goals[:, None], rate_home[i])
                     * poisson.pmf(goals[None, :], rate_away[i]))
            h, a = np.unravel_index(joint.argmax(), joint.shape)
            pick = int(probabilities[i].argmax())
            rows.append({
                "season": cfg.label, "matchday": matchday + 1,
                "date": row.date.date(),
                # Club ids, as the evaluation tables use, so the two join.
                "home": row.home_id, "away": row.away_id,
                "p_home": probabilities[i, 0], "p_draw": probabilities[i, 1],
                "p_away": probabilities[i, 2], "pick": outcomes[pick],
                "p_pick": probabilities[i, pick],
                "modal_score": f"{h}-{a}", "p_modal": float(joint.max()),
            })
        preview = pd.DataFrame(rows)
        # One frame drives both the table and the file, so they cannot disagree.
        print(f"\n=== matchday {matchday + 1}: {len(preview)} fixtures ===")
        print(f"  {'home':22s} {'away':22s} {'home':>6s} {'draw':>6s} {'away':>6s}"
              f"  {'likely':>8s} {'P':>6s}")
        for row in preview.itertuples():
            print(f"  {row.home:22s} {row.away:22s} {row.p_home:6.3f} "
                  f"{row.p_draw:6.3f} {row.p_away:6.3f}  "
                  f"{row.modal_score:<8s} {row.p_modal:6.3f}")
        print("  (the modal scoreline carries only ~8-12% probability; it is not a prediction)")
        # Deliberately not renamed by --out: the forecast may be a scratch run,
        # this table is deterministic and is what the app reads.
        preview_path = OUTPUTS_DIR / f"next_matchday_{matchday + 1}_{cfg.label}.csv"
        preview.to_csv(preview_path, index=False, float_format="%.6f")
        written.append(preview_path.name)

    print(f"\nsetup {setup:.1f}s, simulation {elapsed:.1f}s "
          f"({elapsed / n * 1000:.2f} ms/iteration), written to {', '.join(written)}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
