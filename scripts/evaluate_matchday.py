"""Score the frozen pre-tournament forecast on a matchday that has been played.

The probabilities are the ones that stood at the season's cutoff - the model is
not re-derived with hindsight. A uniform (1/3, 1/3, 1/3) forecast is reported
alongside for scale.

There is deliberately no market comparison: football-data covers domestic
leagues only, and nothing in the cache carries match-level European prices.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ucl.backtest import probabilities_from_rates
from ucl.config import OUTPUTS_DIR, season as season_config
from ucl.data import league_results, transfermarkt
from ucl.evaluate import log_loss, rps
from ucl.models.forecast import fit as fit_forecast
from ucl.models.elo import elo_as_of
from ucl.models.poisson import fit_poisson

RESULT_NAMES = {0: "home", 1: "draw", 2: "away"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--matchday", type=int, default=1)
    parser.add_argument("--model", choices=("v2", "v1", "elo"), default="v2",
                        help="v2: v1 plus the GAP shot ratings; v1: Elo plus the "
                             "squad-value prior; elo: the frozen baseline")
    args = parser.parse_args(argv)
    cfg = season_config(args.season)

    played = league_results.load_played(cfg.label)
    played = played[played.matchday == args.matchday]
    if played.empty:
        raise SystemExit(f"no played matches on matchday {args.matchday}")

    # The information set is the cutoff: this is the pre-tournament forecast.
    ratings = elo_as_of(cfg.cutoff_date)
    if args.model in ("v1", "v2"):
        value_as_of = transfermarkt.as_of_for_season(cfg.label)
        model = fit_forecast(
            args.model, ratings.pre_match,
            transfermarkt.squad_values(value_as_of), ratings.ratings,
            as_of=cfg.cutoff_date, value_as_of=value_as_of)
    else:
        model = fit_poisson(ratings.pre_match)
    missing = [c for c in set(played.home_id) | set(played.away_id) if c not in ratings.ratings]
    if missing:
        raise SystemExit(f"no rating for {missing}")

    elo_home = played.home_id.map(ratings.ratings).to_numpy(dtype=float)
    elo_away = played.away_id.map(ratings.ratings).to_numpy(dtype=float)
    probabilities = probabilities_from_rates(*model.rates_for(
        played.home_id.to_numpy(), played.away_id.to_numpy(), elo_home, elo_away))
    outcomes = np.where(played.hg > played.ag, 0, np.where(played.hg == played.ag, 1, 2))

    per_match = -np.log(np.clip(probabilities[np.arange(len(played)), outcomes], 1e-15, 1))
    frame = pd.DataFrame({
        "date": played.date.dt.date.to_numpy(), "home": played.home_id.to_numpy(),
        "away": played.away_id.to_numpy(),
        "score": [f"{h}-{a}" for h, a in zip(played.hg, played.ag)],
        "p_home": probabilities[:, 0], "p_draw": probabilities[:, 1],
        "p_away": probabilities[:, 2],
        "result": [RESULT_NAMES[o] for o in outcomes],
        "p_actual": probabilities[np.arange(len(played)), outcomes],
        "log_loss": per_match,
        "rps": [rps(probabilities[i:i + 1], outcomes[i:i + 1]) for i in range(len(played))],
    })

    print(f"=== {cfg.label} matchday {args.matchday}: {len(frame)} matches, "
          f"forecast frozen at {cfg.cutoff_date}, model {args.model} ===\n")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    uniform = np.full_like(probabilities, 1 / 3)
    print(f"\n=== aggregate ===")
    print(f"  {'':10s} {'log-loss':>9s} {'RPS':>8s}")
    print(f"  {'model':10s} {log_loss(probabilities, outcomes):9.4f} "
          f"{rps(probabilities, outcomes):8.4f}")
    print(f"  {'uniform':10s} {log_loss(uniform, outcomes):9.4f} "
          f"{rps(uniform, outcomes):8.4f}")
    standard_error = per_match.std(ddof=1) / np.sqrt(len(per_match))
    print(f"\n  standard error on the model's log-loss: {standard_error:.3f} over "
          f"{len(frame)} matches")
    print("  This is anecdote, not evidence. That standard error is roughly a hundred")
    print("  times the differences the five-season backtest was built to resolve, so")
    print("  nothing here distinguishes this model from any other sensible one.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / f"matchday_{args.matchday}_evaluation_{cfg.label}.csv"
    frame.to_csv(out, index=False, float_format="%.6f")
    print(f"\nwritten to {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
