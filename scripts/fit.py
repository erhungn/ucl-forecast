"""Fit the Elo ratings and the baseline Poisson match model at a season's cutoff.

Prints the three diagnostics that say whether the ratings are believable before
anything is simulated: the fitted match-model parameters, the strongest clubs,
and the mean rating per association to compare against the UEFA country ranking.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ucl.config import OUTPUTS_DIR, season as season_config
from ucl.models.elo import elo_as_of
from ucl.models.poisson import fit_poisson


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--associations", type=int, default=25)
    args = parser.parse_args(argv)
    cfg = season_config(args.season)

    print(f"season {cfg.label}, cutoff {cfg.cutoff_date}")
    ratings = elo_as_of(cfg.cutoff_date)
    model = fit_poisson(ratings.pre_match)

    latest = ratings.pre_match.date.max().date()
    assert latest < cfg.cutoff_date, f"a match dated {latest} was used for fitting"
    print(f"matches {ratings.diagnostics['matches']}, clubs {ratings.diagnostics['clubs']}, "
          f"last match used {latest}")
    print(f"association offsets: {ratings.diagnostics['offset_associations_fitted']} fitted "
          f"from {ratings.diagnostics['offset_matches']} European matches, "
          f"draw parameter {ratings.diagnostics['offset_draw_parameter']:.3f}, "
          f"converged {ratings.diagnostics['offset_converged']}")

    print(f"\n=== Poisson match model (log lambda = a + h + b*d, d = elo diff / 400) ===")
    print(f"  a = {model.a:.5f}   (baseline {pd.Series([model.a]).apply('exp').iloc[0]:.3f} goals)")
    print(f"  h = {model.h:.5f}   (home advantage)")
    print(f"  b = {model.b:.5f}   (Elo slope)")
    print(f"  fitted on {model.matches} matches, converged {model.converged}, "
          f"log-likelihood {model.log_likelihood:.1f}")
    equal_home, equal_away = model.rates(1800.0, 1800.0)
    print(f"  equal clubs: {equal_home:.2f} - {equal_away:.2f} "
          f"(total {equal_home + equal_away:.2f})")

    print(f"\n=== top {args.top} Elo at the cutoff ===")
    top = ratings.top(args.top)
    top.index = range(1, len(top) + 1)
    print(top.to_string(float_format=lambda v: f"{v:8.1f}"))

    print(f"\n=== mean Elo per association (top {args.associations}) ===")
    means = ratings.association_means().head(args.associations)
    means.index = range(1, len(means) + 1)
    print(means.to_string(float_format=lambda v: f"{v:8.1f}"))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {"club": list(ratings.ratings), "elo": list(ratings.ratings.values())}
    )
    frame["association"] = frame.club.map(ratings.association_of)
    frame.sort_values("elo", ascending=False).to_csv(
        OUTPUTS_DIR / f"elo_{cfg.label}.csv", index=False)
    ratings.association_means().to_csv(
        OUTPUTS_DIR / f"association_elo_{cfg.label}.csv", index=False)
    with open(OUTPUTS_DIR / f"poisson_{cfg.label}.json", "w", encoding="utf-8") as handle:
        json.dump({**model.as_dict(), "matches": model.matches,
                   "log_likelihood": model.log_likelihood}, handle, indent=2)
        handle.write("\n")
    print(f"\nwritten to outputs/elo_{cfg.label}.csv, "
          f"outputs/association_elo_{cfg.label}.csv, outputs/poisson_{cfg.label}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
