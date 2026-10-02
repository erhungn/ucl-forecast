"""Score the ratings on every backtest season and compare against the market.

Reports log-loss, RPS, Brier and calibration error per season, for the model in
frozen and walk-forward mode and for de-vigged closing odds on the domestic
matches that carry them.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ucl.backtest import score_season
from ucl.config import BACKTEST_SEASONS, OUTPUTS_DIR
from ucl.models.elo import EloParams


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", action="append", help="repeatable; default all")
    parser.add_argument("--out", default="backtest_summary.csv")
    args = parser.parse_args(argv)
    seasons = args.season or list(BACKTEST_SEASONS)

    frames = []
    for label in seasons:
        score = score_season(label, EloParams())
        frames.append(score.summary)
        print(f"scored {label}: {len(score.rows)} rows")
    summary = pd.concat(frames, ignore_index=True)
    summary = summary[["season", "mode", "competition", "n", "log_loss", "rps", "brier", "ece"]]
    summary = summary.sort_values(["season", "competition", "mode"]).reset_index(drop=True)

    pd.set_option("display.width", 200)
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    beaten = []
    for label in seasons:
        rows = summary[summary.season == label]
        model = rows[(rows["mode"] == "frozen") & (rows.competition == "domestic")]
        market = rows[rows["mode"] == "market"]
        if len(model) and len(market):
            beaten.append(float(market.iloc[0].log_loss) < float(model.iloc[0].log_loss))
    print(f"\nmarket beats the model on domestic matches in "
          f"{sum(beaten)}/{len(beaten)} seasons (it should be all of them)")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUTS_DIR / args.out, index=False)
    print(f"written to outputs/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
