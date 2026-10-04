"""Compare the forecast against every outright market held for a season.

Measurement, not a test. The market never enters an objective; it is read
alongside the forecast to show where the model disagrees with informed opinion.

Two markets are quoted differently and must be read differently:

* a **bookmaker** builds a large margin into every price - 30.8% here - so the
  quoted prices have to be de-vigged before they mean anything, and
  proportional normalisation is the crude way to do it;
* a **prediction exchange** quotes in cents that already are probabilities, so
  the raw price is the estimate. Renormalising an exchange's prices over the
  subset it happens to have quoted would inflate every one of them, because the
  clubs it left out had probability too.

So this reports both readings for both markets, and says which one to trust for
each. A price quoted only as a bound ("under 1 cent") is listed but carries no
number and is left out of the arithmetic.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ucl.config import OUTPUTS_DIR, season as season_config
from ucl.data import odds
from ucl.webapp import paths as live_paths

#: Below this many usable prices a market is too thin to serve as the headline
#: benchmark - declared before the prices were fetched.
MINIMUM_PRICED = 15


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--matchday", type=int, default=None,
                        help="which published forecast to compare (default: matchday 0)")
    parser.add_argument("--out", default="market_comparison.csv")
    args = parser.parse_args(argv)
    cfg = season_config(args.season)

    matchday = 0 if args.matchday is None else args.matchday
    forecast = pd.read_csv(live_paths.forecast_path(matchday, cfg.label)) \
        if live_paths.forecast_path(matchday, cfg.label).exists() \
        else pd.read_csv(live_paths.ARCHIVE_DIR / f"forecast_{cfg.label}.csv")
    model = forecast[["team", "p_champion"]]

    # The benchmark is held to the model's own rule: a price observed on or
    # after the cutoff has seen matchday 1.
    market = odds.load_all_outrights(cfg.label, cutoff=cfg.cutoff_date)

    print(f"=== outright markets held for {cfg.label} "
          f"(cutoff {cfg.cutoff_date}) ===")
    for name, info in market.attrs["markets"].items():
        dates = sorted(set(market.loc[market.market == name, "odds_date"]))
        enough = "" if info["priced"] >= MINIMUM_PRICED else \
            f"  <- under {MINIMUM_PRICED} prices: too thin to be the headline benchmark"
        print(f"  {name}")
        print(f"    file {info['path']}, observed {', '.join(dates)}")
        print(f"    {info['priced']} usable price(s) of {info['listed']} listed, "
              f"{36 - info['listed']} of the 36 clubs not listed at all{enough}")
        print(f"    sum of raw implied probabilities over the priced clubs: "
              f"{info['overround']:.4f}")

    pieces = []
    for name, group in market.groupby("market"):
        joined = model.merge(group[["team", "p_raw", "p_market"]], on="team",
                             how="inner")
        joined["market"] = name
        joined["model_minus_raw"] = joined.p_champion - joined.p_raw
        joined["model_over_raw"] = joined.p_champion / joined.p_raw
        joined["model_minus_devigged"] = joined.p_champion - joined.p_market
        pieces.append(joined)
    comparison = pd.concat(pieces, ignore_index=True)

    for name, group in comparison.groupby("market"):
        ranked = group.reindex(group.model_minus_raw.abs()
                               .sort_values(ascending=False).index)
        print(f"\n=== model vs {name} ({len(group)} clubs) ===")
        print(f"  {'club':22s} {'model':>7s} {'raw':>7s} {'de-vig':>7s} "
              f"{'model/raw':>10s}")
        for row in ranked.itertuples():
            print(f"  {row.team:22s} {row.p_champion:7.3f} {row.p_raw:7.3f} "
                  f"{row.p_market:7.3f} {row.model_over_raw:10.2f}")

    # Where the two markets overlap, do they agree with each other?
    wide = comparison.pivot_table(index="team", columns="market",
                                  values="p_raw")
    both = wide.dropna()
    if len(both.columns) > 1 and len(both):
        print(f"\n=== the two markets against each other "
              f"({len(both)} clubs priced by both) ===")
        left, right = both.columns[:2]
        print(f"  {'club':22s} {left[:18]:>18s} {right[:18]:>18s} {'ratio':>8s}")
        for club, row in both.sort_values(left, ascending=False).iterrows():
            print(f"  {club:22s} {row[left]:18.3f} {row[right]:18.3f} "
                  f"{row[left] / row[right]:8.2f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / args.out
    comparison.to_csv(out, index=False, float_format="%.6f")
    print(f"\nwritten to {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
