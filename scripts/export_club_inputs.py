"""Export the model's per-club inputs, with their coverage, for the app.

These are the frozen pre-cutoff covariates: the squad-value prior and the GAP
shot-rating residuals, beside the Elo they were measured against. They existed
only as two hand-written CSVs inside ``results/model_v1`` and
``results/model_v2`` with no script behind them; this is that script, so the
numbers the club page shows can be regenerated rather than trusted.

Nothing here is new information and nothing is fitted: it calls exactly what
``scripts/simulate.py`` calls, at the same cutoff.

    .venv/bin/python scripts/export_club_inputs.py --season 2026-27

The ``elo`` column is the rating **at the cutoff**, which is what the
covariates were built against. It is not the rating the forecast currently runs
on - that moves with every played matchday and is published per matchday as
``ratings_md<k>.csv``. The app must not label the two the same way.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ucl.config import OUTPUTS_DIR, season as season_config
from ucl.data import fixtures as fx
from ucl.data import names, transfermarkt
from ucl.models.elo import elo_as_of
from ucl.models.forecast import gap_residuals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = season_config(args.season)
    cutoff = cfg.cutoff_date
    print(f"{cfg.label}: inputs at the {cutoff} cutoff")

    fixtures = pd.read_csv(fx.fixtures_path(cfg.label))
    lookup = names.alias_to_canonical("wikipedia")
    teams = sorted(set(fixtures.home.map(lookup).fillna(fixtures.home))
                   | set(fixtures.away.map(lookup).fillna(fixtures.away)))

    ratings = elo_as_of(cutoff)
    value_as_of = transfermarkt.as_of_for_season(cfg.label)
    values = transfermarkt.squad_values(value_as_of)
    residuals, _params = gap_residuals(cutoff, ratings.ratings)

    rows = []
    for team in teams:
        value = values.get(team)
        row = {
            "team": team,
            "elo_at_cutoff": ratings.ratings.get(team),
            "squad_value_eur_m": None if value is None else value / 1e6,
        }
        covered = False
        for measure, per_club in residuals.items():
            rated = per_club.get(team)
            # Home and away are averaged, which is the convention the archived
            # table already used; the two differ by about a tenth of a shot.
            for side in ("attack", "defence"):
                row[f"{measure}_{side}"] = None if rated is None else (
                    (rated[f"home_{side}"] + rated[f"away_{side}"]) / 2)
            covered = covered or rated is not None
        row["value_covered"] = value is not None
        row["shots_covered"] = covered
        rows.append(row)

    frame = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / (args.out or "club_inputs.csv")
    frame.to_csv(out, index=False, float_format="%.6f")

    print(f"  {len(frame)} clubs, value as of {value_as_of}")
    print(f"  squad value: {int(frame.value_covered.sum())}/36 covered"
          + ("" if frame.value_covered.all() else
             "; missing " + ", ".join(frame.loc[~frame.value_covered, "team"])))
    print(f"  shot ratings: {int(frame.shots_covered.sum())}/36 covered"
          + ("" if frame.shots_covered.all() else
             "; missing " + ", ".join(frame.loc[~frame.shots_covered, "team"])))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
