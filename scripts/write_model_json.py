"""Write a published model's coefficient file: ``results/model_<m>/model_<season>.json``.

That file is what ``tests/test_promotion.py`` reads to confirm an archive holds
the model it claims to, and what ``docs/METHODOLOGY.md`` quotes its coefficient
table from. It was assembled by hand for v1 and v2, which is fine exactly once
and is why this script exists: a promotion should not depend on remembering
which fields a reader expects, and the fields are not all in ``as_dict()`` -
the coverage shares and the club counts come from the fixture list and the
covariates themselves.

The gate on this script is that it reproduces the archive it did not write:

    python scripts/write_model_json.py --model v2 --check

``--model v1 --check`` does **not** pass, and should not: that file predates the
multi-covariate schema, so it has a scalar ``fixture_coverage`` instead of one
share per covariate, a ``league_phase_clubs_valued`` field that was dropped, and
no ``model`` or ``covariates`` key. Every number it does carry agrees. It is the
record of what v1 published and is left exactly as it was written.

``--check`` writes nothing and compares instead, so the writer is verified
against the archives it did not produce before it is trusted with a new one.

**It compares numerically, not as text, because the fit is not bit-reproducible
between processes.** ``ucl.models.gap.replay`` averages the relegated clubs'
ratings over a **set** of club names (``gap.py:183``) and seeds the rating dict
in set order (``gap.py:194``), so ``PYTHONHASHSEED`` changes the summation
order and the fitted coefficients move in the eleventh significant digit. Two
runs in one process agree exactly; two processes agree to about 1e-11. Pinning
``PYTHONHASHSEED=0`` makes it exact, which is how that diagnosis was confirmed.
That noise is far below anything published - the forecast is written to six
decimals from 50,000 integer counts - but it is why this script asserts a
tolerance rather than equality, and the tolerance is reported so a real change
cannot hide under it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ucl.config import PROJECT_ROOT, season as season_config
from ucl.data import names, transfermarkt
from ucl.data import fixtures as fx
from ucl.models.elo import elo_as_of
from ucl.models.forecast import MODELS as forecast_models
from ucl.models.forecast import fit as fit_forecast
from ucl.webapp import paths

import pandas as pd


def coefficients(season_label: str, model_label: str) -> dict:
    """Everything the archive's ``model_<season>.json`` records, in file order.

    The model is fitted at the season's own cutoff - this is the frozen
    pre-tournament fit, the one the archive is of, not the current as-of one.
    """
    cfg = season_config(season_label)
    ratings = elo_as_of(cfg.cutoff_date)
    value_as_of = transfermarkt.as_of_for_season(cfg.label)
    values = transfermarkt.squad_values(value_as_of)
    model = fit_forecast(model_label, ratings.pre_match, values, ratings.ratings,
                         as_of=cfg.cutoff_date, value_as_of=value_as_of)

    # The same club ids the simulator joins on, so the coverage shares are of
    # the fixtures the forecast actually runs rather than of raw page spellings.
    fixtures = pd.read_csv(fx.fixtures_path(cfg.label))
    lookup = names.alias_to_canonical("wikipedia")
    home = fixtures.home.map(lookup).fillna(fixtures.home).to_numpy()
    away = fixtures.away.map(lookup).fillna(fixtures.away).to_numpy()
    coverage = model.coverage(home, away)

    out: dict = dict(model.as_dict())
    out["model"] = model_label
    out["matches"] = int(model.model.matches)
    out["converged"] = bool(model.model.converged)
    out["log_likelihood"] = float(model.model.log_likelihood)
    out["covariates"] = list(model.names)
    out["value_as_of"] = str(value_as_of)
    residuals = model.details.get("residuals")
    if residuals is not None:
        out["gap_as_of"] = str(model.details["gap_as_of"])
        out["gap_params"] = {
            measure: [float(v) for v in params.as_tuple()]
            for measure, params in model.details["gap_params"].items()}
    out["clubs_valued"] = len(values)
    if residuals is not None:
        rated: set[str] = set()
        for table in residuals.values():
            rated |= set(table)
        out["clubs_rated_for_shots"] = len(rated)
    out["fixture_coverage"] = {name: float(share)
                               for name, share in coverage.items()}
    if model.level is not None:
        # Only a model with per-competition intercepts has one, and then it is
        # not optional: the same coefficients forecast a different competition
        # at a different level, so the archive has to say which.
        out["level"] = model.level
    return out


def _shorten(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def compare(have: dict, want: dict, *, tolerance: float = 1e-8) -> list[str]:
    """Field-by-field differences, floats within ``tolerance`` relative.

    Non-numeric fields - the label, the covariate list, ``converged`` - are
    compared exactly: those are the ones a wrong archive would get wrong, and
    none of them is subject to the summation-order noise above.
    """
    out = []
    for key in sorted(set(have) | set(want)):
        if key not in have:
            out.append(f"{key}: missing on disk, recomputed {want[key]!r}")
            continue
        if key not in want:
            out.append(f"{key}: on disk {have[key]!r}, not recomputed")
            continue
        mine, theirs = have[key], want[key]
        if isinstance(mine, bool) or isinstance(theirs, bool):
            if mine != theirs:
                out.append(f"{key}: on disk {mine!r} -> recomputed {theirs!r}")
        elif isinstance(mine, (int, float)) and isinstance(theirs, (int, float)):
            scale = max(abs(mine), abs(theirs), 1e-12)
            relative = abs(mine - theirs) / scale
            if relative > tolerance:
                out.append(f"{key}: on disk {mine!r} -> recomputed {theirs!r} "
                           f"({relative:.2e} relative)")
        elif isinstance(mine, dict) and isinstance(theirs, dict):
            nested = compare(mine, theirs, tolerance=tolerance)
            out.extend(f"{key}.{line}" for line in nested)
        elif isinstance(mine, list) and isinstance(theirs, list):
            if len(mine) != len(theirs):
                out.append(f"{key}: {len(mine)} entries on disk, "
                           f"{len(theirs)} recomputed")
            else:
                nested = compare(dict(enumerate(mine)), dict(enumerate(theirs)),
                                 tolerance=tolerance)
                out.extend(f"{key}[{line}" .replace(":", "]:", 1)
                           for line in nested)
        elif mine != theirs:
            out.append(f"{key}: on disk {mine!r} -> recomputed {theirs!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=paths.SEASON)
    parser.add_argument("--model", choices=[m for m in forecast_models if m != "elo"],
                        required=True)
    parser.add_argument("--check", action="store_true",
                        help="compare with the file on disk instead of writing it")
    parser.add_argument("--out", default=None,
                        help="write here instead of results/model_<model>/")
    parser.add_argument("--tolerance", type=float, default=1e-8,
                        help="relative tolerance for --check; the between-"
                             "process noise is about 1e-11, so 1e-8 passes it "
                             "and would catch any real change")
    args = parser.parse_args(argv)

    written = coefficients(args.season, args.model)
    text = json.dumps(written, indent=2) + "\n"
    target = Path(args.out) if args.out else (
        paths.archive_dir(args.model) / f"model_{args.season}.json")

    shown = _shorten(target)
    if args.check:
        if not target.exists():
            print(f"{shown} does not exist, so there is nothing to check")
            return 1
        differences = compare(json.loads(target.read_text(encoding="utf-8")),
                           written, tolerance=args.tolerance)
        if not differences:
            print(f"{shown} reproduced ({len(written)} fields, every number "
                  f"within {args.tolerance:g} relative)")
            return 0
        print(f"{shown} DIFFERS in {len(differences)} field(s)")
        for line in differences:
            print(f"  {line}")
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"written to {shown} ({len(written)} fields)")
    for key, value in written.items():
        if isinstance(value, float):
            print(f"  {key:26} {value:+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
