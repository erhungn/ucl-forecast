"""Forecast evaluation: log-loss, RPS, Brier and calibration.

All of these take 1X2 probability triples in the order (home, draw, away) and
integer outcomes 0 = home win, 1 = draw, 2 = away win, so the model and the
bookmaker go through exactly the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["log_loss", "rps", "brier", "calibration", "summarise"]

EPSILON = 1e-15


def _prepare(probabilities, outcomes) -> tuple[np.ndarray, np.ndarray]:
    # Normalise first, then clip: clipping to 1 beforehand would turn an
    # unnormalised row like (2, 1, 1) into a uniform forecast.
    p = np.asarray(probabilities, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    p = np.clip(p, EPSILON, 1.0)
    return p / p.sum(axis=1, keepdims=True), np.asarray(outcomes, dtype=int)


def log_loss(probabilities, outcomes, weights=None) -> float:
    """Mean negative log probability of the outcome that happened."""
    p, y = _prepare(probabilities, outcomes)
    losses = -np.log(p[np.arange(len(y)), y])
    if weights is None:
        return float(losses.mean())
    w = np.asarray(weights, dtype=float)
    return float((losses * w).sum() / w.sum())


def rps(probabilities, outcomes, weights=None) -> float:
    """Ranked probability score, which penalises being wrong by more.

    1X2 is ordered home > draw > away, so predicting an away win when the home
    team wins costs more than predicting a draw.
    """
    p, y = _prepare(probabilities, outcomes)
    observed = np.zeros_like(p)
    observed[np.arange(len(y)), y] = 1.0
    cumulative = np.cumsum(p, axis=1)[:, :-1] - np.cumsum(observed, axis=1)[:, :-1]
    scores = (cumulative ** 2).sum(axis=1) / (p.shape[1] - 1)
    if weights is None:
        return float(scores.mean())
    w = np.asarray(weights, dtype=float)
    return float((scores * w).sum() / w.sum())


def brier(probabilities, outcomes) -> float:
    """Multi-class Brier score."""
    p, y = _prepare(probabilities, outcomes)
    observed = np.zeros_like(p)
    observed[np.arange(len(y)), y] = 1.0
    return float(((p - observed) ** 2).sum(axis=1).mean())


def calibration(probabilities, outcomes, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Reliability of every predicted probability, pooled over the three outcomes.

    Returns the per-bin table and the expected calibration error.
    """
    p, y = _prepare(probabilities, outcomes)
    observed = np.zeros_like(p)
    observed[np.arange(len(y)), y] = 1.0
    flat_p = p.ravel()
    flat_y = observed.ravel()

    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(flat_p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "low": edges[b],
                "high": edges[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(flat_p[mask].mean()),
                "observed": float(flat_y[mask].mean()),
            }
        )
    table = pd.DataFrame(rows)
    error = float(
        (table.n * (table.mean_predicted - table.observed).abs()).sum() / len(flat_p)
    )
    return table, error


def summarise(probabilities, outcomes, label: str = "") -> dict:
    table, ece = calibration(probabilities, outcomes)
    return {
        "label": label,
        "n": int(len(outcomes)),
        "log_loss": log_loss(probabilities, outcomes),
        "rps": rps(probabilities, outcomes),
        "brier": brier(probabilities, outcomes),
        "ece": ece,
    }
