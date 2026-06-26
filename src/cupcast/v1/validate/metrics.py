from __future__ import annotations

import numpy as np
import pandas as pd

# probs: (n, 3) ordered [home win, draw, away win]; outcome: (n,) in {0, 1, 2}.


def log_loss(probs: np.ndarray, outcome: np.ndarray) -> float:
    chosen = probs[np.arange(len(outcome)), outcome]
    return float(-np.mean(np.log(np.clip(chosen, 1e-12, None))))


def brier_score(probs: np.ndarray, outcome: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcome)), outcome] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ranked_probability_score(probs: np.ndarray, outcome: np.ndarray) -> float:
    # Standard RPS for the ordered (home, draw, away) outcome, Constantinou &
    # Fenton (2012).
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcome)), outcome] = 1.0
    cum_diff = np.cumsum(probs - onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum(cum_diff**2, axis=1) / (probs.shape[1] - 1)))


def summarize(probs: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    return {
        "n": len(outcome),
        "log_loss": log_loss(probs, outcome),
        "brier": brier_score(probs, outcome),
        "rps": ranked_probability_score(probs, outcome),
    }


def calibration_table(probs: np.ndarray, outcome: np.ndarray, bins: int = 10) -> pd.DataFrame:
    # Pool all three outcome probabilities, reliability-diagram style.
    flat_p = probs.ravel()
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcome)), outcome] = 1.0
    flat_y = onehot.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(flat_p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        members = bucket == b
        if not members.any():
            continue
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "n": int(members.sum()),
                "mean_predicted": float(flat_p[members].mean()),
                "observed_rate": float(flat_y[members].mean()),
            }
        )
    return pd.DataFrame(rows)
