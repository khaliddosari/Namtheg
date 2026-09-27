"""Forecast accuracy metrics and naive baselines. numpy only, so the backend
can score baselines without any deep-learning library.

MASE is the selection metric: error relative to a seasonal-naive forecast's
in-sample error, so it is comparable across series of any scale. MASE < 1
beats the naive benchmark; MASE >= 1 means the model adds nothing.
"""
import numpy as np


def mase_scale(insample: np.ndarray, season: int) -> float:
    insample = np.asarray(insample, dtype=np.float64)
    m = season if len(insample) > season else 1
    diffs = np.abs(insample[m:] - insample[:-m])
    scale = float(diffs.mean()) if len(diffs) else 0.0
    if scale <= 1e-12:  # flat history: fall back to one-step differences, then to 1
        diffs = np.abs(np.diff(insample))
        scale = float(diffs.mean()) if len(diffs) and diffs.mean() > 1e-12 else 1.0
    return scale


def evaluate(actual, pred, insample, season: int) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = actual - pred
    denom = np.abs(actual) + np.abs(pred)
    smape = np.where(denom > 1e-12, 2 * np.abs(err) / np.where(denom > 1e-12, denom, 1), 0.0)
    return {
        "mase": float(np.abs(err).mean() / mase_scale(insample, season)),
        "smape": float(100 * smape.mean()),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
    }


def aggregate(results: list[dict]) -> dict:
    """Mean of each metric over (series x backtest window) evaluations."""
    return {k: round(float(np.mean([r[k] for r in results])), 4) for k in results[0]} if results else {}


def naive(insample, horizon: int) -> np.ndarray:
    return np.repeat(float(np.asarray(insample)[-1]), horizon)


def seasonal_naive(insample, horizon: int, season: int) -> np.ndarray:
    insample = np.asarray(insample, dtype=np.float64)
    if season <= 1 or len(insample) < season:
        return naive(insample, horizon)
    last = insample[-season:]
    return np.array([last[h % season] for h in range(horizon)])


def backtest_cuts(length: int, horizon: int, windows: int) -> list[int]:
    """Training-end indices for rolling-origin evaluation, oldest first; each
    window's test span is values[cut:cut + horizon]."""
    return [length - horizon * w for w in range(windows, 0, -1)]
