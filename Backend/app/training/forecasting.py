"""Time-series forecasting on the GPU. One call evaluates one candidate; the
backend runs candidates in parallel on separate GPUs.

Input is a regular table (series_id, timestamp, value) from
app/pipeline/timeseries.py. Candidates:
- Chronos-2: pretrained forecasting model, used zero-shot (no training).
- LSTM and GRU (recurrent networks) and a TCN (dilated causal 1-D CNN):
  global models trained on sliding windows from every series.
- XGBoost on standardised lag windows and calendar features.

Scoring is rolling-origin backtesting: for each of the last `windows`
horizons the model sees only data before the cut and forecasts the next
`horizon` steps; trainable models are refit per window, so no window's future
reaches its training data. The 80% interval is symmetric, from the 80th
percentile of the model's own absolute backtest errors at each step (in
standardised units): with a handful of backtest windows, separate lower and
upper quantiles are too noisy and can even exclude the forecast itself.
"""
import io
import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd

from app.training import tsmetrics
from app.training.common import RANDOM_STATE, library_versions
from app.training.runtime import BUNDLE_FORMAT, build_net, input_window, lag_features, window_stats

CANDIDATES = ("Chronos-2", "LSTM", "GRU", "TCN", "XGBoost (lags)")
NETS = {"LSTM": "lstm", "GRU": "gru", "TCN": "tcn"}
CHRONOS_MODEL = "amazon/chronos-2"
MAX_EPOCHS = 60
PATIENCE = 6
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MAX_TRAIN_WINDOWS = 200_000
XGB_MAX_TREES = 2000
HISTORY_KEPT = 512
INTERVAL_COVERAGE = 0.8


Series = dict[str, tuple[pd.DatetimeIndex, np.ndarray]]


def load_series(data: bytes) -> Series:
    df = pd.read_parquet(io.BytesIO(data))
    return {str(sid): (pd.DatetimeIndex(g["timestamp"]), g["value"].to_numpy(dtype=np.float64))
            for sid, g in df.sort_values(["series_id", "timestamp"]).groupby("series_id", sort=True)}


def training_windows(values: list[np.ndarray], L: int, H: int, rng) -> tuple[np.ndarray, np.ndarray, list]:
    """All (input, target) windows of length L and H from each series, subsampled
    to MAX_TRAIN_WINDOWS. Also returns (series index, start) for timestamps."""
    xs, ys, where = [], [], []
    for i, v in enumerate(values):
        n = len(v) - L - H + 1
        if n <= 0:
            continue
        w = np.lib.stride_tricks.sliding_window_view(v, L + H)[:n]
        xs.append(w[:, :L])
        ys.append(w[:, L:])
        where += [(i, s) for s in range(n)]
    if not xs:
        raise ValueError(f"No series is long enough for {L} input steps plus a {H}-step horizon.")
    X, Y = np.concatenate(xs), np.concatenate(ys)
    if len(X) > MAX_TRAIN_WINDOWS:
        keep = np.sort(rng.choice(len(X), MAX_TRAIN_WINDOWS, replace=False))
        X, Y, where = X[keep], Y[keep], [where[k] for k in keep]
    return X, Y, where


# -- neural networks -------------------------------------------------------------

def fit_net(arch: str, values: list[np.ndarray], L: int, H: int, device: str, rng):
    import torch

    torch.manual_seed(RANDOM_STATE)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    X, Y, _ = training_windows(values, L, H, rng)
    mu, sd = window_stats(X)
    Xn = torch.tensor((X - mu[:, None]) / sd[:, None], dtype=torch.float32, device=device)
    Yn = torch.tensor((Y - mu[:, None]) / sd[:, None], dtype=torch.float32, device=device)
    order = rng.permutation(len(Xn))
    n_val = max(1, len(order) // 10)
    val, train = order[:n_val], order[n_val:] if len(order) > n_val else order

    net = build_net(arch, H).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss()
    best, best_state, stale = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        shuffled = rng.permutation(train)
        for start in range(0, len(shuffled), BATCH_SIZE):
            idx = torch.as_tensor(shuffled[start:start + BATCH_SIZE], device=device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(Xn[idx]), Yn[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(Xn[val]), Yn[val]))
        if v < best - 1e-5:
            best, stale = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net.eval()


def predict_net(net, contexts: list[np.ndarray], L: int, device: str) -> list[np.ndarray]:
    import torch

    W = np.stack([input_window(c, L) for c in contexts])
    mu, sd = window_stats(W)
    with torch.no_grad():
        x = torch.tensor((W - mu[:, None]) / sd[:, None], dtype=torch.float32, device=device)
        out = net(x).float().cpu().numpy()
    return list(out * sd[:, None] + mu[:, None])


# -- gradient boosting on lags ---------------------------------------------------------

def fit_xgb(parts: list[tuple[pd.DatetimeIndex, np.ndarray]], L: int, H: int, device: str, rng):
    import xgboost as xgb

    X, Y, where = training_windows([v for _, v in parts], L, H, rng)
    n = len(X)
    per_window = max(1, MAX_TRAIN_WINDOWS // H)
    if n > per_window:
        keep = np.sort(rng.choice(n, per_window, replace=False))
        X, Y, where = X[keep], Y[keep], [where[k] for k in keep]
        n = len(X)
    target_times = np.concatenate([parts[i][0][s + L:s + L + H] for i, s in where])
    feats = lag_features(np.repeat(X, H, axis=0), np.tile(np.arange(1, H + 1), n), H, target_times)
    mu, sd = window_stats(X)
    y = ((Y - mu[:, None]) / sd[:, None]).ravel()
    order = rng.permutation(len(y))
    n_val = max(1, len(order) // 10)
    val, train = order[:n_val], order[n_val:]
    model = xgb.XGBRegressor(device=device, tree_method="hist", n_estimators=XGB_MAX_TREES, learning_rate=0.05,
                             max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
                             early_stopping_rounds=50)
    model.fit(feats[train], y[train], eval_set=[(feats[val], y[val])], verbose=False)
    return model


def predict_xgb(model, parts: list[tuple[pd.DatetimeIndex, np.ndarray]], L: int, H: int, freq: str) -> list[np.ndarray]:
    out = []
    for ts, v in parts:
        window = input_window(v, L)
        mu, sd = (float(a[0]) for a in window_stats(window))
        future = pd.date_range(ts[-1], periods=H + 1, freq=freq)[1:]
        feats = lag_features(np.repeat(window[None], H, axis=0), np.arange(1, H + 1), H, future)
        out.append(np.asarray(model.predict(feats), dtype=np.float64) * sd + mu)
    return out


# -- Chronos -------------------------------------------------------------------------

def chronos_pipeline(device: str):
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map=device)


def predict_chronos(pipe, parts: dict[str, tuple[pd.DatetimeIndex, np.ndarray]], H: int) -> dict[str, np.ndarray]:
    frame = pd.concat([pd.DataFrame({"id": sid, "timestamp": ts, "target": v}) for sid, (ts, v) in parts.items()])
    pred = pipe.predict_df(frame, prediction_length=H, quantile_levels=[0.5], id_column="id",
                           timestamp_column="timestamp", target="target")
    return {str(sid): g.sort_values("timestamp")["predictions"].to_numpy(dtype=np.float64)
            for sid, g in pred.groupby("id")}


# -- one candidate ---------------------------------------------------------------------

class Candidate:
    """Fit on a set of series histories, then forecast H steps for each."""

    def __init__(self, name: str, L: int, H: int, freq: str, device: str, rng):
        self.name, self.L, self.H, self.freq, self.device, self.rng = name, L, H, freq, device, rng
        self.model = chronos_pipeline(device) if name == "Chronos-2" else None

    def fit(self, parts: dict[str, tuple[pd.DatetimeIndex, np.ndarray]]):
        if self.name in NETS:
            self.model = fit_net(NETS[self.name], [v for _, v in parts.values()], self.L, self.H, self.device, self.rng)
        elif self.name == "XGBoost (lags)":
            self.model = fit_xgb(list(parts.values()), self.L, self.H, self.device, self.rng)

    def predict(self, parts: dict[str, tuple[pd.DatetimeIndex, np.ndarray]]) -> dict[str, np.ndarray]:
        if self.name == "Chronos-2":
            return predict_chronos(self.model, parts, self.H)
        ids = list(parts)
        if self.name in NETS:
            preds = predict_net(self.model, [parts[s][1] for s in ids], self.L, self.device)
        else:
            preds = predict_xgb(self.model, [parts[s] for s in ids], self.L, self.H, self.freq)
        return dict(zip(ids, preds))

    def estimator_spec(self) -> dict:
        if self.name == "Chronos-2":
            return {"kind": "chronos", "model_id": CHRONOS_MODEL}
        if self.name in NETS:
            import torch

            buf = io.BytesIO()
            torch.save({k: t.cpu() for k, t in self.model.state_dict().items()}, buf)
            return {"kind": "torch", "arch": NETS[self.name], "state_dict": buf.getvalue()}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.json")
            self.model.save_model(path)
            return {"kind": "xgboost-json", "bytes": open(path, "rb").read()}


def run(data: bytes, spec: dict, device: str) -> dict:
    """spec: {"candidate", "horizon", "freq", "season", "input_length", "windows"}"""
    started = time.monotonic()
    name, H, L = spec["candidate"], int(spec["horizon"]), int(spec["input_length"])
    freq, season, windows = spec["freq"], int(spec["season"]), int(spec["windows"])
    series = load_series(data)
    rng = np.random.default_rng(RANDOM_STATE)
    cand = Candidate(name, L, H, freq, device, rng)

    evals, scaled_errors, backtest = [], [], None
    first = next(iter(series))
    for w in range(windows):
        parts, tests = {}, {}
        for sid, (ts, v) in series.items():
            cut = tsmetrics.backtest_cuts(len(v), H, windows)[w]
            if cut >= H + 2:
                parts[sid] = (ts[:cut], v[:cut])
                tests[sid] = v[cut:cut + H]
        if not parts:
            continue
        cand.fit(parts)
        preds = cand.predict(parts)
        for sid, actual in tests.items():
            pred = preds[sid]
            evals.append(tsmetrics.evaluate(actual, pred, parts[sid][1], season))
            sd = float(window_stats(input_window(parts[sid][1], L))[1][0])
            scaled_errors.append((actual - pred) / sd)
        if first in parts:
            ts_first, v_first = series[first]
            cut = len(parts[first][1])
            backtest = {"series_id": first, "timestamps": [t.isoformat() for t in ts_first[cut:cut + H]],
                        "actual": v_first[cut:cut + H].tolist(), "forecast": preds[first].tolist()}
    if not evals:
        raise ValueError("Series are too short to backtest this horizon.")

    cand.fit(series)
    final = cand.predict(series)
    width = np.quantile(np.abs(np.stack(scaled_errors)), INTERVAL_COVERAGE, axis=0)
    lower, upper = -width, width
    rows = []
    for sid, (ts, v) in series.items():
        sd = float(window_stats(input_window(v, L))[1][0])
        future = pd.date_range(ts[-1], periods=H + 1, freq=freq)[1:]
        p = final[sid]
        rows += [{"series_id": sid, "timestamp": t.isoformat(), "forecast": float(p[h]),
                  "lower": float(p[h] + lower[h] * sd), "upper": float(p[h] + upper[h] * sd)}
                 for h, t in enumerate(future)]

    history = {sid: {"timestamps": [t.isoformat() for t in ts[-HISTORY_KEPT:]], "values": v[-HISTORY_KEPT:].tolist()}
               for sid, (ts, v) in series.items()}
    buf = io.BytesIO()
    joblib.dump({
        "format": BUNDLE_FORMAT, "task": "forecasting", "problem_type": "forecasting", "model_name": name,
        "feature_cols": [], "class_labels": None, "estimator": cand.estimator_spec(),
        "forecast": {"horizon": H, "input_length": L, "freq": freq, "season": season, "history": history,
                     "interval": {"lower": lower.tolist(), "upper": upper.tolist(),
                                  "coverage": INTERVAL_COVERAGE}},
    }, buf)
    return {
        "task": "forecasting", "candidate": name, "selection_metric": "mase",
        "metrics": tsmetrics.aggregate(evals), "evaluations": len(evals),
        "forecast": rows, "backtest": backtest,
        "bundle_bytes": buf.getvalue(), "library_versions": library_versions(), "device": device,
        "seconds": round(time.monotonic() - started, 1),
    }
