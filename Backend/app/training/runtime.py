"""Load and run any model Namtheg trains.

Used by the inference endpoint (app/deploy/inference_app.py) and copied
verbatim as runtime.py into every downloadable model package, so it must stay
self-contained: numpy and pandas at the top, everything else imported only by
the task that needs it (a tabular model never needs torch).

Models are stored in portable forms, never as library pickles that break
across builds: XGBoost JSON, CatBoost .cbm, torch state_dicts, plain numpy
arrays, and scikit-learn objects (portable at the pinned version).
"""
import io

import numpy as np
import pandas as pd

BUNDLE_FORMAT = "namtheg-bundle-v2"
LEGACY_FORMAT = "namtheg-portable-v1"


def device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# -- free text -------------------------------------------------------------------

_ENCODERS: dict = {}


def encode_text(values, encoder_name: str, batch_size: int = 128) -> np.ndarray:
    """Unit-normalised sentence embeddings from a pretrained encoder."""
    if encoder_name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer

        _ENCODERS[encoder_name] = SentenceTransformer(encoder_name, device=device())
    texts = ["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in values]
    emb = _ENCODERS[encoder_name].encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                         show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def text_feature_names(column: str, k: int) -> list[str]:
    return [f"{column}__text{i}" for i in range(k)]


def attach_text_features(X: pd.DataFrame, column: str, Z: np.ndarray) -> pd.DataFrame:
    """Drop a text column and append its projected embedding. Training and
    prediction both go through here, so columns come out in the same order
    (scikit-learn rejects reordered columns)."""
    X = X.drop(columns=[column])
    features = pd.DataFrame(np.asarray(Z, dtype=np.float32), index=X.index,
                            columns=text_feature_names(column, Z.shape[1]))
    return pd.concat([X, features], axis=1)


def apply_text(X: pd.DataFrame, text: dict | None) -> pd.DataFrame:
    """Replace each free-text column with its embedding, projected onto the
    principal components computed at training time."""
    if not text:
        return X
    for column, proj in text["columns"].items():
        emb = encode_text(X[column].tolist(), text["encoder"])
        Z = (emb - np.asarray(proj["mean"], np.float32)) @ np.asarray(proj["components"], np.float32).T
        X = attach_text_features(X, column, Z)
    return X


# -- estimators ------------------------------------------------------------------

def load_estimator(spec: dict, task: str):
    kind = spec["kind"]
    classification = task == "classification"
    if kind == "xgboost-json":
        import xgboost

        model = xgboost.XGBClassifier() if classification else xgboost.XGBRegressor()
        model.load_model(bytearray(spec["bytes"]))
        return model
    if kind == "catboost-cbm":
        import catboost

        model = catboost.CatBoostClassifier() if classification else catboost.CatBoostRegressor()
        model.load_model(blob=spec["bytes"])
        return model
    if kind == "sklearn":
        return spec["object"]
    raise ValueError(f"Unknown estimator kind {kind!r}")


class TabularModel:
    """Classification/regression: raw columns in, predictions out."""

    def __init__(self, bundle: dict):
        self.feature_cols = bundle["feature_cols"]
        self.preprocessor = bundle["preprocessor"]
        self.text = bundle.get("text")
        self.estimator = load_estimator(bundle["estimator"], bundle["task"])
        if hasattr(self.estimator, "predict_proba"):
            self.predict_proba = self._predict_proba

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(apply_text(X[self.feature_cols], self.text))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # ravel: CatBoost multiclass predict returns shape (n, 1).
        return np.asarray(self.estimator.predict(self.transform(X))).ravel()

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(self.transform(X)))


def nearest(points: np.ndarray, refs: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Index of, and Euclidean distance to, the nearest row of `refs` for each point."""
    ref_sq = (refs ** 2).sum(axis=1)
    idx = np.empty(len(points), dtype=np.int64)
    dist = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        d2 = (block ** 2).sum(axis=1)[:, None] - 2 * block @ refs.T + ref_sq[None, :]
        j = d2.argmin(axis=1)
        idx[start:start + chunk] = j
        dist[start:start + chunk] = np.sqrt(np.maximum(d2[np.arange(len(block)), j], 0))
    return idx, dist


class ClusterModel:
    """Assigns new rows to the clusters found in training.

    K-means assigns to the nearest centroid (exactly KMeans.predict). Density
    and spectral methods have no predict step, so a new row takes the label of
    its nearest training point, or -1 (noise) if it is farther than the
    method's neighbourhood radius.
    """

    def __init__(self, bundle: dict):
        self.feature_cols = bundle["feature_cols"]
        self.preprocessor = bundle["preprocessor"]
        self.reducer = bundle.get("reducer")
        self.text = bundle.get("text")
        self.assignment = bundle["assignment"]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.preprocessor.transform(apply_text(X[self.feature_cols], self.text))
        if self.reducer is not None:
            Z = self.reducer.transform(Z)
        return np.asarray(Z, dtype=np.float64)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.transform(X)
        a = self.assignment
        if a["kind"] == "centroids":
            return nearest(Z, np.asarray(a["centers"], dtype=np.float64))[0]
        refs = np.asarray(a["points"], dtype=np.float64)
        idx, dist = nearest(Z, refs)
        labels = np.asarray(a["labels"])[idx]
        return np.where(dist <= a["radius"], labels, -1)


# -- forecasting -------------------------------------------------------------------

def calendar_features(timestamps) -> np.ndarray:
    """Cyclical day-of-week, month and hour encodings, shape (n, 6)."""
    ts = pd.DatetimeIndex(timestamps)
    cols = []
    for value, period in ((ts.dayofweek, 7), (ts.month - 1, 12), (ts.hour, 24)):
        angle = 2 * np.pi * np.asarray(value, dtype=np.float64) / period
        cols += [np.sin(angle), np.cos(angle)]
    return np.stack(cols, axis=1)


def window_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window mean and standard deviation (sd floored to 1 for flat windows).
    Every forecaster predicts in these standardised units."""
    windows = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    mu = windows.mean(axis=1)
    sd = windows.std(axis=1)
    return mu, np.where(sd > 1e-8, sd, 1.0)


def lag_features(windows: np.ndarray, steps, horizon: int, target_times) -> np.ndarray:
    """Features for the gradient-boosting forecaster, one row per (window,
    step): the standardised window, its recent level and slope, how far ahead
    the target is, and the target's calendar position."""
    windows = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    mu, sd = window_stats(windows)
    z = (windows - mu[:, None]) / sd[:, None]
    tail = z[:, -min(z.shape[1], 7):]
    extra = np.stack([tail.mean(axis=1), z[:, -1] - z[:, 0], np.asarray(steps, dtype=np.float64) / horizon], axis=1)
    return np.hstack([z, extra, calendar_features(target_times)])


def input_window(values: np.ndarray, length: int) -> np.ndarray:
    """The last `length` values; shorter histories are left-padded with their
    first value."""
    values = np.asarray(values, dtype=np.float64)
    return np.pad(values, (max(0, length - len(values)), 0), mode="edge")[-length:]


def _nets():
    import torch
    from torch import nn

    class RNNForecaster(nn.Module):
        """LSTM/GRU encoder over the input window; a linear head emits all steps."""

        def __init__(self, cell: str, horizon: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
            super().__init__()
            rnn = nn.LSTM if cell == "lstm" else nn.GRU
            self.rnn = rnn(1, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(hidden, horizon)

        def forward(self, x):
            out, _ = self.rnn(x.unsqueeze(-1))
            return self.head(out[:, -1])

    class CausalBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float):
            super().__init__()
            self.pad = 2 * dilation
            self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
            self.drop = nn.Dropout(dropout)
            self.act = nn.GELU()

        def forward(self, x):
            h = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
            h = self.drop(self.act(self.conv2(nn.functional.pad(h, (self.pad, 0)))))
            return x + h

    class TCNForecaster(nn.Module):
        """Temporal convolutional network: causal dilated 1-D convolutions."""

        def __init__(self, horizon: int, channels: int = 64, levels: int = 4, dropout: float = 0.1):
            super().__init__()
            self.inp = nn.Conv1d(1, channels, 1)
            self.blocks = nn.Sequential(*(CausalBlock(channels, 2 ** i, dropout) for i in range(levels)))
            self.head = nn.Linear(channels, horizon)

        def forward(self, x):
            h = self.blocks(self.inp(x.unsqueeze(1)))
            return self.head(h[:, :, -1])

    return torch, {"lstm": RNNForecaster, "gru": RNNForecaster, "tcn": TCNForecaster}


def build_net(arch: str, horizon: int, **kwargs):
    _, nets = _nets()
    if arch in ("lstm", "gru"):
        return nets[arch](arch, horizon, **kwargs)
    return nets[arch](horizon, **kwargs)


class ForecastModel:
    """Forecasts the next `horizon` steps of every series."""

    def __init__(self, bundle: dict):
        self.cfg = bundle["forecast"]
        self.kind = bundle["estimator"]["kind"]
        self.spec = bundle["estimator"]
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self.kind == "torch":
            import torch

            net = build_net(self.spec["arch"], self.cfg["horizon"], **self.spec.get("net_kwargs", {}))
            net.load_state_dict(torch.load(io.BytesIO(self.spec["state_dict"]), map_location="cpu", weights_only=True))
            self._model = net.to(device()).eval()
        elif self.kind == "xgboost-json":
            self._model = load_estimator(self.spec, "regression")
        elif self.kind == "chronos":
            from chronos import BaseChronosPipeline

            self._model = BaseChronosPipeline.from_pretrained(self.spec["model_id"], device_map=device())
        return self._model

    def _history(self, history: pd.DataFrame | None) -> dict[str, pd.Series]:
        if history is None:
            return {sid: pd.Series(s["values"], index=pd.DatetimeIndex(s["timestamps"]))
                    for sid, s in self.cfg["history"].items()}
        id_col, ts_col, target = "series_id", "timestamp", "value"
        frame = history.copy()
        if id_col not in frame:
            frame[id_col] = "series"
        frame[ts_col] = pd.to_datetime(frame[ts_col])
        return {str(sid): g.sort_values(ts_col).set_index(ts_col)[target].astype(float)
                for sid, g in frame.groupby(id_col)}

    def point_forecasts(self, series: dict[str, pd.Series]) -> dict[str, np.ndarray]:
        H, L = self.cfg["horizon"], self.cfg["input_length"]
        freq = self.cfg["freq"]
        model = self._load()
        out: dict[str, np.ndarray] = {}
        if self.kind == "chronos":
            frame = pd.concat([pd.DataFrame({"id": sid, "timestamp": s.index, "target": s.values})
                               for sid, s in series.items()])
            pred = model.predict_df(frame, prediction_length=H, quantile_levels=[0.5], id_column="id",
                                    timestamp_column="timestamp", target="target")
            for sid, g in pred.groupby("id"):
                out[str(sid)] = g.sort_values("timestamp")["predictions"].to_numpy(dtype=float)
            return out
        for sid, s in series.items():
            window = input_window(s.to_numpy(dtype=float), L)
            mu, sd = (float(v[0]) for v in window_stats(window))
            if self.kind == "torch":
                import torch

                with torch.no_grad():
                    x = torch.tensor((window - mu) / sd, dtype=torch.float32, device=device())[None]
                    out[sid] = model(x)[0].float().cpu().numpy() * sd + mu
            else:
                future = pd.date_range(s.index[-1], periods=H + 1, freq=freq)[1:]
                feats = lag_features(np.repeat(window[None], H, axis=0), np.arange(1, H + 1), H, future)
                out[sid] = np.asarray(model.predict(feats), dtype=float) * sd + mu
        return out

    def forecast(self, history: pd.DataFrame | None = None) -> pd.DataFrame:
        """history: columns timestamp, value (and series_id for several series).
        Omit it to forecast from the end of the training data."""
        series = self._history(history)
        H, L = self.cfg["horizon"], self.cfg["input_length"]
        points = self.point_forecasts(series)
        lower_q = np.asarray(self.cfg["interval"]["lower"], dtype=float)
        upper_q = np.asarray(self.cfg["interval"]["upper"], dtype=float)
        rows = []
        for sid, s in series.items():
            sd = float(window_stats(input_window(s.to_numpy(dtype=float), L))[1][0])
            future = pd.date_range(s.index[-1], periods=H + 1, freq=self.cfg["freq"])[1:]
            p = points[sid]
            rows.append(pd.DataFrame({
                "series_id": sid, "timestamp": future, "forecast": p,
                "lower": p + lower_q * sd, "upper": p + upper_q * sd,
            }))
        return pd.concat(rows, ignore_index=True)


# -- images ----------------------------------------------------------------------------

class ImageModel:
    """Image classification with a fine-tuned pretrained CNN."""

    def __init__(self, bundle: dict):
        import timm
        import torch

        spec = bundle["estimator"]
        self.class_labels = bundle["class_labels"]
        self.net = timm.create_model(spec["arch"], pretrained=False, num_classes=len(self.class_labels))
        self.net.load_state_dict(torch.load(io.BytesIO(spec["state_dict"]), map_location="cpu", weights_only=True))
        self.net = self.net.to(device()).eval()
        self.transform = timm.data.create_transform(**spec["data_config"], is_training=False)

    def predict_proba(self, images) -> np.ndarray:
        """images: file paths, raw bytes, or PIL images."""
        import torch
        from PIL import Image

        batch = []
        for img in images:
            if isinstance(img, (bytes, bytearray)):
                img = Image.open(io.BytesIO(img))
            elif not isinstance(img, Image.Image):
                img = Image.open(img)
            batch.append(self.transform(img.convert("RGB")))
        with torch.no_grad():
            logits = self.net(torch.stack(batch).to(device()))
        return torch.softmax(logits.float(), dim=1).cpu().numpy()

    def predict(self, images) -> np.ndarray:
        return self.predict_proba(images).argmax(axis=1)


# -- entry point -------------------------------------------------------------------------

_MODELS = {
    "classification": TabularModel,
    "regression": TabularModel,
    "clustering": ClusterModel,
    "forecasting": ForecastModel,
    "image_classification": ImageModel,
}


def load_bundle(bundle: dict) -> dict:
    """Turn a loaded model.joblib into {"model": <predictor>, ...}. Bundles from
    before the portable formats already hold a pickled pipeline under "model"."""
    if bundle.get("format") == LEGACY_FORMAT:
        bundle = {
            **bundle,
            "format": BUNDLE_FORMAT,
            "task": bundle["problem_type"],
            "estimator": {"kind": bundle["model_format"], "bytes": bundle["model_bytes"]},
        }
    if bundle.get("format") != BUNDLE_FORMAT:
        return bundle
    return {**bundle, "model": _MODELS[bundle["task"]](bundle)}
