"""Clustering, free text, forecasting and images: preparation rules and full
runs through the orchestrator (engines in-process on CPU, see conftest)."""
import io
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from app import storage
from app.agent import orchestrator
from app.data.images import ingest_image_zip
from app.data.ingest import IngestError
from app.pipeline import text_features, timeseries
from app.training import clustering, tsmetrics
from app.training.runtime import load_bundle
from tests.conftest import make_run


def _package(run_id, tmp_path):
    zipfile.ZipFile(storage.artifact_path(run_id, "model_package.zip")).extractall(tmp_path / "pkg")
    return tmp_path / "pkg"


# -- clustering --------------------------------------------------------------------

def blobs_frame():
    X, y = make_blobs(n_samples=600, centers=3, n_features=4, cluster_std=0.8, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["customer_id"] = [f"C{i:05d}" for i in range(len(df))]  # identifier: must be dropped
    return df, y


def test_clustering_run_finds_the_groups_and_exports_them(tmp_path):
    df, truth = blobs_frame()
    run_id = make_run(df)
    result = orchestrator.run_agent(run_id, {"task": "clustering", "target": None})
    assert result["status"] == "succeeded", result.get("error")
    extra = result["extra"]
    assert result["score_metric"] == "silhouette" and extra["cluster_metrics"]["n_clusters"] == 3
    assert {m["name"] for m in extra["all_models"]} >= {"K-Means", "HDBSCAN", "DBSCAN"}
    assert "customer_id" not in storage.read_json(run_id, "model_meta.json")["feature_cols"]
    assert sum(p["size"] for p in extra["cluster_profiles"]) == len(df)
    assert "structure" in result["justification"] and extra["grounding"]["source"] == "template"

    cleaned = pd.read_csv(storage.artifact_path(run_id, "cleaned.csv"), encoding="utf-8-sig")
    assert adjusted_rand_score(truth, cleaned["cluster"]) > 0.95

    pkg = _package(run_id, tmp_path)
    df.head(30).to_csv(tmp_path / "in.csv", index=False)
    subprocess.run([sys.executable, "predict.py", str(tmp_path / "in.csv"), str(tmp_path / "out.csv")],
                   cwd=pkg, check=True, capture_output=True)
    assert (pd.read_csv(tmp_path / "out.csv")["cluster"] == cleaned["cluster"].head(30)).all()


def test_density_clusters_flag_far_away_rows_as_outliers():
    X, _ = make_blobs(n_samples=400, centers=[[0, 0], [8, 8]], cluster_std=0.4, random_state=1)
    buf = io.BytesIO()
    pd.DataFrame(X, columns=["x", "y"]).to_parquet(buf)
    out = clustering.run(buf.getvalue(), {"candidates": ["HDBSCAN", "DBSCAN"]}, "cpu")
    model = load_bundle(joblib.load(io.BytesIO(out["bundle_bytes"])))["model"]
    far = pd.DataFrame({"x": [0.1, 50.0], "y": [0.0, -50.0]})
    labels = model.predict(far)
    assert labels[0] >= 0 and labels[1] == -1


# -- free text ------------------------------------------------------------------------------

def reviews_frame(n=240):
    rng = np.random.default_rng(0)
    good = ["great product works perfectly", "excellent quality fast delivery", "love it very happy with it"]
    bad = ["broke after one day terrible", "awful quality would not buy again", "very disappointed poor service"]
    rows = []
    for i in range(n):
        positive = i % 2 == 0
        base = (good if positive else bad)[i % 3]
        rows.append({"review": f"{base} order {i}", "price": float(rng.normal(50, 10)),
                     "city": ["Riyadh", "Jeddah", "Dammam"][i % 3], "positive": "yes" if positive else "no"})
    return pd.DataFrame(rows)


def test_free_text_is_detected_but_short_labels_are_not():
    found = text_features.detect_text_columns(reviews_frame(), exclude=["positive"])
    assert list(found) == ["review"]


def test_text_column_is_embedded_and_used_end_to_end(tmp_path):
    df = reviews_frame()
    run_id = make_run(df)
    result = orchestrator.run_agent(run_id, {"task": "classification", "target": "positive"})
    assert result["status"] == "succeeded", result.get("error")
    extra = result["extra"]
    assert "review" in extra["text_features"]["columns"]
    fe = storage.read_json(run_id, "feature_engineering.json")
    assert fe["text_columns"] == ["review"] and not any("review" in d for d in fe["dropped_columns"])
    assert result["accuracy_score"] > 0.9  # the review text carries the label

    bundle = load_bundle(joblib.load(storage.artifact_path(run_id, "model.joblib")))
    assert "review" in bundle["feature_cols"]  # raw text in, embedded at prediction time
    new = pd.DataFrame([{"review": "terrible awful broke", "price": 50.0, "city": "Riyadh"}])
    labels = bundle["class_labels"]
    assert labels[int(bundle["model"].predict(new)[0])] == "no"


# -- forecasting ----------------------------------------------------------------------------

def sales_frame(days=240, stores=("north", "south")):
    rng = np.random.default_rng(0)
    rows = []
    for k, store in enumerate(stores):
        t = pd.date_range("2025-01-01", periods=days, freq="D")
        y = 100 + 30 * k + 15 * np.sin(2 * np.pi * np.arange(days) / 7) + rng.normal(0, 2, days)
        rows.append(pd.DataFrame({"date": t.strftime("%Y-%m-%d"), "store": store, "sales": y}))
    return pd.concat(rows, ignore_index=True)


def test_timeseries_prepare_infers_frequency_and_fills_small_gaps():
    df = sales_frame().drop(index=[5, 40])  # two missing days in one store
    run_id = make_run(df)
    ts = timeseries.prepare(run_id, "date", "sales", "store")
    assert ts["freq"] == "D" and ts["season"] == 7 and ts["horizon"] == 14 and ts["n_series"] == 2
    assert ts["interpolated_steps"] == {"north": 2}
    assert ts["baselines"]["Seasonal naive"]["mase"] < ts["baselines"]["Naive (last value)"]["mase"]


@pytest.mark.parametrize("mutate,message", [
    (lambda df: pd.concat([df, df.head(3)]), "repeat a timestamp"),
    (lambda df: df.iloc[::3], None),  # regular every-3-days data is fine
    (lambda df: df.drop(index=range(100, 160)), "missing"),  # one 60-day hole: 25% of the series
])
def test_timeseries_prepare_refuses_to_guess(mutate, message):
    run_id = make_run(mutate(sales_frame(stores=("north",)).reset_index(drop=True)))
    if message is None:
        assert timeseries.prepare(run_id, "date", "sales")["freq"] == "3D"
    else:
        with pytest.raises(timeseries.TimeSeriesError, match=message):
            timeseries.prepare(run_id, "date", "sales")


def test_tsmetrics_mase_is_one_for_the_seasonal_naive_of_a_perfect_season():
    history = np.tile([1.0, 2.0, 3.0, 4.0], 6)
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    pred = tsmetrics.seasonal_naive(history, 4, 4)
    assert tsmetrics.evaluate(actual, pred, history, 4)["mae"] == 0
    assert tsmetrics.backtest_cuts(100, 10, 3) == [70, 80, 90]


def test_forecasting_run_end_to_end(tmp_path):
    run_id = make_run(sales_frame())
    result = orchestrator.run_agent(run_id, {"task": "forecasting", "target": "sales", "date_column": "date",
                                             "series_id_column": "store", "horizon": 7})
    assert result["status"] == "succeeded", result.get("error")
    extra = result["extra"]
    assert result["score_metric"] == "mase" and result["higher_is_better"] is False
    assert {m["name"] for m in extra["all_models"]} == {"LSTM", "TCN", "XGBoost (lags)"}
    assert extra["baseline"]["name"] == "Seasonal naive"
    assert result["downloads"] == ["cleaned_csv", "forecast", "model"]

    forecast = pd.read_csv(storage.artifact_path(run_id, "forecast.csv"), encoding="utf-8-sig")
    assert len(forecast) == 14 and (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper"]).all()
    assert pd.to_datetime(forecast["timestamp"]).min() == pd.Timestamp("2025-08-29")

    pkg = _package(run_id, tmp_path)
    subprocess.run([sys.executable, "predict.py", "--forecast", str(tmp_path / "f.csv")],
                   cwd=pkg, check=True, capture_output=True)
    again = pd.read_csv(tmp_path / "f.csv")
    assert np.allclose(again["forecast"], forecast["forecast"], rtol=1e-4, atol=1e-3)


# -- images ---------------------------------------------------------------------------------

def image_zip(classes=("cat", "dog"), per_class=12, root="pets", extra: dict | None = None) -> bytes:
    rng = np.random.default_rng(0)
    colors = [(220, 40, 40), (40, 60, 220), (40, 200, 60)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, label in enumerate(classes):
            for i in range(per_class):
                arr = np.clip(np.array(colors[k]) + rng.normal(0, 20, (32, 32, 3)), 0, 255).astype("uint8")
                b = io.BytesIO()
                Image.fromarray(arr).save(b, "PNG")
                z.writestr(f"{root}/{label}/{i}.png", b.getvalue())
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


def test_image_zip_ingest_is_safe_and_reports_skips(tmp_path):
    src = tmp_path / "in.zip"
    src.write_bytes(image_zip(extra={"pets/cat/notes.txt": b"hi", "__MACOSX/pets/._x.png": b"x",
                                     "pets/dog/broken.png": b"not an image", "../evil.png": b"x"}))
    report = ingest_image_zip(src, tmp_path / "run")
    manifest = pd.read_parquet(tmp_path / "run" / "dataset.parquet")
    assert report["classes"] == {"cat": 12, "dog": 12} and len(manifest) == 24
    # Unreadable and non-image files are reported; hidden/traversal paths are never touched.
    assert report["skipped_count"] == 2 and any("unreadable" in w for w in report["warnings"])
    assert not (tmp_path / "evil.png").exists() and not any("evil" in p for p in manifest["image"])
    assert all(p.startswith("images/") for p in manifest["image"])


@pytest.mark.parametrize("data,message", [
    (b"not a zip", "not a valid zip"),
    (image_zip(classes=("cat",)), "at least 2"),
    (image_zip(per_class=3), "at least 5 images"),
], ids=["not-zip", "one-class", "too-few"])
def test_image_zip_ingest_rejects_unusable_data(tmp_path, data, message):
    src = tmp_path / "in.zip"
    src.write_bytes(data)
    with pytest.raises(IngestError, match=message):
        ingest_image_zip(src, tmp_path / "run")


def test_image_classification_end_to_end(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    r = client.post("/upload", files={"file": ("pets.zip", image_zip(classes=("cat", "dog", "bird"), per_class=14))})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ingest"]["source_format"] == "images" and body["columns"][:2] == ["image", "label"]
    run_id = body["run_id"]
    r = client.post(f"/runs/{run_id}/start", json={"task": "regression", "target": "label"})
    assert r.status_code == 400 and "image classification" in r.text

    result = orchestrator.run_agent(run_id, {"task": "image_classification", "target": "label"})
    assert result["status"] == "succeeded", result.get("error")
    extra = result["extra"]
    assert extra["split_sizes"]["test"] > 0 and extra["baseline"]["name"] == "Majority class"
    assert set(np.load(storage.artifact_path(run_id, "y_test.npy"), allow_pickle=True)) <= {"cat", "dog", "bird"}

    pkg = _package(run_id, tmp_path)
    manifest = storage.load_dataset(run_id)
    files = [str(storage.run_dir(run_id) / p) for p in manifest["image"].head(3)]
    subprocess.run([sys.executable, "predict.py", "--images", str(tmp_path / "p.csv"), *files],
                   cwd=pkg, check=True, capture_output=True)
    preds = pd.read_csv(tmp_path / "p.csv")
    assert len(preds) == 3 and set(preds["prediction"]) <= {"cat", "dog", "bird"}
