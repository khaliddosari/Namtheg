"""Tabular training engine (app/training/tabular.py), run in-process on CPU as a
harness, plus the deployment invariants that tie the GPU and inference images together."""
import io
import sys
import types
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.model_selection import train_test_split

from app.deploy import inference_app
from app.pipeline import imbalance, train
from app.training import common, forecasting, gpu_app, tabular, text, vision
from app.training.runtime import load_bundle
from tests.conftest import INSURANCE_CSV

ALL = ["XGBoost", "XGBoost Leaf-wise", "CatBoost", "SVM", "KNN", "Linear"]


def _run(df: pd.DataFrame, target: str, task: str, candidates=None) -> dict:
    buf = io.BytesIO()
    df.to_parquet(buf)
    plan = {"imbalance": imbalance.assess(df[target], task), "tuning_trials": 2, "tuning_timeout_seconds": 30}
    return tabular.run(buf.getvalue(), {"task": task, "target": target, "candidates": candidates or ALL,
                                         "plan": plan}, "cpu")


# -- deployment invariants -------------------------------------------------------

def test_training_and_inference_images_pin_identical_versions():
    """scikit-learn objects pickled in the GPU images are loaded in the inference image."""
    shared = {"numpy", "scipy", "scikit-learn", "pandas", "joblib"}
    for pins in (gpu_app.CLASSIC_PIN, gpu_app.DEEP_PIN):
        for pkg in shared:
            assert pins[pkg] == inference_app.IMAGE_PIN[pkg], pkg
        assert pins["xgboost"] == inference_app.IMAGE_PIN["xgboost-cpu"]
    assert gpu_app.CLASSIC_PIN["catboost"] == inference_app.IMAGE_PIN["catboost"]
    for pkg in ("timm", "sentence-transformers", "chronos-forecasting"):
        assert gpu_app.DEEP_PIN[pkg] == inference_app.IMAGE_PIN[pkg], pkg
    for pkg in ("torch", "torchvision"):
        assert gpu_app.DEEP_PIN[pkg] == inference_app.TORCH_PIN[pkg], pkg
    assert gpu_app.PYTHON_VERSION == inference_app.PYTHON_VERSION


def test_gpu_settings_and_baked_weights_match_the_engines():
    assert gpu_app.GPU == train.TRAINING_GPU == "H200"
    assert gpu_app.PRETRAINED["text"] == text.ENCODER
    assert gpu_app.PRETRAINED["chronos"] == forecasting.CHRONOS_MODEL
    assert set(gpu_app.PRETRAINED["timm"]) == set(vision.ARCHITECTURES.values())


def test_gpu_only_rejects_cpu_fallback(monkeypatch):
    stats = types.SimpleNamespace(cpu_calls=1, fallback_reasons={"Multiclass `y` is not supported"})

    @contextmanager
    def profile(quiet=True):
        yield types.SimpleNamespace(method_calls={"SVC.fit": stats})

    fake = types.ModuleType("cuml")
    fake.accel = types.SimpleNamespace(profile=profile)
    monkeypatch.setitem(sys.modules, "cuml", fake)
    monkeypatch.setitem(sys.modules, "cuml.accel", fake.accel)
    with pytest.raises(common.CPUFallback, match="SVC.fit: Multiclass"):
        with common.gpu_only("cuda"):
            pass
    with common.gpu_only("cpu"):  # the CPU test harness is not policed
        pass


# -- tabular engine -------------------------------------------------------------------

def test_all_candidates_train_and_imbalance_is_handled():
    df = pd.read_csv(INSURANCE_CSV)  # smoker: 1064 no / 274 yes
    out = _run(df, "smoker", "classification")
    assert {c["name"] for c in out["candidates"] if c["status"] == "ok"} == set(ALL)
    assert out["imbalance_applied"] == {"class_weights": True, "selection_metric": "f1_macro"}
    assert out["baseline"]["metric"] == "f1_macro" and out["cv_mean"] > out["baseline"]["cv_mean"]
    assert {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "pr_auc"} <= set(out["test_metrics"])
    assert out["class_labels"] == ["no", "yes"]
    assert out["tuning_trials"][0]["trial"] == 0


def test_class_weights_are_computed_per_training_fold(monkeypatch):
    seen = []
    real = tabular.compute_sample_weight

    def spy(kind, y):
        seen.append(len(y))
        return real(kind, y)

    monkeypatch.setattr(tabular, "compute_sample_weight", spy)
    df = pd.read_csv(INSURANCE_CSV)
    out = _run(df, "smoker", "classification", ["XGBoost"])
    n_train = len(df) - out["test_size"]
    assert seen and all(n <= n_train for n in seen) and any(n < n_train for n in seen)


def test_boosted_trees_early_stop_and_refit_with_the_found_tree_count():
    out = _run(pd.read_csv(INSURANCE_CSV), "charges", "regression", ["XGBoost"])
    assert 1 <= out["params"]["n_estimators"] <= tabular.MAX_TREES


def test_tuning_stops_once_trials_stop_beating_the_cv_noise():
    df = pd.read_csv(INSURANCE_CSV)
    X, y = df.drop(columns=["charges"]), df["charges"].to_numpy(dtype=float)
    folds = common.make_folds("regression", y)
    # No R2 can beat 0 by more than 1, so every trial is stale.
    _, _, rows = tabular.tune("Linear", X, y, folds, "regression", "cpu", False, 0, "r2",
                              start_score=0.0, n_trials=20, timeout=60, min_gain=1.0)
    assert len(rows) == 1 + tabular.TUNING_PATIENCE
    assert tabular.cv_noise([0.8, 0.9]) == pytest.approx(0.05 / np.sqrt(2))


def test_multiclass_svm_stays_binary_per_class():
    out = _run(pd.read_csv(INSURANCE_CSV), "region", "classification", ["SVM"])
    est = joblib.load(io.BytesIO(out["bundle_bytes"]))["estimator"]["object"]
    from sklearn.multiclass import OneVsRestClassifier

    assert isinstance(est, OneVsRestClassifier) and len(est.estimators_) == 4
    assert np.load(io.BytesIO(out["weights_bytes"])).files  # support vectors exported


def test_svm_regression_standardises_the_target():
    out = _run(pd.read_csv(INSURANCE_CSV), "charges", "regression", ["SVM"])
    assert out["cv_mean"] > 0.5  # without target scaling SVR scores below zero here


def test_balanced_classes_keep_accuracy_and_no_weights():
    df = pd.DataFrame({"x": np.arange(200) % 7, "y": np.where(np.arange(200) % 2, "a", "b")})
    out = _run(df, "y", "classification", ["XGBoost", "Linear"])
    assert out["imbalance_applied"] == {"class_weights": False, "selection_metric": "accuracy"}


@pytest.mark.parametrize("labels", [(1, 2), (-1, 1), (True, False)])
def test_numeric_and_bool_labels_are_encoded(labels):
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "y": np.where(x > 0, labels[0], labels[1])})
    out = _run(df, "y", "classification", ["XGBoost", "KNN"])
    assert sorted(out["class_labels"]) == sorted(labels) and set(out["y_pred"]) <= {0, 1}


def test_too_few_rows_in_a_class_is_a_clear_error():
    df = pd.DataFrame({"x": range(40), "y": ["a"] * 38 + ["b"] * 2})
    with pytest.raises(ValueError, match="fewer than 3 rows"):
        _run(df, "y", "classification")


def test_oversized_svm_is_skipped_with_a_reason(monkeypatch):
    monkeypatch.setattr(tabular, "SVM_MAX_ROWS", 100)
    out = _run(pd.read_csv(INSURANCE_CSV), "charges", "regression", ["SVM", "Linear"])
    svm = next(c for c in out["candidates"] if c["name"] == "SVM")
    assert svm["status"] == "skipped" and "100" in svm["reason"] and out["best"] == "Linear"


@pytest.mark.parametrize("champion", ["XGBoost", "CatBoost", "SVM", "KNN", "Linear"])
def test_bundles_reproduce_predictions_without_library_pickles(champion):
    """XGBoost/CatBoost pickles break across builds and OSes, so bundles hold
    them only in native formats; rebuilt, they predict exactly as trained."""
    from joblib.numpy_pickle import NumpyUnpickler

    class Guard(NumpyUnpickler):
        def find_class(self, module, name):
            assert module.split(".")[0] not in ("xgboost", "catboost"), f"pickled {module}.{name}"
            return super().find_class(module, name)

    df = pd.read_csv(INSURANCE_CSV)
    out = _run(df, "smoker", "classification", [champion])
    raw = Guard("model.joblib", io.BytesIO(out["bundle_bytes"]), ensure_native_byte_order=False).load()
    model = load_bundle(raw)["model"]
    codes = np.unique(df["smoker"], return_inverse=True)[1]
    _, X_test = train_test_split(df.drop(columns=["smoker"]), test_size=common.TEST_SIZE,
                                 random_state=common.RANDOM_STATE, stratify=codes)
    assert model.predict(X_test).tolist() == out["y_pred"]


def test_native_xgboost_weights_reproduce_the_pipeline():
    df = pd.read_csv(INSURANCE_CSV)
    out = _run(df, "charges", "regression", ["XGBoost"])
    model = load_bundle(joblib.load(io.BytesIO(out["bundle_bytes"])))["model"]
    X = df.drop(columns=["charges"]).head(5)
    booster = xgb.Booster()
    booster.load_model(bytearray(out["weights_bytes"]))
    assert np.allclose(booster.predict(xgb.DMatrix(model.transform(X))), model.predict(X), rtol=1e-4)


def test_catboost_multiclass_predictions_are_flat():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    out = _run(pd.DataFrame({"x": x, "y": np.digitize(x, [-0.5, 0.5])}), "y", "classification", ["CatBoost"])
    assert out["model_name"] == "CatBoost" and np.asarray(out["y_pred"]).ndim == 1
