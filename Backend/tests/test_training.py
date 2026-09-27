"""The training engine (app/training/core.py), run in-process on CPU as a harness."""
import io

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.model_selection import train_test_split

from app.deploy import inference_app
from app.pipeline import imbalance
from app.training import core, gpu_app
from tests.conftest import INSURANCE_CSV


def _run(df: pd.DataFrame, target: str, problem_type: str) -> dict:
    buf = io.BytesIO()
    df.to_parquet(buf)
    plan = {"imbalance": imbalance.assess(df[target], problem_type), "tuning_trials": 1, "tuning_timeout_seconds": 30}
    return core.run_training(buf.getvalue(), target, problem_type, plan, device="cpu")


def test_gpu_and_inference_images_pin_identical_versions():
    """Pipelines pickled in the GPU image are unpickled in the inference image."""
    shared = {"numpy", "scipy", "scikit-learn", "pandas", "joblib", "catboost"}
    for pkg in shared:
        assert gpu_app.IMAGE_PIN[pkg] == inference_app.IMAGE_PIN[pkg], pkg
    assert gpu_app.IMAGE_PIN["xgboost"] == inference_app.IMAGE_PIN["xgboost-cpu"]
    assert gpu_app.PYTHON_VERSION == inference_app.PYTHON_VERSION
    assert gpu_app.GPU == "H200"


def test_imbalanced_classification_uses_class_weights_and_macro_f1():
    df = pd.read_csv(INSURANCE_CSV)  # smoker: 1064 no / 274 yes
    out = _run(df, "smoker", "classification")
    assert out["imbalance_applied"] == {"class_weights": True, "selection_metric": "f1_macro"}
    assert out["selection_metric"] == "f1_macro" and out["baseline"]["metric"] == "f1_macro"
    assert {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "pr_auc"} <= set(out["test_metrics"])
    assert out["class_labels"] == ["no", "yes"]
    assert out["cv_mean"] > out["baseline"]["cv_mean"]
    assert {m["name"] for m in out["all_models"]} == set(core.CANDIDATES)


def test_class_weights_are_computed_per_training_fold(monkeypatch):
    seen = []
    real = core.compute_sample_weight

    def spy(kind, y):
        seen.append(len(y))
        return real(kind, y)

    monkeypatch.setattr(core, "compute_sample_weight", spy)
    df = pd.read_csv(INSURANCE_CSV)
    out = _run(df, "smoker", "classification")
    n_train = len(df) - out["test_size"]
    # Every weighted fit saw fewer rows than the training split: fold-only weights.
    assert seen and all(n <= n_train for n in seen)
    assert any(n < n_train for n in seen)


def test_balanced_classes_keep_accuracy_and_no_weights():
    df = pd.DataFrame({"x": np.arange(200) % 7, "y": np.where(np.arange(200) % 2, "a", "b")})
    out = _run(df, "y", "classification")
    assert out["imbalance_applied"] == {"class_weights": False, "selection_metric": "accuracy"}


@pytest.mark.parametrize("labels", [(1, 2), (-1, 1), (True, False)])
def test_numeric_and_bool_labels_are_encoded(labels):
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "y": np.where(x > 0, labels[0], labels[1])})
    out = _run(df, "y", "classification")
    assert sorted(out["class_labels"]) == sorted(labels)
    assert set(out["y_pred"]) <= {0, 1}


def test_too_few_rows_in_a_class_is_a_clear_error():
    df = pd.DataFrame({"x": range(40), "y": ["a"] * 38 + ["b"] * 2})
    with pytest.raises(ValueError, match="fewer than 3 rows"):
        _run(df, "y", "classification")


@pytest.mark.parametrize("champion", ["XGBoost", "CatBoost"])
def test_bundle_holds_no_library_pickles(monkeypatch, champion):
    """XGBoost's pickle is a memory snapshot that fails on other builds/OSes, so
    the bundle must store models only in their native formats."""
    from joblib.numpy_pickle import NumpyUnpickler

    class Guard(NumpyUnpickler):
        def find_class(self, module, name):
            assert module.split(".")[0] not in ("xgboost", "catboost"), f"pickled {module}.{name}"
            return super().find_class(module, name)

    monkeypatch.setattr(core, "CANDIDATES", (champion,))
    out = _run(pd.read_csv(INSURANCE_CSV), "smoker", "classification")
    raw = Guard("model.joblib", io.BytesIO(out["bundle_bytes"]), ensure_native_byte_order=False).load()
    bundle = core.load_bundle(raw)
    assert bundle["model_name"] == champion

    # Rebuilt from native weights, it reproduces the trained model's test predictions exactly.
    df = pd.read_csv(INSURANCE_CSV)
    codes = np.unique(df["smoker"], return_inverse=True)[1]
    _, X_test = train_test_split(df.drop(columns=["smoker"]), test_size=core.TEST_SIZE,
                                 random_state=core.RANDOM_STATE, stratify=codes)
    assert np.asarray(bundle["model"].predict(X_test)).ravel().tolist() == out["y_pred"]
    proba = bundle["model"].predict_proba(X_test)
    assert proba.shape == (len(X_test), 2) and np.allclose(proba.sum(axis=1), 1)


def test_bundle_serves_on_cpu_and_native_weights_load():
    df = pd.read_csv(INSURANCE_CSV)
    out = _run(df, "charges", "regression")
    bundle = core.load_bundle(joblib.load(io.BytesIO(out["bundle_bytes"])))
    model = bundle["model"]
    preds = np.asarray(model.predict(df[bundle["feature_cols"]].head(5))).ravel()
    assert preds.shape == (5,) and np.isfinite(preds).all()
    encoded = model[:-1].transform(df[bundle["feature_cols"]].head(5))
    if out["weights_format"] == "xgboost-json":
        booster = xgb.Booster()
        booster.load_model(bytearray(out["weights_bytes"]))
        native = booster.predict(xgb.DMatrix(encoded))
    else:
        from catboost import CatBoost

        cb = CatBoost()
        cb.load_model(blob=out["weights_bytes"])
        native = cb.predict(encoded)
    # The native weights alone reproduce the pipeline's predictions.
    assert np.allclose(native, preds, rtol=1e-4)
    enc = out["preprocessing"]["encoded_feature_names"]
    assert len(enc) >= len(bundle["feature_cols"])


def test_catboost_multiclass_predictions_are_flat(monkeypatch):
    monkeypatch.setattr(core, "CANDIDATES", ("CatBoost",))
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "y": np.digitize(x, [-0.5, 0.5])})
    out = _run(df, "y", "classification")
    assert out["model_name"] == "CatBoost" and np.asarray(out["y_pred"]).ndim == 1
