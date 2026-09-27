import inspect
import shutil
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import r2, storage
from app.config import settings
from app.main import app as fastapi_app
from app.pipeline import audit, imbalance
from tests.conftest import INSURANCE_CSV


class FakeS3:
    """In-memory stand-in for the boto3 S3 client, covering what app/r2.py uses."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def upload_file(self, path, bucket, key):
        with open(path, "rb") as f:
            self.objects[key] = f.read()

    def download_file(self, bucket, key, path):
        if key not in self.objects:
            raise _missing()
        with open(path, "wb") as f:
            f.write(self.objects[key])

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _missing()
        return {}

    def generate_presigned_url(self, op, Params, ExpiresIn):
        return f"https://r2.example/{Params['Key']}?expires={ExpiresIn}&cd={Params['ResponseContentDisposition']}"


def _missing():
    e = Exception("not found")
    e.response = {"Error": {"Code": "404"}}
    return e


@pytest.fixture
def fake_r2(monkeypatch):
    for key, value in {"r2_account_id": "acct", "r2_access_key_id": "id", "r2_secret_access_key": "secret",
                       "r2_bucket": "bucket", "r2_prefix": "test/"}.items():
        monkeypatch.setattr(settings, key, value)
    s3 = FakeS3()
    monkeypatch.setattr(r2, "_client", s3)
    return s3


def test_r2_is_off_without_credentials():
    assert not r2.enabled()
    assert r2.upload(INSURANCE_CSV, "abcdef123456", "x") is False


def test_artifacts_mirror_to_r2_and_restore_after_local_loss(fake_r2, uploaded_run):
    from app.agent import orchestrator

    result = orchestrator.run_agent(uploaded_run, "charges")
    assert result["status"] == "succeeded", result.get("error")
    keys = {k.rsplit("/", 1)[-1] for k in fake_r2.objects if k.startswith(f"test/runs/{uploaded_run}/")}
    assert {"dataset.parquet", "engineered.parquet", "cleaned.csv", "model.joblib", "model_package.zip",
            "result.json", "status.json", "plot.png", "metrics.json", "model_meta.json"} <= keys
    assert f"test/runs/{uploaded_run}/raw/source.csv" in fake_r2.objects

    # Lose the local copy entirely: the run comes back from R2 on demand.
    shutil.rmtree(storage.run_dir(uploaded_run))
    client = TestClient(fastapi_app)
    assert client.get(f"/runs/{uploaded_run}/result").json()["status"] == "succeeded"
    assert client.get(f"/runs/{uploaded_run}/preview").status_code == 200

    r = client.get(f"/runs/{uploaded_run}/download/model", follow_redirects=False)
    assert r.status_code == 307
    location = unquote(r.headers["location"])
    assert location.startswith(f"https://r2.example/test/runs/{uploaded_run}/model_package.zip")
    assert "expires=900" in location and 'filename="insurance_model.zip"' in location


def test_leakage_audit_flags_leaks_without_training_models(uploaded_run):
    df = storage.load_dataset(uploaded_run)
    rng = np.random.default_rng(0)
    df["charges_copy"] = np.log(df["charges"]) + rng.normal(0, 1e-6, len(df))  # monotonic transform
    df["is_smoker_flag"] = (df["smoker"] == "yes").astype(int)
    df.to_parquet(storage.dataset_path(uploaded_run))

    report = audit.audit_dataset(uploaded_run, "charges", "regression")
    leaks = {f["column"] for f in report["findings"] if f["check"] == "leakage_suspect"}
    assert "charges_copy" in leaks and "age" not in leaks

    report = audit.audit_dataset(uploaded_run, "smoker", "classification")
    leaks = {f["column"] for f in report["findings"] if f["check"] == "leakage_suspect"}
    assert "is_smoker_flag" in leaks and "bmi" not in leaks

    # Model fitting happens only on the GPU trainer: the audit uses no ML library.
    assert "sklearn" not in inspect.getsource(audit)


def test_imbalance_assessment_thresholds():
    balanced = imbalance.assess(pd.Series(["a"] * 55 + ["b"] * 45), "classification")
    assert not balanced["detected"] and balanced["selection_metric"] == "accuracy"
    skewed = imbalance.assess(pd.Series(["a"] * 80 + ["b"] * 20), "classification")
    assert skewed["detected"] and skewed["ratio"] == 4.0
    assert skewed["strategy"] == "balanced_class_weights" and skewed["selection_metric"] == "f1_macro"
    assert imbalance.assess(pd.Series([1.0, 2.0]), "regression")["applies"] is False
