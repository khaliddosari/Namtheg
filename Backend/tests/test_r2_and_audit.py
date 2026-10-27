import inspect
import shutil
import time
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main, r2, storage
from app.config import settings
from app.main import app as fastapi_app
from app.pipeline import audit, imbalance
from tests.conftest import INSURANCE_CSV


class FakeS3:
    """In-memory stand-in for the boto3 S3 client, covering what app/r2.py uses."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploaded: list[str] = []  # every key sent up, in order

    def upload_file(self, path, bucket, key):
        with open(path, "rb") as f:
            self.objects[key] = f.read()
        self.uploaded.append(key)

    def download_file(self, bucket, key, path):
        if key not in self.objects:
            raise _missing()
        with open(path, "wb") as f:
            f.write(self.objects[key])

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _missing()
        return {"ContentLength": len(self.objects[Key])}

    def generate_presigned_url(self, op, Params, ExpiresIn):
        if op == "put_object":
            return f"https://r2.example/{Params['Key']}?upload&expires={ExpiresIn}"
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


def _direct_upload(client, fake_r2, content: bytes) -> str:
    """Start a direct upload of a CSV and put it where the browser's PUT would."""
    r = client.post("/uploads/direct", json={"filename": "big.csv", "size": len(content)})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    fake_r2.objects[f"test/runs/{run_id}/raw/source.csv"] = content
    return run_id


def test_large_files_upload_straight_to_r2(fake_r2):
    """The browser PUTs to a presigned URL; /complete answers at once and the
    file is read after the response (TestClient runs that before returning)."""
    client = TestClient(fastapi_app)
    csv = INSURANCE_CSV.read_bytes()
    r = client.post("/uploads/direct", json={"filename": "big.csv", "size": len(csv)})
    assert r.status_code == 200, r.text
    run_id, url = r.json()["run_id"], r.json()["upload_url"]
    assert url.startswith(f"https://r2.example/test/runs/{run_id}/raw/source.csv?upload")
    assert client.get(f"/runs/{run_id}/status").json()["status"] == "uploading"
    assert client.post(f"/uploads/{run_id}/complete").status_code == 409  # nothing uploaded yet

    raw_key = f"test/runs/{run_id}/raw/source.csv"
    fake_r2.objects[raw_key] = csv  # what the browser's PUT does
    r = client.post(f"/uploads/{run_id}/complete")
    assert r.status_code == 202 and r.json() == {"run_id": run_id, "status": "uploading"}
    assert client.get(f"/runs/{run_id}/status").json()["status"] == "uploaded"
    assert storage.read_json(run_id, "ingest.json")["n_rows"] == 1338
    assert client.get(f"/runs/{run_id}/preview").json()["n_columns"] == 7
    # The raw file came from R2 under this key, so only the dataset goes back up.
    assert raw_key not in fake_r2.uploaded
    assert f"test/runs/{run_id}/{storage.DATASET_FILE}" in fake_r2.uploaded
    assert client.post(f"/uploads/{run_id}/complete").status_code == 409  # only once
    assert client.post("/uploads/direct", json={"filename": "x.pdf", "size": 10}).status_code == 400


def test_direct_upload_that_cannot_be_read_fails_with_the_reason(fake_r2):
    client = TestClient(fastapi_app)
    run_id = _direct_upload(client, fake_r2, b"a,b\n")  # a header and no rows
    assert client.post(f"/uploads/{run_id}/complete").status_code == 202
    status = client.get(f"/runs/{run_id}/status").json()
    assert status["status"] == "failed" and "no data rows" in status["error"]


def test_direct_upload_is_read_once_and_a_stale_read_can_restart(fake_r2, monkeypatch):
    client = TestClient(fastapi_app)
    run_id = _direct_upload(client, fake_r2, INSURANCE_CSV.read_bytes())
    reads = []
    monkeypatch.setattr(main, "_read_direct_upload", lambda *args: reads.append(args))
    assert client.post(f"/uploads/{run_id}/complete").status_code == 202
    assert client.post(f"/uploads/{run_id}/complete").status_code == 202  # still reading
    assert len(reads) == 1
    storage.update_status(run_id, reading_started=time.time() - main.READ_STALE_SECONDS - 1)
    assert client.post(f"/uploads/{run_id}/complete").status_code == 202  # presumed dead: read again
    assert len(reads) == 2


def test_direct_upload_over_the_limit_is_refused_before_reading(fake_r2, monkeypatch):
    client = TestClient(fastapi_app)
    run_id = _direct_upload(client, fake_r2, b"a,b\n1,2\n")
    monkeypatch.setattr(r2, "object_size", lambda *args: main.MAX_DIRECT_UPLOAD_BYTES + 1)
    assert client.post(f"/uploads/{run_id}/complete").status_code == 413
    assert client.get(f"/runs/{run_id}/status").json()["status"] == "uploading"


def test_direct_upload_needs_r2():
    client = TestClient(fastapi_app)
    assert client.post("/uploads/direct", json={"filename": "big.csv", "size": 10}).status_code == 409
