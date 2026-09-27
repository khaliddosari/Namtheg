"""End-to-end runs of the orchestrator with a scripted LLM, the local sandbox,
and the training engine in-process (see conftest.cpu_training_harness)."""
import io
import shutil
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import storage
from app.agent import orchestrator
from app.pipeline import train
from tests.conftest import INSURANCE_CSV, ScriptedLLM

ANALYST_SCRIPT = [
    [("run_python", {"code": "df.groupby('smoker')['charges'].mean().round(2)"})],
    [("submit_analysis", {
        "problem_type": "regression",
        "problem_type_rationale": "charges is a continuous amount.",
        "drop_columns": [
            {"column": "region", "reason": "other", "evidence": "Dropped for the test."},
            {"column": "nope", "reason": "leakage", "evidence": "x"},
            {"column": "charges", "reason": "leakage", "evidence": "x"},
        ],
        "findings": [
            {"finding": "Smokers are charged far more.", "evidence": "mean 32050.23 vs 8434.27", "severity": "warning"},
            {"finding": "Charges grow 73% a year.", "evidence": "trend", "severity": "info"},
        ],
        "open_questions": ["Are charges annual or lifetime?"],
        "summary": "Smoking status dominates charges (32050.23 vs 8434.27).",
    })],
]


def _use_llm(monkeypatch, llm):
    monkeypatch.setattr(orchestrator, "LLMClient", lambda: llm)


def test_full_run_with_analyst(uploaded_run, monkeypatch):
    llm = ScriptedLLM(ANALYST_SCRIPT)
    _use_llm(monkeypatch, llm)

    result = orchestrator.run_agent(uploaded_run, "charges")

    assert result["status"] == "succeeded", result.get("error")
    assert result["problem_type"] == "regression" and result["score_metric"] == "r2"
    extra = result["extra"]

    analysis = extra["analysis"]
    assert analysis["status"] == "completed" and analysis["code_runs"] == 1
    assert [f["verified"] for f in analysis["findings"]] == [True, False]
    assert analysis["findings"][1]["unverified_numbers"] == ["73%"]
    assert [d["column"] for d in analysis["drop_columns"]] == ["region"]
    assert any("'nope': no such column" in n for n in analysis["notes"])
    assert any("target column" in n for n in analysis["notes"])

    # The agent saw the real sandbox output, and the log keeps it.
    tool_output = llm.analyst_transcripts[1][-1]["content"]
    assert "32050.23" in tool_output and "8434.27" in tool_output
    log = storage.read_json(uploaded_run, "agent_log.json")
    assert len(log["cells"]) == 1 and "32050.23" in log["cells"][0]["output"]

    # The analyst's drop reached training.
    bundle = joblib.load(storage.artifact_path(uploaded_run, "model.joblib"))
    assert "region" not in bundle["feature_cols"] and "smoker" in bundle["feature_cols"]

    assert extra["baseline"]["name"] == "Mean of target"
    assert extra["beats_baseline"] is True
    assert extra["imbalance"]["applies"] is False
    assert extra["hardware"]["gpu"] == "CPU test harness"
    assert extra["grounding"]["source"] == "llm" and extra["grounding"]["verified"]
    assert {"rmse", "mae", "test_score", "train_score"} <= set(extra)
    assert result["downloads"] == ["cleaned_csv", "model"]
    assert storage.read_status(uploaded_run)["stage"] == "done"


def test_hallucinated_summary_falls_back_to_template(uploaded_run, monkeypatch):
    _use_llm(monkeypatch, ScriptedLLM(ANALYST_SCRIPT, hallucinate="97.5%"))
    result = orchestrator.run_agent(uploaded_run, "charges")
    assert result["status"] == "succeeded"
    assert result["extra"]["grounding"]["source"] == "template"
    assert "97.5" not in result["justification"]


def test_imbalanced_run_without_llm(uploaded_run, monkeypatch):
    def no_sandbox(*a, **k):
        raise AssertionError("sandbox must not start without an LLM")

    monkeypatch.setattr(orchestrator, "open_sandbox", no_sandbox)
    result = orchestrator.run_agent(uploaded_run, "smoker")
    assert result["status"] == "succeeded", result.get("error")
    extra = result["extra"]
    assert result["problem_type"] == "classification"
    # smoker is 1064 "no" to 274 "yes": handled before modelling.
    assert extra["imbalance"]["detected"] and extra["imbalance"]["strategy"] == "balanced_class_weights"
    assert extra["imbalance_applied"] == {"class_weights": True, "selection_metric": "f1_macro"}
    assert result["score_metric"] == "f1_macro"
    assert any(f["check"] == "class_imbalance" for f in extra["audit"]["findings"])
    assert extra["analysis"]["status"] == "skipped"
    assert extra["grounding"]["source"] == "template"
    assert extra["baseline"]["name"] == "Majority class"
    # The confusion-matrix inputs carry real class names, not codes.
    y_test = np.load(storage.artifact_path(uploaded_run, "y_test.npy"), allow_pickle=True)
    assert set(y_test) == {"no", "yes"}


def test_gpu_failure_fails_the_run_with_no_cpu_fallback(uploaded_run, monkeypatch):
    def gpu_down(*a, **k):
        raise RuntimeError("No NVIDIA GPU visible in the training container; refusing to train on CPU.")

    monkeypatch.setattr(train, "_remote_train", gpu_down)
    result = orchestrator.run_agent(uploaded_run, "charges")
    assert result["status"] == "failed" and "refusing to train on CPU" in result["error"]
    assert not (storage.run_dir(uploaded_run) / "model.joblib").exists()


def test_legacy_csv_run_is_migrated_on_first_access():
    run_id = storage.new_run_id()
    shutil.copy(INSURANCE_CSV, storage.run_dir(run_id) / "dataset.csv")
    df = storage.load_dataset(run_id)
    assert df.shape == (1338, 7)
    assert (storage.run_dir(run_id) / "dataset.parquet").exists()
    assert storage.read_json(run_id, "ingest.json")["n_rows"] == 1338


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_upload_excel_and_start(client, monkeypatch):
    monkeypatch.setattr("app.main.run_agent", lambda run_id, target: None)
    buf = io.BytesIO()
    pd.read_csv(INSURANCE_CSV).to_excel(buf, index=False)
    r = client.post("/upload", files={"file": ("insurance.xlsx", buf.getvalue())})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_rows"] == 1338 and body["ingest"]["source_format"] == "xlsx"
    assert len(body["preview"]) == 5 and "charges" in body["columns"]

    run_id = body["run_id"]
    assert client.get(f"/runs/{run_id}/preview").json()["n_columns"] == 7
    assert client.post(f"/runs/{run_id}/start", json={"target": "nope"}).status_code == 400
    assert client.post(f"/runs/{run_id}/start", json={"target": "charges"}).status_code == 200


def test_upload_rejects_bad_files_and_cleans_up(client):
    assert client.post("/upload", files={"file": ("x.pdf", b"%PDF")}).status_code == 400
    r = client.post("/upload", files={"file": ("x.csv", b"")})
    assert r.status_code == 400 and "empty" in r.json()["detail"]
    assert not any((storage.settings.storage_dir / "runs").iterdir())


def test_run_ids_are_validated(client):
    for bad in ("..", "abc", "ABCDEF123456"):
        assert client.get(f"/runs/{bad}/status").status_code == 404


def test_downloads_cleaned_csv_and_working_model_package(uploaded_run, client, tmp_path):
    result = orchestrator.run_agent(uploaded_run, "smoker")
    assert result["status"] == "succeeded", result.get("error")

    r = client.get(f"/runs/{uploaded_run}/download/cleaned_csv")
    assert r.status_code == 200
    assert 'filename="insurance_cleaned.csv"' in r.headers["content-disposition"]
    assert r.content.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Excel
    cleaned = pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig")
    assert len(cleaned) == 1338 and "smoker" in cleaned.columns

    r = client.get(f"/runs/{uploaded_run}/download/model")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = set(z.namelist())
        assert {"model.joblib", "metadata.json", "predict.py", "requirements.txt", "README.md"} <= names
        assert names & {"model_weights.json", "model_weights.cbm"}
        z.extractall(tmp_path / "pkg")
    assert "xgboost==" in (tmp_path / "pkg" / "requirements.txt").read_text() or \
        "catboost==" in (tmp_path / "pkg" / "requirements.txt").read_text()

    # The packaged predict.py runs as shipped.
    cleaned.drop(columns=["smoker"]).head(20).to_csv(tmp_path / "in.csv", index=False)
    subprocess.run([sys.executable, "predict.py", str(tmp_path / "in.csv"), str(tmp_path / "out.csv")],
                   cwd=tmp_path / "pkg", check=True, capture_output=True)
    preds = pd.read_csv(tmp_path / "out.csv")
    assert set(preds["prediction"]) <= {"no", "yes"}
    assert {"probability_no", "probability_yes"} <= set(preds.columns)

    assert client.get(f"/runs/{uploaded_run}/download/weights").status_code == 404
