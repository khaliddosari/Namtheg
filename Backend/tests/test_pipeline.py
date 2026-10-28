"""End-to-end tabular runs of the orchestrator with a scripted LLM, the local
sandbox and the training engines in-process (see conftest.cpu_training_harness),
plus the HTTP API around them."""
import io
import os
import shutil
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import jobs, storage
from app.agent import analyst, orchestrator
from app.pipeline import export, train
from tests.conftest import INSURANCE_CSV, ScriptedLLM, flight_times, make_run

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

    # The analyst's drop reached training; every model family competed.
    meta = storage.read_json(uploaded_run, "model_meta.json")
    assert "region" not in meta["feature_cols"] and "smoker" in meta["feature_cols"]
    names = {m["name"] for m in extra["all_models"]}
    assert {"XGBoost", "CatBoost", "SVM", "KNN", "Ridge Regression"} <= names

    assert extra["baseline"]["name"] == "Mean of target"
    assert extra["beats_baseline"] is True
    assert extra["hardware"]["gpu"] == "CPU test harness"
    assert extra["grounding"]["source"] == "llm" and extra["grounding"]["verified"]
    assert {"rmse", "mae", "test_score", "train_score"} <= set(extra)
    assert result["downloads"] == ["cleaned_csv", "model"]
    assert storage.read_status(uploaded_run)["stage"] == "done"


def test_the_analyst_gets_the_stated_prediction_time(uploaded_run, monkeypatch):
    llm = ScriptedLLM(ANALYST_SCRIPT)
    _use_llm(monkeypatch, llm)
    spec = {"task": "auto", "target": "charges", "prediction_time": "when the quote is issued"}
    result = orchestrator.run_agent(uploaded_run, spec)
    assert result["status"] == "succeeded", result.get("error")
    assert '"prediction_time": "when the quote is issued"' in llm.analyst_transcripts[0][0]["content"]
    assert result["extra"]["analysis"]["prediction_time"] == "when the quote is issued"


def test_dropping_a_column_for_its_timing_needs_a_stated_prediction_time():
    def reasons(tool):
        return tool.parameters["properties"]["drop_columns"]["items"]["properties"]["reason"]["enum"]

    timing = "not_available_at_prediction_time"
    assert timing not in reasons(analyst.submit_analysis_tool(None))
    assert timing in reasons(analyst.submit_analysis_tool("when the quote is issued"))
    assert timing in reasons(analyst.SUBMIT_ANALYSIS)  # the shared spec is left as it was


def _flight_submission(drops: list[str]) -> dict:
    return {
        "problem_type": "regression", "problem_type_rationale": "A delay in minutes.",
        "drop_columns": [{"column": c, "reason": "leakage", "evidence": f"{c} is recorded at or after touchdown."}
                         for c in drops],
        "findings": [], "open_questions": [], "summary": "Delays carry over from departure.",
    }


@pytest.mark.parametrize("second_drops", [["TaxiIn", "AirTime"], ["TaxiIn"]])
def test_kept_columns_that_rebuild_the_target_are_sent_back_once(monkeypatch, second_drops):
    llm = ScriptedLLM([[("submit_analysis", _flight_submission(["TaxiIn"]))],
                       [("submit_analysis", _flight_submission(second_drops))]])
    _use_llm(monkeypatch, llm)
    result = orchestrator.run_agent(make_run(flight_times()), {"task": "auto", "target": "ArrDelay"})
    assert result["status"] == "succeeded", result.get("error")

    # Keeping the time in the air lets the columns add up to the delay: the analysis comes back.
    sent_back = llm.analyst_transcripts[1][-1]["content"]
    assert sent_back.startswith("Not accepted yet") and "AirTime" in sent_back
    analysis = result["extra"]["analysis"]
    assert {d["column"] for d in analysis["drop_columns"]} == set(second_drops)
    assert any(note.startswith("Sent the analysis back once") for note in analysis["notes"])
    # Only once: a second submission that still keeps it is accepted, with a critical finding.
    leftover = [f for f in analysis["findings"] if "plain arithmetic" in f["finding"]]
    if "AirTime" in second_drops:
        assert leftover == []
    else:
        assert leftover[0]["severity"] == "critical" and leftover[0]["verified"]
        assert "AirTime" in leftover[0]["evidence"]


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
    result = orchestrator.run_agent(uploaded_run, {"task": "classification", "target": "smoker"})
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
    def gpu_down(function, data, specs, run_id):
        return [RuntimeError("No NVIDIA GPU visible in the training container; refusing to train on CPU.")] * len(specs)

    monkeypatch.setattr(train, "_gpu_map", gpu_down)
    result = orchestrator.run_agent(uploaded_run, "charges")
    assert result["status"] == "failed" and "refusing to train on CPU" in result["error"]
    assert not (storage.run_dir(uploaded_run) / "model.joblib").exists()


def test_one_failed_model_group_does_not_sink_the_run(uploaded_run, monkeypatch):
    real = train._gpu_map

    def half_down(function, data, specs, run_id):
        results = real(function, data, specs[:1], run_id)
        return results + [RuntimeError("container lost")] * (len(specs) - 1)

    monkeypatch.setattr(train, "_gpu_map", half_down)
    result = orchestrator.run_agent(uploaded_run, "charges")
    assert result["status"] == "succeeded", result.get("error")
    assert any("container lost" in (m["reason"] or "") for m in result["extra"]["excluded_models"])


def test_legacy_csv_run_is_migrated_on_first_access():
    run_id = storage.new_run_id()
    shutil.copy(INSURANCE_CSV, storage.run_dir(run_id) / "dataset.csv")
    df = storage.load_dataset(run_id)
    assert df.shape == (1338, 7)
    assert (storage.run_dir(run_id) / "dataset.parquet").exists()
    assert storage.read_json(run_id, "ingest.json")["n_rows"] == 1338


@pytest.fixture
def client(monkeypatch):
    from app.main import app

    started = []
    monkeypatch.setattr(jobs, "start", lambda run_id, spec, background: started.append(spec) or {"executor": "local"})
    c = TestClient(app)
    c.started = started
    return c


def test_upload_excel_and_start(client):
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
    r = client.post(f"/runs/{run_id}/start", json={"target": "charges"})
    assert r.status_code == 200 and client.started[-1] == {"task": "auto", "target": "charges"}
    assert client.post(f"/runs/{run_id}/start", json={"target": "charges"}).status_code == 409  # already queued


@pytest.mark.parametrize("body,message", [
    ({"task": "regression", "target": "region"}, "numeric target"),
    # Removed tasks: the 422 lists the three that remain.
    ({"task": "clustering"}, "regression"),
    ({"task": "forecasting", "target": "charges"}, "regression"),
    ({"task": "image_classification"}, "regression"),
    ({"task": "magic", "target": "charges"}, ""),
])
def test_start_validates_the_task(client, uploaded_run, body, message):
    r = client.post(f"/runs/{uploaded_run}/start", json=body)
    assert r.status_code in (400, 422) and message in r.text


def test_a_target_is_required(client, uploaded_run):
    r = client.post(f"/runs/{uploaded_run}/start", json={})
    assert r.status_code == 400 and "target column" in r.text


def test_start_passes_on_the_prediction_time(client, uploaded_run):
    body = {"target": "charges", "prediction_time": "x" * 301}
    assert client.post(f"/runs/{uploaded_run}/start", json=body).status_code == 422
    body["prediction_time"] = "  when the quote\n is issued "
    assert client.post(f"/runs/{uploaded_run}/start", json=body).status_code == 200
    assert client.started[-1] == {"task": "auto", "target": "charges", "prediction_time": "when the quote is issued"}
    assert storage.read_status(uploaded_run)["prediction_time"] == "when the quote is issued"


def test_cancel_and_diagnostics(client, uploaded_run):
    assert client.post(f"/runs/{uploaded_run}/cancel").status_code == 409  # not running
    client.post(f"/runs/{uploaded_run}/start", json={"target": "charges"})
    assert client.post(f"/runs/{uploaded_run}/cancel").status_code == 409  # local runs can't be cancelled
    diag = client.get(f"/runs/{uploaded_run}/diagnostics").json()
    assert diag["status"] == "queued" and diag["gpu"] == "NVIDIA H200"
    assert "speed" not in diag and "cpu" not in diag  # no invented numbers


def test_upload_rejects_bad_files_and_cleans_up(client):
    assert client.post("/upload", files={"file": ("x.pdf", b"%PDF")}).status_code == 400
    r = client.post("/upload", files={"file": ("x.csv", b"")})
    assert r.status_code == 400 and "empty" in r.json()["detail"]
    assert not any((storage.settings.storage_dir / "runs").iterdir())


def test_run_ids_are_validated(client):
    for bad in ("..", "abc", "ABCDEF123456"):
        assert client.get(f"/runs/{bad}/status").status_code == 404


def test_downloads_cleaned_csv_and_working_model_package(uploaded_run, client, tmp_path, monkeypatch):
    # Files inside a Modal image have a 1970 mtime; packaging must not choke on it.
    runtime = tmp_path / "runtime.py"
    shutil.copy(export.RUNTIME_SOURCE, runtime)
    os.utime(runtime, (0, 0))
    monkeypatch.setattr(export, "RUNTIME_SOURCE", runtime)

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
        assert {"model.joblib", "runtime.py", "metadata.json", "predict.py", "requirements.txt", "README.md"} <= names
        assert names & {"model_weights.json", "model_weights.cbm", "model_weights.npz"}
        z.extractall(tmp_path / "pkg")
    assert "scikit-learn==" in (tmp_path / "pkg" / "requirements.txt").read_text()

    # The packaged predict.py runs as shipped, importing only its own runtime.py.
    cleaned.drop(columns=["smoker"]).head(20).to_csv(tmp_path / "in.csv", index=False)
    subprocess.run([sys.executable, "predict.py", str(tmp_path / "in.csv"), str(tmp_path / "out.csv")],
                   cwd=tmp_path / "pkg", check=True, capture_output=True)
    preds = pd.read_csv(tmp_path / "out.csv")
    assert set(preds["prediction"]) <= {"no", "yes"}

    assert client.get(f"/runs/{uploaded_run}/download/weights").status_code == 404


def test_cancel_stops_the_job_and_its_gpu_calls(monkeypatch):
    import modal

    cancelled = []

    class FakeCall:
        def __init__(self, call_id):
            self.call_id = call_id

        def cancel(self, terminate_containers=False):
            cancelled.append((self.call_id, terminate_containers))

    monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(FakeCall))
    assert jobs.cancel({"executor": "modal", "call_id": "fc-job", "gpu_calls": ["fc-a", "fc-b"]})
    assert cancelled == [("fc-job", True), ("fc-a", True), ("fc-b", True)]
    assert not jobs.cancel({"executor": "local"})
