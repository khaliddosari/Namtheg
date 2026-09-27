import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.llm import AssistantTurn, ToolCall

REPO_ROOT = Path(__file__).resolve().parents[2]
INSURANCE_CSV = REPO_ROOT / "insurance.csv"


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Every test gets its own storage dir, the local sandbox, no real LLM keys,
    and R2 off."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    settings.storage_dir.mkdir()
    monkeypatch.setattr(settings, "sandbox_backend", "local")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    monkeypatch.setattr(settings, "r2_bucket", "")
    yield


def fake_encode_text(values, encoder_name, batch_size=128):
    """Deterministic stand-in for the pretrained sentence encoder: hashed bag
    of words, unit-normalised. Same text -> same vector, like the real one."""
    out = np.zeros((len(values), 64), dtype=np.float32)
    for i, v in enumerate(values):
        for word in str("" if v is None else v).lower().split():
            out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms > 0, norms, 1)


@pytest.fixture(autouse=True)
def cpu_training_harness(monkeypatch):
    """Tests can't reach the GPU service, so its two seams run the same engines
    in-process on CPU, with small models. Production has no such path:
    train._gpu_map / _gpu_call only talk to the GPU service."""
    from app.pipeline import train
    from app.training import forecasting, runtime, tabular, text, vision

    monkeypatch.setattr(tabular, "MAX_TREES", 80)
    monkeypatch.setattr(tabular, "EARLY_STOPPING_ROUNDS", 10)
    monkeypatch.setattr(settings, "tuning_trials", 2)
    monkeypatch.setattr(forecasting, "MAX_EPOCHS", 3)
    monkeypatch.setattr(forecasting, "CANDIDATES", ("LSTM", "TCN", "XGBoost (lags)"))
    monkeypatch.setattr(vision, "CANDIDATES", {"ResNet-18 (test)": "resnet18"})
    monkeypatch.setattr(vision, "PRETRAINED", False)
    monkeypatch.setattr(vision, "MAX_EPOCHS", 2)
    monkeypatch.setattr(runtime, "encode_text", fake_encode_text)
    monkeypatch.setattr(text, "encode_text", fake_encode_text)

    def run_locally(function, data, specs, run_id):
        from app.training import clustering

        results = []
        for spec in specs:
            if function == "train_classic":
                engine = clustering if spec["task"] == "clustering" else tabular
            else:
                engine = forecasting if spec["task"] == "forecasting" else vision
            try:
                out = engine.run(data, spec, "cpu")
                out["hardware"] = {"gpu": "CPU test harness", "requested": "H200"}
                results.append(out)
            except Exception as e:
                results.append(e)
        return results

    def call_locally(function, *args):
        assert function == "embed_text"
        return text.embed_columns(*args)

    monkeypatch.setattr(train, "_gpu_map", run_locally)
    monkeypatch.setattr(train, "_gpu_call", call_locally)
    yield


def make_run(frame, filename: str = "data.csv") -> str:
    """A run whose upload (a DataFrame saved as CSV) has been ingested."""
    from app import storage
    from app.data.ingest import ingest_file

    run_id = storage.new_run_id()
    raw = storage.raw_upload_path(run_id, ".csv")
    frame.to_csv(raw, index=False)
    report = ingest_file(raw, storage.run_dir(run_id) / storage.DATASET_FILE, filename)
    storage.persist(run_id, "raw/source.csv")  # as the /upload endpoint does
    storage.persist(run_id, storage.DATASET_FILE)
    storage.write_json(run_id, "ingest.json", report)
    storage.write_status(run_id, "uploaded", filename=filename, source_format=report["source_format"])
    return run_id


@pytest.fixture
def uploaded_run():
    """A run whose upload (insurance.csv) has been ingested to Parquet."""
    from app import storage
    from app.data.ingest import ingest_file

    run_id = storage.new_run_id()
    raw = storage.raw_upload_path(run_id, ".csv")
    shutil.copy(INSURANCE_CSV, raw)
    report = ingest_file(raw, storage.run_dir(run_id) / storage.DATASET_FILE, "insurance.csv")
    storage.persist(run_id, "raw/source.csv")  # as the /upload endpoint does
    storage.persist(run_id, storage.DATASET_FILE)
    storage.write_json(run_id, "ingest.json", report)
    storage.write_status(run_id, "uploaded", filename="insurance.csv", source_format="csv")
    return run_id


class ScriptedLLM:
    """Stands in for LLMClient. `analyst_script` is a list of turns (each a list
    of (tool_name, args) calls) replayed for the analyst; the justification is
    written from the FACTS it receives, optionally with an invented number."""

    def __init__(self, analyst_script, hallucinate: str | None = None):
        self.configured = True
        self.calls: list[dict] = []
        self._script = list(analyst_script)
        self._hallucinate = hallucinate
        self.analyst_transcripts: list[list] = []

    def complete(self, system, transcript, tools=None, tool_choice=None, effort=None):
        self.calls.append({"provider": "fake", "model": "scripted", "ok": True, "seconds": 0})
        if tools and any(t.name == "submit_analysis" for t in tools):
            self.analyst_transcripts.append(list(transcript))
            calls = self._script.pop(0)
            return AssistantTurn(
                text="",
                tool_calls=[ToolCall(id=f"call_{len(self.calls)}_{i}", name=n, raw_arguments=json.dumps(a), arguments=a)
                            for i, (n, a) in enumerate(calls)],
                provider="fake", model="scripted",
            )
        if "FACTS:" in transcript[0]["content"]:
            facts = json.loads(transcript[0]["content"].split("FACTS:\n", 1)[1])
            m = facts["model"]
            text = f"{m['name']} reached {m['metric']} {m['score']:.3f}"
            if facts.get("baseline") and facts["baseline"].get("cv_mean") is not None:
                text += f" versus {facts['baseline']['cv_mean']:.3f} for the baseline"
            text += "."
            if self._hallucinate:
                text += f" It is accurate {self._hallucinate} of the time."
            return AssistantTurn(text=text, tool_calls=[], provider="fake", model="scripted")
        return AssistantTurn(text="", tool_calls=[], provider="fake", model="scripted")

    def text(self, system, user, effort=None):
        return self.complete(system, [{"role": "user", "content": user}], effort=effort).text

    def summary(self):
        return {"primary": "scripted", "fallback": None, "fell_back": False,
                "calls": len(self.calls), "failed_calls": 0, "models_used": ["scripted"]}
