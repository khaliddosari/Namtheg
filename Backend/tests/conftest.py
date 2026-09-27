import json
import shutil
from pathlib import Path

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


FAST_PARAMS = {
    "XGBoost": {"n_estimators": 40, "learning_rate": 0.2, "max_depth": 4},
    "XGBoost Leaf-wise": {"n_estimators": 40, "learning_rate": 0.2, "max_leaves": 15},
    "CatBoost": {"iterations": 60, "learning_rate": 0.2, "depth": 4},
}


@pytest.fixture(autouse=True)
def cpu_training_harness(monkeypatch):
    """Tests can't reach the GPU service, so its remote call is replaced by the
    same training engine run in-process on CPU, with small models and one
    tuning trial. Production has no such path: train._remote_train only calls
    the GPU service."""
    from app.pipeline import train
    from app.training import core

    def run_locally(data, target, problem_type, plan):
        out = core.run_training(data, target, problem_type, plan, device="cpu")
        out["hardware"] = {"gpu": "CPU test harness", "requested": "H200"}
        return out

    monkeypatch.setattr(core, "DEFAULT_PARAMS", FAST_PARAMS)
    monkeypatch.setattr(core, "_search_space", lambda name, trial: {
        **FAST_PARAMS[name], "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
    })
    monkeypatch.setattr(settings, "tuning_trials", 1)
    monkeypatch.setattr(train, "_remote_train", run_locally)
    yield


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
    storage.write_status(run_id, "uploaded", filename="insurance.csv")
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
            text = (f"{m['name']} reached a cross-validated {m['metric']} of {m['cv_score']:.3f} "
                    f"versus {facts['baseline']['cv_mean']:.3f} for the baseline.")
            if self._hallucinate:
                text += f" It is accurate {self._hallucinate} of the time."
            return AssistantTurn(text=text, tool_calls=[], provider="fake", model="scripted")
        # Hyperparameter proposals (only used for large datasets): stop immediately.
        return AssistantTurn(text='{"parameters": {}, "reasoning": "stop", "stop_tuning": true}',
                             tool_calls=[], provider="fake", model="scripted")

    def text(self, system, user, effort=None):
        return self.complete(system, [{"role": "user", "content": user}], effort=effort).text

    def summary(self):
        return {"primary": "scripted", "fallback": None, "fell_back": False,
                "calls": len(self.calls), "failed_calls": 0, "models_used": ["scripted"]}
