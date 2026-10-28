"""Which learning task a run is, from the user's choices, validated before the
run starts so a bad setup fails in the request, not after GPU time is spent.

Every run predicts a target column: classification or regression. "auto" lets
the detector (and then the analyst agent, with evidence) decide which.
"""
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from app import storage

TASKS = ("auto", "classification", "regression")
MAX_CLASSES = 100


class TaskError(ValueError):
    """The requested task doesn't fit the data. The message is safe to show the user."""


def resolve(run_id: str, task: str, target: str | None, prediction_time: str | None = None) -> dict:
    if task not in TASKS:
        raise TaskError(f"Unknown task {task!r}. Choose one of: {', '.join(TASKS)}.")
    ingest = storage.read_json(run_id, "ingest.json") or {}
    if ingest.get("source_format") == "images":
        raise TaskError("Image datasets are no longer supported; upload a table.")
    if not target:
        raise TaskError("Choose the target column.")
    if target not in storage.dataset_columns(run_id):
        raise TaskError(f"Target column {target!r} is not in the data.")
    if task in ("classification", "regression"):
        y = storage.load_columns(run_id, [target])[target].dropna()
        if task == "regression" and (not is_numeric_dtype(y) or is_bool_dtype(y)):
            raise TaskError(f"Regression needs a numeric target; {target!r} is not numeric.")
        if task == "classification" and y.nunique() > MAX_CLASSES:
            raise TaskError(f"{target!r} has {y.nunique()} distinct values; classification supports up to "
                            f"{MAX_CLASSES}. Did you mean regression?")
    spec = {"task": task, "target": target}
    if prediction_time and prediction_time.strip():
        spec["prediction_time"] = " ".join(prediction_time.split())
    return spec
