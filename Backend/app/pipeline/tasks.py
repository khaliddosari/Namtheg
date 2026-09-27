"""Which learning task a run is, from the user's choices, validated before the
run starts so a bad setup fails in the request, not after GPU time is spent.

- An image zip upload is always image classification (target: its folders).
- No target column means clustering.
- A date column plus a numeric target means forecasting.
- A target alone is classification or regression; "auto" lets the detector
  (and then the analyst agent, with evidence) decide.
"""
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from app import storage

TASKS = ("auto", "classification", "regression", "clustering", "forecasting", "image_classification")
MAX_CLASSES = 100
MAX_HORIZON = 1000


class TaskError(ValueError):
    """The requested task doesn't fit the data. The message is safe to show the user."""


def resolve(run_id: str, task: str, target: str | None, date_column: str | None = None,
            series_id_column: str | None = None, horizon: int | None = None) -> dict:
    if task not in TASKS:
        raise TaskError(f"Unknown task {task!r}. Choose one of: {', '.join(TASKS)}.")
    ingest = storage.read_json(run_id, "ingest.json") or {}
    if ingest.get("source_format") == "images":
        if task not in ("auto", "image_classification"):
            raise TaskError("An image upload can only be used for image classification.")
        return {"task": "image_classification", "target": "label"}
    if task == "image_classification":
        raise TaskError("Image classification needs a zip of images with one folder per class.")

    columns = storage.dataset_columns(run_id)

    def require(col: str | None, role: str) -> str:
        if not col:
            raise TaskError(f"Choose the {role} column.")
        if col not in columns:
            raise TaskError(f"{role.capitalize()} column {col!r} is not in the data.")
        return col

    if task == "forecasting":
        spec = {"task": "forecasting", "target": require(target, "target"),
                "date_column": require(date_column, "date"),
                "series_id_column": require(series_id_column, "series id") if series_id_column else None,
                "horizon": horizon}
        if len({spec["target"], spec["date_column"], spec["series_id_column"]} - {None}) < (
                3 if series_id_column else 2):
            raise TaskError("The target, date and series id columns must all be different.")
        if horizon is not None and not 1 <= int(horizon) <= MAX_HORIZON:
            raise TaskError(f"The horizon must be between 1 and {MAX_HORIZON} steps.")
        y = storage.load_dataset(run_id)[spec["target"]]
        if not is_numeric_dtype(y) or is_bool_dtype(y):
            raise TaskError(f"Forecast target {target!r} must be numeric.")
        return spec

    if task == "clustering" or (task == "auto" and not target):
        if target:
            raise TaskError("Clustering finds groups without a target; leave the target empty.")
        return {"task": "clustering", "target": None}

    target = require(target, "target")
    if task in ("classification", "regression"):
        y = storage.load_dataset(run_id)[target].dropna()
        if task == "regression" and (not is_numeric_dtype(y) or is_bool_dtype(y)):
            raise TaskError(f"Regression needs a numeric target; {target!r} is not numeric.")
        if task == "classification" and y.nunique() > MAX_CLASSES:
            raise TaskError(f"{target!r} has {y.nunique()} distinct values; classification supports up to "
                            f"{MAX_CLASSES}. Did you mean regression?")
    return {"task": task, "target": target}
