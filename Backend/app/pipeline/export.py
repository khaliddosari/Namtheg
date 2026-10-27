"""Downloadable deliverables: the cleaned dataset (CSV) and the model package
(zip)."""
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from app import storage

CLEANED_CSV = "cleaned.csv"
MODEL_PACKAGE = "model_package.zip"
RUNTIME_SOURCE = Path(__file__).resolve().parent.parent / "training" / "runtime.py"

PREDICT_PY = '''"""Predictions with an exported Namtheg model.

    pip install -r requirements.txt

    # a CSV with the columns in metadata.json -> feature_cols
    python predict.py input.csv predictions.csv
"""
import sys

import joblib
import numpy as np
import pandas as pd

from runtime import load_bundle


def main(argv):
    # joblib runs code while loading: only load model files from a source you trust.
    bundle = load_bundle(joblib.load("model.joblib"))
    model, labels = bundle["model"], bundle.get("class_labels")

    df = pd.read_csv(argv[0])
    missing = [c for c in model.feature_cols if c not in df.columns]
    if missing:
        sys.exit(f"input is missing columns: {missing}")
    pred = np.asarray(model.predict(df)).ravel()
    out = df.copy()
    if labels:
        out["prediction"] = [labels[int(p)] for p in pred]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            for i, label in enumerate(labels):
                out[f"probability_{label}"] = proba[:, i]
    else:
        out["prediction"] = pred
    out.to_csv(argv[1], index=False)
    print(f"wrote {len(out)} rows to {argv[1]}")


if __name__ == "__main__":
    main(sys.argv[1:])
'''

README_MD = """# {model_name} model ({task})

Trained by Namtheg on {trained_on} ({created_at}). Run `{run_id}`.

| File | What it is |
|------|------------|
| `model.joblib` | The model in portable form: fitted preprocessing plus weights. |
| `runtime.py` | Loads `model.joblib` and predicts; the same code Namtheg's live endpoint runs. |
| `predict.py` | Command-line predictions (see its docstring). |
| `metadata.json` | Inputs, classes, settings, metrics, and exact library versions. |
| `requirements.txt` | The libraries this model needs, at the versions it was trained with. |
{extra_files}
## Use it

```bash
pip install -r requirements.txt
{usage}
```

`joblib.load` executes code, so only load `model.joblib` from a source you trust.
{notes}
## Performance

Selection metric: **{metric}** ({direction}). {evaluation}: **{score}**.{baseline}
"""

USAGE = {
    "classification": "python predict.py input.csv predictions.csv",
    "regression": "python predict.py input.csv predictions.csv",
}
EVALUATION = {
    "classification": "Cross-validated", "regression": "Cross-validated",
}


def export_cleaned_csv(run_id: str) -> str:
    """The data exactly as the models received it: after column drops, with
    rows missing the target removed. Only real rows: class weighting happens
    inside training, not in the data. UTF-8 with BOM so Excel shows Arabic and
    other non-Latin text correctly."""
    frame = storage.load_engineered(run_id)
    frame.to_csv(storage.run_dir(run_id) / CLEANED_CSV, index=False, encoding="utf-8-sig")
    storage.persist(run_id, CLEANED_CSV)
    return CLEANED_CSV


def _requirements(meta: dict) -> str:
    """Only the libraries this model needs. Read from model_meta.json: the
    backend never unpickles a model (its Python and library versions differ
    from the training images')."""
    kind, versions = meta["estimator_kind"], meta.get("library_versions") or {}
    pkgs = ["numpy", "pandas", "scikit-learn", "joblib"]
    if kind == "xgboost-json":
        pkgs.append("xgboost")
    if kind == "catboost-cbm":
        pkgs.append("catboost")
    return "\n".join(f"{p}=={versions[p]}" if p in versions else p for p in pkgs) + "\n"


def build_model_package(run_id: str, result: dict) -> str:
    meta = storage.read_json(run_id, "model_meta.json")
    task = meta.get("task") or result.get("problem_type")
    extra = result["extra"]
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    hardware = (meta.get("hardware") or {}).get("gpu", "GPU")
    lower_better = result.get("higher_is_better") is False
    base = extra.get("baseline") or {}
    metadata = {
        **meta, "run_id": run_id, "created_at": created_at, "task": task,
        "selection_metric": result["score_metric"], "higher_is_better": not lower_better,
        "metrics": {k: extra.get(k) for k in ("cv_mean", "test_score", "test_metrics", "train_metrics", "baseline")
                    if extra.get(k) is not None},
        "imbalance_handling": extra.get("imbalance_applied"),
    }
    extra_files = ""
    if meta.get("weights_file"):
        extra_files += f"| `{meta['weights_file']}` | The trained model's learned parameters on their own (no pickle). |\n"
    readme = README_MD.format(
        model_name=meta["model_name"], task=task.replace("_", " "), trained_on=hardware, created_at=created_at,
        run_id=run_id, extra_files=extra_files, usage=USAGE[task],
        notes=(f"\nClass predictions are indexes into `class_labels` in metadata.json: {meta['class_labels']}.\n"
               if meta.get("class_labels") else "")
              + ("\nPass raw values: the model cleans its inputs the way training did (`cleaning` in metadata.json).\n"
                 if meta.get("cleaning") else ""),
        metric=result["score_metric"], direction="lower is better" if lower_better else "higher is better",
        evaluation=EVALUATION[task], score=result["accuracy_score"],
        baseline=(f" Feature-blind baseline ({base['name']}): {base.get('cv_mean')}." if base.get("name") else ""),
    )

    buf = io.BytesIO()
    # Modal images give their files a 1970 mtime, which ZIP can't store: clamp to 1980.
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:
        z.write(storage.artifact_path(run_id, "model.joblib"), "model.joblib")
        z.write(RUNTIME_SOURCE, "runtime.py")
        if meta.get("weights_file"):
            z.write(storage.artifact_path(run_id, meta["weights_file"]), meta["weights_file"])
        z.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
        z.writestr("predict.py", PREDICT_PY)
        z.writestr("requirements.txt", _requirements(meta))
        z.writestr("README.md", readme)
    (storage.run_dir(run_id) / MODEL_PACKAGE).write_bytes(buf.getvalue())
    storage.persist(run_id, MODEL_PACKAGE)
    return MODEL_PACKAGE
