"""Downloadable deliverables: the cleaned dataset (CSV) and the model package (zip)."""
import inspect
import io
import json
import zipfile
from datetime import datetime, timezone

from app import storage
from app.training.core import load_bundle

CLEANED_CSV = "cleaned.csv"
MODEL_PACKAGE = "model_package.zip"

PREDICT_PY = '''"""Batch predictions with the exported Namtheg model.

    pip install -r requirements.txt
    python predict.py input.csv predictions.csv

input.csv needs the columns listed under "feature_cols" in metadata.json.
"""
import sys

import joblib
import numpy as np
import pandas as pd


# __LOAD_BUNDLE__


# joblib runs code while loading: only load model files from a source you trust.
bundle = load_bundle(joblib.load("model.joblib"))
model, features, labels = bundle["model"], bundle["feature_cols"], bundle["class_labels"]

df = pd.read_csv(sys.argv[1])
missing = [c for c in features if c not in df.columns]
if missing:
    sys.exit(f"input is missing columns: {missing}")

X = df[features]
pred = np.asarray(model.predict(X)).ravel()
out = df.copy()
if labels:
    out["prediction"] = [labels[int(p)] for p in pred]
    proba = model.predict_proba(X)
    for i, label in enumerate(labels):
        out[f"probability_{label}"] = proba[:, i]
else:
    out["prediction"] = pred
out.to_csv(sys.argv[2], index=False)
print(f"wrote {len(out)} predictions to {sys.argv[2]}")
'''

README_MD = """# {model_name} model for `{target}`

Trained by Namtheg on {trained_on} ({created_at}). Run `{run_id}`.

| File | What it is |
|------|------------|
| `model.joblib` | The fitted preprocessing plus the model's weights. `predict.py` rebuilds the full pipeline from it: raw columns in, predictions out. |
| `{weights_file}` | The trained model's weights in {weights_format_desc} format, loadable without Python pickles. |
| `metadata.json` | Features and dtypes, class labels, preprocessing, hyperparameters, metrics, library versions. |
| `predict.py` | Batch predictions from a CSV. |
| `requirements.txt` | Exact library versions the model was trained with. |

## Use the pipeline

```bash
pip install -r requirements.txt
python predict.py input.csv predictions.csv
```

`joblib.load` executes code, so only load `model.joblib` from a source you trust. The weights
file below needs no pickle at all.

## Use the weights directly

{weights_usage}

The weights expect the *encoded* features, in the order given by
`metadata.json` → `preprocessing.encoded_feature_names`: numeric columns pass
through unchanged; `one_hot` columns expand to one 0/1 column per listed
category (binary columns keep a single column for the second category);
`ordinal` columns become the index of the value in its category list, or -1
for a category not seen in training.
{labels_note}
## Performance

Selection metric: **{metric}**. Cross-validated: **{cv_mean}**. Held-out test set: **{test_score}**.
Feature-blind baseline ({baseline_name}): {baseline_cv} (CV).
"""

WEIGHTS_USAGE = {
    "xgboost-json": (
        "```python\nimport xgboost as xgb\nbooster = xgb.Booster()\nbooster.load_model(\"model_weights.json\")\n"
        "pred = booster.predict(xgb.DMatrix(encoded_features))\n```"
    ),
    "catboost-cbm": (
        "```python\nfrom catboost import CatBoost\nmodel = CatBoost()\nmodel.load_model(\"model_weights.cbm\")\n"
        "pred = model.predict(encoded_features)\n```"
    ),
}


def export_cleaned_csv(run_id: str) -> str:
    """The dataset exactly as the models received it: after the analyst's and
    the pipeline's column drops, with rows missing the target removed. It holds
    only real rows; class weighting happens inside training, not in the data.
    UTF-8 with BOM so Excel shows Arabic and other non-Latin text correctly."""
    storage.load_engineered(run_id).to_csv(storage.run_dir(run_id) / CLEANED_CSV, index=False, encoding="utf-8-sig")
    storage.persist(run_id, CLEANED_CSV)
    return CLEANED_CSV


def build_model_package(run_id: str, result: dict) -> str:
    meta = storage.read_json(run_id, "model_meta.json")
    extra = result["extra"]
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    hardware = (meta.get("hardware") or {}).get("gpu", "GPU")
    versions = meta["library_versions"]
    library = "xgboost" if meta["weights_format"] == "xgboost-json" else "catboost"
    requirements = "\n".join(
        f"{pkg}=={versions[pkg]}" for pkg in ("numpy", "pandas", "scikit-learn", "joblib", library)
    ) + "\n"
    labels = meta.get("class_labels")
    metadata = {
        **meta,
        "run_id": run_id,
        "created_at": created_at,
        "selection_metric": result["score_metric"],
        "metrics": {
            "cv_mean": extra.get("cv_mean"),
            "test": extra.get("test_metrics"),
            "train": extra.get("train_metrics"),
            "baseline": extra.get("baseline"),
        },
        "imbalance_handling": extra.get("imbalance_applied"),
    }
    readme = README_MD.format(
        model_name=meta["model_name"],
        target=meta["target"],
        trained_on=hardware,
        created_at=created_at,
        run_id=run_id,
        weights_file=meta["weights_file"],
        weights_format_desc="XGBoost JSON" if library == "xgboost" else "CatBoost native (.cbm)",
        weights_usage=WEIGHTS_USAGE[meta["weights_format"]],
        labels_note=(
            f"\nClass predictions are indexes into `class_labels` in metadata.json: {labels}.\n" if labels else ""
        ),
        metric=result["score_metric"],
        cv_mean=extra.get("cv_mean"),
        test_score=extra.get("test_score"),
        baseline_name=(extra.get("baseline") or {}).get("name", "n/a"),
        baseline_cv=(extra.get("baseline") or {}).get("cv_mean", "n/a"),
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(storage.artifact_path(run_id, "model.joblib"), "model.joblib")
        z.write(storage.artifact_path(run_id, meta["weights_file"]), meta["weights_file"])
        z.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
        # Same loader the inference endpoint uses, so the package needs nothing from Namtheg.
        z.writestr("predict.py", PREDICT_PY.replace("# __LOAD_BUNDLE__", inspect.getsource(load_bundle)))
        z.writestr("requirements.txt", requirements)
        z.writestr("README.md", readme)
    (storage.run_dir(run_id) / MODEL_PACKAGE).write_bytes(buf.getvalue())
    storage.persist(run_id, MODEL_PACKAGE)
    return MODEL_PACKAGE
