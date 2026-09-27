"""Live check of the deployed namtheg-train-gpu service on real H200 hardware.
Calls the exact deployed functions (no ephemeral app), with tiny synthetic
data, keeping cost minimal while exercising every code path for real."""
import io
import json
import time
import zipfile

import modal
import numpy as np
import pandas as pd
from PIL import Image

train_classic = modal.Function.from_name("namtheg-train-gpu", "train_classic")
train_deep = modal.Function.from_name("namtheg-train-gpu", "train_deep")
embed_text = modal.Function.from_name("namtheg-train-gpu", "embed_text")

results = {}


def record(name, fn):
    t0 = time.time()
    try:
        out = fn()
        results[name] = {"ok": True, "wall_s": round(time.time() - t0, 1), "out": out}
        print(f"PASS {name} ({results[name]['wall_s']}s)")
    except Exception as e:
        results[name] = {"ok": False, "wall_s": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"}
        print(f"FAIL {name} ({results[name]['wall_s']}s): {type(e).__name__}: {str(e)[:300]}")


# ---- tabular: GBDT group (native CUDA: XGBoost, CatBoost) ----
rng = np.random.default_rng(0)
n = 300
df = pd.DataFrame({
    "x1": rng.normal(size=n), "x2": rng.normal(size=n),
    "cat": rng.choice(["a", "b", "c"], size=n),
    "y_reg": rng.normal(size=n),
})
df["y_cls"] = (df["x1"] + rng.normal(0, 0.3, n) > 0).astype(int).astype(str)
buf = io.BytesIO()
df.drop(columns=["y_cls"]).to_parquet(buf)
tabular_reg_data = buf.getvalue()
buf2 = io.BytesIO()
df.drop(columns=["y_reg"]).to_parquet(buf2)
tabular_cls_data = buf2.getvalue()

plan_fast = {"tuning_trials": 2, "tuning_timeout_seconds": 30}

record("tabular_gbdt (XGBoost/XGBoost-leafwise/CatBoost, native CUDA)", lambda: train_classic.remote(
    tabular_reg_data, {"task": "regression", "target": "y_reg",
                        "candidates": ["XGBoost", "XGBoost Leaf-wise", "CatBoost"], "plan": plan_fast}))

record("tabular_cuml (SVM/KNN/Linear via cuml.accel)", lambda: train_classic.remote(
    tabular_cls_data, {"task": "classification", "target": "y_cls",
                        "candidates": ["SVM", "KNN", "Linear"], "plan": plan_fast}))

# ---- clustering (cuml.accel: K-Means/HDBSCAN/DBSCAN) ----
buf3 = io.BytesIO()
df[["x1", "x2"]].to_parquet(buf3)
record("clustering (K-Means/HDBSCAN/DBSCAN via cuml.accel)", lambda: train_classic.remote(
    buf3.getvalue(), {"task": "clustering", "candidates": ["K-Means", "HDBSCAN", "DBSCAN"], "plan": {}}))

# ---- text embeddings ----
record("text embeddings (pretrained multilingual encoder)", lambda: embed_text.remote(
    {"review": ["great product", "سيء جدا", "okay I guess"]}, "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))

# ---- forecasting (torch: one fast candidate) ----
t = pd.date_range("2025-01-01", periods=90, freq="D")
ts = pd.DataFrame({"series_id": "s", "timestamp": t, "value": 10 + np.sin(np.arange(90) / 7 * 2 * np.pi) + rng.normal(0, 0.3, 90)})
buf4 = io.BytesIO()
ts.to_parquet(buf4)
record("forecasting (LSTM on GPU)", lambda: train_deep.remote(
    buf4.getvalue(), {"task": "forecasting", "candidate": "LSTM", "horizon": 7, "freq": "D",
                       "season": 7, "input_length": 21, "windows": 2}))

# ---- images (fine-tune a tiny CNN) ----
imgbuf = io.BytesIO()
rows = []
with zipfile.ZipFile(imgbuf, "w") as z:
    for label, rgb in {"red": (220, 30, 30), "blue": (30, 60, 220)}.items():
        for i in range(8):
            arr = np.clip(np.array(rgb) + rng.normal(0, 20, (24, 24, 3)), 0, 255).astype("uint8")
            b = io.BytesIO()
            Image.fromarray(arr).save(b, "PNG")
            path = f"images/{label}/{i}.png"
            z.writestr(path, b.getvalue())
            rows.append({"image": path, "label": label})
    mb = io.BytesIO()
    pd.DataFrame(rows).to_parquet(mb)
    z.writestr("dataset.parquet", mb.getvalue())
record("image classification (fine-tune ResNet-50 on GPU)", lambda: train_deep.remote(
    imgbuf.getvalue(), {"task": "image_classification", "candidate": "ResNet-50", "plan": {}}))

print("\n" + "=" * 70)
summary = {k: {kk: vv for kk, vv in v.items() if kk != "out"} for k, v in results.items()}
print(json.dumps(summary, indent=2))

with open("live_gpu_check_full.json", "w") as f:
    def clean(o):
        if isinstance(o, bytes):
            return f"<{len(o)} bytes>"
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    json.dump(clean(results), f, indent=2, default=str)
print("\nFull output saved to live_gpu_check_full.json")
