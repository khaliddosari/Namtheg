"""Mock of the Namtheg backend (Backend/app/main.py) for the e2e scripts.

Shapes and status codes follow the real API: errors are {"detail": ...}, a
result that isn't written yet is 409, unknown runs are 404, and diagnostics'
elapsed_seconds counts from queued_at forever. Seeded run ids cover the
scenarios (see seeded()); POST /__mock/config switches R2 direct upload, a
response delay and the way /uploads/{id}/complete behaves (see complete());
POST /__mock/reset puts everything back.

Run from this folder with any Python that has fastapi and uvicorn, for example
the backend's virtualenv:

    ../../Backend/.venv/Scripts/python -m uvicorn mock_backend:app --port 8001
"""
import asyncio
import csv
import json
import random
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response

CSV = Path(__file__).resolve().parent.parent / "public" / "insurance.csv"
with open(CSV, newline="") as f:
    ROWS = list(csv.DictReader(f))
COLUMNS = list(ROWS[0].keys())
NUMERIC = {"age", "bmi", "children", "charges"}
PREVIEW = [{k: (float(v) if k in NUMERIC else v) for k, v in r.items()} for r in ROWS[:20]]
PREVIEW[3]["bmi"] = None
IMG_COLUMNS = ["image", "label", "width", "height"]
IMG_PREVIEW = [{"image": f"{c}/{c}_{i:03d}.jpg", "label": c, "width": 224, "height": 224} for i in range(10) for c in ("cat", "dog")]

STAGES = ["starting", "profile", "detect", "audit", "analysis", "eda", "feature_engineering", "train", "visualize", "report", "package"]
STAGE_SECONDS = 1.2
DEFAULTS = {"r2": False, "delay": 0, "complete": "ok"}
CONFIG = dict(DEFAULTS)
UNREADABLE = "Couldn't parse the file: line 12 has 9 fields, expected 7."

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def slow_reads(request: Request, call_next):
    # CONFIG["delay"] seconds on run reads, to catch loading states on screen.
    if CONFIG.get("delay") and request.method == "GET" and request.url.path.startswith("/runs/"):
        await asyncio.sleep(CONFIG["delay"])
    return await call_next(request)


def err(code, detail):
    return JSONResponse({"detail": detail}, status_code=code)


def seeded():
    now = time.time()
    base = {"filename": "insurance.csv", "source_format": "csv", "queued_at": now - 214}
    return {
        "fresh": {"status": "uploaded", "filename": "insurance.csv", "source_format": "csv"},
        "reading": {"status": "uploading", "filename": "Combined_Flights_2018.csv", "size": 1992289760},
        "reg-done": {**base, "status": "succeeded", "task": "auto", "target": "charges", "stage": "done", "kind": "reg"},
        "clf-done": {**base, "status": "succeeded", "task": "auto", "target": "smoker", "stage": "done", "kind": "clf"},
        "clu-done": {**base, "status": "succeeded", "task": "clustering", "target": None, "stage": "done", "kind": "clu"},
        "fc-done": {**base, "status": "succeeded", "task": "forecasting", "target": "charges", "stage": "done", "kind": "fc"},
        "img-done": {**base, "filename": "pets.zip", "source_format": "images", "status": "succeeded",
                     "task": "image_classification", "target": "label", "stage": "done", "kind": "img"},
        "bare-done": {**base, "status": "succeeded", "task": "auto", "target": "charges", "stage": "done", "kind": "bare"},
        "running": {**base, "status": "running", "task": "auto", "target": "charges", "stage": "train", "frozen": True},
        "queued": {**base, "status": "queued", "task": "auto", "target": "charges", "frozen": True},
        "failed": {**base, "status": "failed", "task": "auto", "target": "charges", "stage": "train",
                   "error": "GPU container exited: CUDA out of memory while fitting XGBRegressor (fold 3)."},
        "cancelled": {**base, "status": "cancelled", "task": "auto", "target": "charges", "stage": "eda", "error": "Cancelled by the user."},
        "deployfail-done": {**base, "status": "succeeded", "task": "auto", "target": "charges", "stage": "done", "kind": "reg"},
        "live-done": {**base, "status": "succeeded", "task": "auto", "target": "charges", "stage": "done", "kind": "reg"},
        "liveclf-done": {**base, "status": "succeeded", "task": "auto", "target": "smoker", "stage": "done", "kind": "clf"},
        "predicterr-done": {**base, "status": "succeeded", "task": "auto", "target": "charges", "stage": "done", "kind": "reg"},
        "noresult-done": {**base, "status": "succeeded", "task": "auto", "target": "charges", "stage": "done", "kind": "none"},
        "schemaerr-done": {**base, "status": "succeeded", "task": "auto", "target": "charges", "stage": "done", "kind": "reg", "noschema": True},
    }


runs: dict[str, dict] = seeded()
deployments: dict[str, dict] = {
    "live-done": {"status": "succeeded", "app_name": "namtheg-inference", "predict_url": "https://ws--namtheg-inference-predict.modal.run/live-done", "elapsed_seconds": 41},
    "liveclf-done": {"status": "succeeded", "app_name": "namtheg-inference", "predict_url": "https://ws--namtheg-inference-predict.modal.run/liveclf-done", "elapsed_seconds": 38},
    "predicterr-done": {"status": "succeeded", "app_name": "namtheg-inference", "predict_url": "https://ws--namtheg-inference-predict.modal.run/predicterr-done", "elapsed_seconds": 40},
}
deploy_attempts: dict[str, int] = {}


def advance(r):
    # A file being read in the background is done at ready_at: read, or failed
    # with read_error.
    if r["status"] == "uploading" and r.get("ready_at") and time.time() >= r["ready_at"]:
        r.pop("ready_at")
        error = r.pop("read_error", None)
        if error:
            r.update(status="failed", error=error)
        else:
            r["status"] = "uploaded"
    if r.get("frozen") or r["status"] not in ("queued", "running") or "started_at" not in r:
        return r
    i = int((time.time() - r["started_at"]) / STAGE_SECONDS)
    if i < 0:
        r["status"] = "queued"
    elif i >= len(STAGES):
        r["status"], r["stage"] = "succeeded", "done"
    else:
        r["status"], r["stage"] = "running", STAGES[i]
    return r


def public(r):
    return {k: v for k, v in r.items() if k not in ("started_at", "frozen", "kind", "noschema", "ready_at", "read_error")}


@app.post("/__mock/config")
async def config(request: Request):
    CONFIG.update(await request.json())
    return CONFIG


@app.post("/__mock/reset")
def reset():
    runs.clear()
    runs.update(seeded())
    deploy_attempts.clear()
    for k in list(deployments):
        if k not in ("live-done", "liveclf-done", "predicterr-done"):
            del deployments[k]
    CONFIG.clear()
    CONFIG.update(DEFAULTS)
    return {"ok": True}


# ---- uploads
def new_run(filename):
    rid = uuid.uuid4().hex[:16]
    images = filename.lower().endswith(".zip")
    runs[rid] = {"status": "uploaded", "filename": filename, "source_format": "images" if images else filename.rsplit(".", 1)[-1]}
    return rid


def upload_response(rid):
    r = runs[rid]
    cols, prev = (IMG_COLUMNS, IMG_PREVIEW) if r["source_format"] == "images" else (COLUMNS, PREVIEW)
    return {"run_id": rid, "filename": r["filename"], "columns": cols, "preview": prev[:5], "n_rows": 1338}


@app.post("/uploads/direct")
async def direct(request: Request):
    body = await request.json()
    if not CONFIG["r2"]:
        return err(409, "Direct uploads need R2 storage, which is not configured on this backend.")
    if body.get("size", 0) > 2 * 1024 ** 3:
        return err(413, "File too large. Maximum size is 2 GB.")
    rid = new_run(body["filename"])
    runs[rid].update(status="uploading", size=body.get("size"))
    return {"run_id": rid, "method": "PUT", "upload_url": f"{request.base_url}__r2/{rid}", "expires_in": 3600}


@app.put("/__r2/{rid}")
async def r2_put(rid: str, request: Request):
    runs[rid]["size"] = len(await request.body())
    return Response(status_code=200)


@app.post("/uploads/{rid}/complete")
async def complete(rid: str):
    """Reads a stored upload. CONFIG["complete"] picks how. As the backend does:
    ok       answers 202 at once and reads in the background for read_seconds
             (1); a call while it reads starts no other read
    fail     answers 202 at once, and the read fails with a reason
    As the backend did before reads moved to the background, which the frontend
    still handles:
    dropped  reads inside the request and answers a plain 500 after
             hold_seconds (27), the way a proxy that stopped waiting does; the
             run is ready 5 seconds later
    reject   answers 400 with a reason, and the run fails
    And either way:
    crash    a plain 500 at once, with the run left mid-read
    """
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    if r["status"] != "uploading":
        return err(409, "This upload was already completed.")
    mode = CONFIG["complete"]
    if mode in ("ok", "fail"):
        if not r.get("ready_at"):
            r["ready_at"] = time.time() + CONFIG.get("read_seconds", 1)
            if mode == "fail":
                r["read_error"] = UNREADABLE
        return JSONResponse({"run_id": rid, "status": "uploading"}, status_code=202)
    if mode == "dropped":
        hold = CONFIG.get("hold_seconds", 27)
        r["ready_at"] = time.time() + hold + 5
        await asyncio.sleep(hold)
        return PlainTextResponse("Internal Server Error", status_code=500)
    if mode == "reject":
        r.update(status="failed", error=UNREADABLE)
        return err(400, r["error"])
    if mode == "crash":
        return PlainTextResponse("Internal Server Error", status_code=500)
    return err(500, f"Unknown mock complete mode {mode!r}.")


@app.post("/upload")
async def upload(request: Request):
    raw = await request.body()
    m = re.search(rb'filename="([^"]+)"', raw)
    name = m.group(1).decode() if m else "dataset.csv"
    return upload_response(new_run(name))


# ---- runs
def get_run(rid):
    r = runs.get(rid)
    return advance(r) if r else None


@app.get("/runs/{rid}/preview")
def preview(rid: str):
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    cols, prev = (IMG_COLUMNS, IMG_PREVIEW) if r.get("source_format") == "images" else (COLUMNS, PREVIEW)
    return {"run_id": rid, "filename": r.get("filename"), "source_format": r.get("source_format"), "n_columns": len(cols), "columns": cols, "preview": prev}


@app.get("/runs/{rid}/status")
def status(rid: str):
    r = get_run(rid)
    return {"run_id": rid, **public(r)} if r else err(404, "run_id not found")


@app.post("/runs/{rid}/start")
async def start(rid: str, request: Request):
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    if r["status"] in ("queued", "running"):
        return err(409, "This run is already in progress.")
    body = await request.json()
    task = body.get("task", "auto")
    target = body.get("target")
    if task == "forecasting" and not body.get("date_column"):
        return err(400, "Forecasting needs a date column.")
    if task in ("auto", "classification", "regression", "forecasting") and target not in (COLUMNS if r.get("source_format") != "images" else IMG_COLUMNS):
        return err(400, f"Target column {target!r} is not in the dataset.")
    kind = {"clustering": "clu", "forecasting": "fc", "image_classification": "img"}.get(task) or ("reg" if target in NUMERIC else "clf")
    r.update(status="queued", task=task, target=target if task != "clustering" else None, queued_at=time.time(),
             started_at=time.time() + 1.5, kind=kind)
    r.pop("stage", None)
    r.pop("error", None)
    return {"run_id": rid, "status": "queued", "task": task, "target": r["target"]}


@app.post("/runs/{rid}/cancel")
def cancel(rid: str):
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    if r["status"] not in ("queued", "running"):
        return err(409, f"The run is {r['status']}; nothing to cancel.")
    r.update(status="cancelled", error="Cancelled by the user.", frozen=True)
    return {"run_id": rid, "status": "cancelled"}


@app.get("/runs/{rid}/diagnostics")
def diagnostics(rid: str):
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    q = r.get("queued_at")
    return {"status": r["status"], "stage": r.get("stage"), "task": r.get("task"), "gpu": "NVIDIA H200 (requested)",
            "gpu_calls": 2 if r["status"] != "queued" else 0, "elapsed_seconds": round(time.time() - q) if q else None}


def trials_for(metric, lower):
    base = 1.182 if lower else 0.8612
    out = [{"trial": 0, "parameters": "{}", "score": base, "result": "Baseline"}]
    for i in range(1, 9):
        step = (i % 4) * (0.021 if lower else 0.0041)
        score = round(base - step if lower else base + step, 4)
        out.append({
            "trial": i,
            "parameters": json.dumps({"n_estimators": 200 + 50 * i, "max_depth": 3 + i % 4, "learning_rate": round(0.1 / i, 4)}),
            "score": None if i == 6 else score,
            "result": "Error: trial diverged." if i == 6 else (
                "New champion. Reason: deeper trees captured the smoker x bmi interaction without overfitting." if i == 3
                else "Kept previous champion. Reason: gain within cross-validation noise."),
        })
    return out


def result_for(rid, r):
    kind = r["kind"]
    common = {"run_id": rid, "status": "succeeded", "task": r["task"], "target": r.get("target")}
    just = ("{m} won on cross-validated {s}, ahead of the other candidates. Smoking status dominates the prediction, "
            "followed by BMI and age. The train-test gap is small, so the model generalises rather than memorising. "
            "نَمذِج tuned it with 8 Optuna trials; trial 3 improved the score with deeper trees.")
    feats = [{"feature": f, "importance": v} for f, v in [("smoker_yes", 0.6152), ("bmi", 0.1931), ("age", 0.1287), ("children", 0.0233),
                                                          ("region_southeast", 0.0142), ("sex_male", 0.0098), ("region_northwest", 0.0081)]]
    if kind == "reg":
        models = [("GradientBoostingRegressor", 0.8735, 0.0121), ("RandomForestRegressor", 0.8581, 0.0144), ("XGBRegressor", 0.8566, 0.0152),
                  ("LGBMRegressor", 0.8519, 0.0133), ("ExtraTreesRegressor", 0.8307, 0.0198), ("Ridge", 0.7489, 0.0211), ("LinearRegression", 0.7481, 0.0215)]
        return {**common, "problem_type": "regression", "accuracy_score": 0.8735, "score_metric": "r2", "higher_is_better": True,
                "model_name": models[0][0], "justification": just.format(m=models[0][0], s="R² (0.874 ± 0.012)"),
                "downloads": ["cleaned_csv", "model"],
                "extra": {"train_score": 0.9046, "test_score": 0.8691, "overfit_gap": 0.031, "hardware": {"gpu": "NVIDIA H200"},
                          "all_models": [{"name": n, "cv_mean": m, "cv_std": s} for n, m, s in models], "top_features": feats,
                          "tuning_trials": trials_for("r2", False)}}
    if kind == "clf":
        models = [("LGBMClassifier", 0.9612, 0.0081), ("XGBClassifier", 0.9577, 0.0093), ("RandomForestClassifier", 0.9533, 0.0102), ("LogisticRegression", 0.9401, 0.0110)]
        return {**common, "problem_type": "classification", "accuracy_score": 0.9612, "score_metric": "accuracy", "higher_is_better": True,
                "model_name": models[0][0], "justification": just.format(m=models[0][0], s="accuracy (0.961)"), "downloads": ["model"],
                "extra": {"train_score": 0.9987, "test_score": 0.8211, "overfit_gap": 0.1776, "n_classes": 2,
                          "all_models": [{"name": n, "cv_mean": m, "cv_std": s} for n, m, s in models], "top_features": feats[:5]}}
    if kind == "clu":
        models = [("KMeans (k=4)", 0.4121, 0.0), ("HDBSCAN", 0.3877, 0.0), ("Spectral (k=4)", 0.3512, 0.0), ("DBSCAN", 0.2140, 0.0)]
        return {**common, "problem_type": "clustering", "accuracy_score": 0.4121, "score_metric": "silhouette", "higher_is_better": True,
                "model_name": models[0][0], "justification": "KMeans with four clusters separated the rows most cleanly (silhouette 0.41).",
                "downloads": ["cleaned_csv"], "extra": {"all_models": [{"name": n, "cv_mean": m, "cv_std": s} for n, m, s in models]}}
    if kind == "fc":
        models = [("AutoETS", 0.9731, 0.0412), ("AutoARIMA", 1.0442, 0.0520), ("SeasonalNaive", 1.1820, 0.0611), ("LightGBM (lags)", 1.0993, 0.0588)]
        return {**common, "problem_type": "forecasting", "accuracy_score": 0.9731, "score_metric": "mase", "higher_is_better": False,
                "model_name": models[0][0], "justification": "AutoETS had the lowest backtest MASE, beating the seasonal naive baseline.",
                "downloads": ["forecast", "model"],
                "extra": {"train_score": 0.8812, "test_score": 0.9731, "all_models": [{"name": n, "cv_mean": m, "cv_std": s} for n, m, s in models],
                          "tuning_trials": trials_for("mase", True)}}
    if kind == "img":
        models = [("convnext_tiny", 0.9420, 0.0110), ("resnet50", 0.9310, 0.0132), ("efficientnet_b0", 0.9255, 0.0140)]
        return {**common, "problem_type": "classification", "accuracy_score": 0.942, "score_metric": "accuracy", "higher_is_better": True,
                "model_name": models[0][0], "justification": "convnext_tiny fine-tuned best on the held-out images.", "downloads": ["model"],
                "extra": {"all_models": [{"name": n, "cv_mean": m, "cv_std": s} for n, m, s in models]}}
    if kind == "bare":
        return {**common, "problem_type": "regression", "accuracy_score": 0.7123, "score_metric": "r2", "model_name": "Ridge"}
    return None


@app.get("/runs/{rid}/result")
def result(rid: str):
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    res = result_for(rid, r) if r["status"] == "succeeded" else None
    return res if res else err(409, "Run has not produced a result yet.")


@app.get("/runs/{rid}/plot")
def plot(rid: str):
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    if r.get("kind") in ("img", "bare") or r["status"] != "succeeded":
        return err(404, "Plot not yet generated.")
    random.seed(4)
    pts = "".join(
        f'<circle cx="{x + 60:.1f}" cy="{620 - (x + random.gauss(0, 38)):.1f}" r="5" fill="#5b8def" fill-opacity="0.6"/>'
        for x in (random.uniform(40, 560) for _ in range(140))
    )
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="680" viewBox="0 0 680 680"><rect width="680" height="680" fill="#fff"/>'
           f'<line x1="60" y1="620" x2="640" y2="40" stroke="#c0392b" stroke-width="2" stroke-dasharray="8 6"/>{pts}</svg>')
    return Response(svg, media_type="image/svg+xml")


@app.get("/runs/{rid}/download/{kind}")
def download(rid: str, kind: str):
    return Response(f"mock {kind} for {rid}\n", media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{rid}_{kind}.csv"'})


@app.get("/runs/{rid}/model_schema")
def schema(rid: str):
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    if r["status"] != "succeeded" or r.get("noschema") or r.get("kind") == "none":
        return err(409, "No trained model - finish a successful run first.")
    kind = r["kind"]
    if kind == "img":
        return {"run_id": rid, "task": "image_classification", "model_name": "convnext_tiny", "problem_type": "classification",
                "feature_cols": [], "class_labels": ["cat", "dog"], "sample": {}}
    if kind == "fc":
        return {"run_id": rid, "task": "forecasting", "model_name": "AutoETS", "problem_type": "forecasting", "feature_cols": [], "sample": {}}
    target = r.get("target") or "charges"
    feats = [c for c in COLUMNS if c != target]
    clf = kind == "clf"
    return {"run_id": rid, "task": "auto", "model_name": "LGBMClassifier" if clf else "GradientBoostingRegressor",
            "problem_type": "classification" if clf else "regression", "feature_cols": feats,
            "class_labels": ["no", "yes"] if clf else None, "sample": {c: PREVIEW[0][c] for c in feats}}


@app.get("/runs/{rid}/deployment")
def deployment(rid: str):
    if not get_run(rid):
        return err(404, "run_id not found")
    d = deployments.get(rid)
    if not d:
        return {"run_id": rid, "status": "not_deployed"}
    if d["status"] == "deploying" and time.time() - d["t"] > 3:
        if rid == "deployfail-done" and deploy_attempts.get(rid, 0) == 1:
            d.update(status="failed", error="Container crashed on start: ModuleNotFoundError: No module named 'lightgbm'.")
        else:
            d.update(status="succeeded", predict_url=f"https://ws--namtheg-inference-predict.modal.run/{rid}", elapsed_seconds=41)
    return {"run_id": rid, **{k: v for k, v in d.items() if k != "t"}}


@app.post("/runs/{rid}/deploy")
def deploy(rid: str):
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    if r["status"] != "succeeded":
        return err(409, "No trained model to deploy - finish a successful run first.")
    deploy_attempts[rid] = deploy_attempts.get(rid, 0) + 1
    deployments[rid] = {"status": "deploying", "app_name": "namtheg-inference", "t": time.time()}
    return {"run_id": rid, "status": "deploying"}


@app.post("/runs/{rid}/predict")
async def predict(rid: str, request: Request):
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    d = deployments.get(rid)
    if not d or d.get("status") != "succeeded":
        return err(409, "Model is not deployed yet. Click 'Deploy to Modal' first.")
    if rid == "predicterr-done":
        return err(502, "Could not reach Modal endpoint: timed out")
    try:
        body = json.loads(await request.body())
    except json.JSONDecodeError:
        return err(422, [{"msg": "Body is not valid JSON"}])
    rows = body.get("rows") or ([body["features"]] if isinstance(body.get("features"), dict) else None)
    if not rows:
        return err(422, [{"loc": ["body"], "msg": "Send either features (one row) or rows (a batch)."}])
    if r.get("kind") == "clf":
        probs = [[0.12, 0.88] if i % 2 == 0 else [0.71, 0.29] for i in range(len(rows))]
        return {"predictions": [1 if p[1] > 0.5 else 0 for p in probs], "predicted_labels": ["yes" if p[1] > 0.5 else "no" for p in probs],
                "probabilities": probs, "class_labels": ["no", "yes"], "model": "LGBMClassifier"}
    return {"predictions": [round(16884.9241 + 731.2 * i, 4) for i in range(len(rows))], "model": "GradientBoostingRegressor"}
