"""Mock of the Namtheg backend (Backend/app/main.py) for the e2e scripts.

Shapes and status codes follow the real API: errors are {"detail": ...}, a
result that isn't written yet is 409, unknown runs are 404, and diagnostics'
elapsed_seconds counts from queued_at forever. Seeded run ids cover the
scenarios (see seeded()); POST /__mock/config switches R2 direct upload, a
response delay, the way /uploads/{id}/complete behaves (see complete()) and
whether accounts are offered ("auth"); POST /__mock/reset puts everything back.
Accounts send no email: GET /__mock/last_code?email=... reads the code that
would have gone out, and POST /__mock/account makes an account with runs.

Run from this folder with any Python that has fastapi and uvicorn, for example
the backend's virtualenv:

    ../../Backend/.venv/Scripts/python -m uvicorn mock_backend:app --port 8001
"""
import asyncio
import csv
import json
import random
import re
import secrets
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

# What the analyst agent understood (Backend/app/agent/analyst.py), on regression results.
ANALYST_EXTRA = {
    "analysis": {
        "status": "completed", "code_runs": 4, "seconds": 41.6, "prediction_time": None,
        "summary": "Smoking status dominates charges; BMI and age add steady increases.",
        "data_dictionary": {
            "row_meaning": "One insured person.",
            "target_meaning": "Yearly medical charges, in US dollars, from 1,121.87 to 63,770.43.",
            "columns": [
                {"column": "smoker", "meaning": "Whether the person smokes: yes or no.", "basis": "both", "verified": True},
                {"column": "bmi", "meaning": "Body mass index, from 15.96 to 53.13.", "basis": "data", "verified": True},
                {"column": "children", "meaning": "Children covered by the plan, 0 to 5.", "basis": "both", "verified": True},
                {"column": "region", "meaning": "US region of residence, four values.", "basis": "name", "verified": True},
            ],
            "relationships": [{"finding": "Smokers are charged about 3.8 times more.",
                               "evidence": "mean 32,050.23 vs 8,434.27", "verified": True}],
            "verified": True,
        },
        "findings": [{"finding": "Charges are right-skewed.", "evidence": "skewness 1.52", "severity": "info", "verified": True},
                     {"finding": "One exact duplicate row.", "evidence": "1 duplicate of 1,338", "severity": "warning",
                      "verified": True}],
        "open_questions": ["Are the charges yearly or lifetime totals?"],
        "drop_columns": [],
    },
    "sample": {"rows_total": 1338, "rows_used": 1000, "sampled": True, "stratified": False},
    "cleaning_applied": ["Treated children as categories, not quantities (6 codes)."],
}

STAGES = ["starting", "profile", "detect", "audit", "analysis", "eda", "feature_engineering", "train", "visualize", "report", "package"]
STAGE_SECONDS = 1.2
DEFAULTS = {"r2": False, "delay": 0, "complete": "ok", "auth": True}
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
    accounts.clear()
    codes.clear()
    sessions.clear()
    return {"ok": True}


# ---- uploads
def new_run(filename):
    rid = uuid.uuid4().hex[:16]
    runs[rid] = {"status": "uploaded", "filename": filename, "source_format": filename.rsplit(".", 1)[-1]}
    return rid


def upload_response(rid):
    r = runs[rid]
    return {"run_id": rid, "filename": r["filename"], "columns": COLUMNS, "preview": PREVIEW[:5], "n_rows": 1338}


@app.post("/uploads/direct")
async def direct(request: Request):
    body = await request.json()
    if not CONFIG["r2"]:
        return err(409, "Direct uploads need R2 storage, which is not configured on this backend.")
    if body.get("size", 0) > 2 * 1024 ** 3:
        return err(413, "File too large. Maximum size is 2 GB.")
    rid = new_run(body["filename"])
    runs[rid].update(status="uploading", size=body.get("size"))
    link(request, rid)
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
    rid = new_run(name)
    link(request, rid)
    return upload_response(rid)


# ---- runs
def get_run(rid):
    r = runs.get(rid)
    return advance(r) if r else None


@app.get("/runs/{rid}/preview")
def preview(rid: str):
    r = get_run(rid)
    if not r:
        return err(404, "run_id not found")
    return {"run_id": rid, "filename": r.get("filename"), "source_format": r.get("source_format"), "n_columns": len(COLUMNS), "columns": COLUMNS, "preview": PREVIEW}


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
    if not target:
        return err(400, "Choose the target column.")
    if target not in COLUMNS:
        return err(400, f"Target column {target!r} is not in the dataset.")
    prediction_time = " ".join(str(body.get("prediction_time") or "").split()) or None
    if prediction_time and len(prediction_time) > 300:
        return err(422, "prediction_time: at most 300 characters.")
    new_column = " ".join(str(body.get("new_column") or "").split()) or None
    if new_column and len(new_column) > 500:
        return err(422, "new_column: at most 500 characters.")
    kind = "reg" if target in NUMERIC else "clf"
    r.update(status="queued", task=task, target=target, prediction_time=prediction_time, new_column=new_column,
             queued_at=time.time(), started_at=time.time() + 1.5, kind=kind)
    r.pop("stage", None)
    r.pop("error", None)
    link(request, rid)
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
    return {"status": r["status"], "stage": r.get("stage"), "task": r.get("task"), "gpu": "NVIDIA H200",
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
                          "tuning_trials": trials_for("r2", False), **ANALYST_EXTRA,
                          # As the backend reports a column the user asked for: added in cleaning, named in the analysis.
                          "cleaning_applied": ANALYST_EXTRA["cleaning_applied"] + (
                              [f"Added the column you asked for: {r['new_column']}."] if r.get("new_column") else []),
                          "analysis": {**ANALYST_EXTRA["analysis"], "prediction_time": r.get("prediction_time"),
                                       "new_column": r.get("new_column")}}}
    if kind == "clf":
        models = [("LGBMClassifier", 0.9612, 0.0081), ("XGBClassifier", 0.9577, 0.0093), ("RandomForestClassifier", 0.9533, 0.0102), ("LogisticRegression", 0.9401, 0.0110)]
        return {**common, "problem_type": "classification", "accuracy_score": 0.9612, "score_metric": "accuracy", "higher_is_better": True,
                "model_name": models[0][0], "justification": just.format(m=models[0][0], s="accuracy (0.961)"), "downloads": ["model"],
                "extra": {"train_score": 0.9987, "test_score": 0.8211, "overfit_gap": 0.1776, "n_classes": 2,
                          "all_models": [{"name": n, "cv_mean": m, "cv_std": s} for n, m, s in models], "top_features": feats[:5]}}
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
    if r.get("kind") == "bare" or r["status"] != "succeeded":
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
    target = r.get("target") or "charges"
    feats = [c for c in COLUMNS if c != target]
    clf = kind == "clf"
    return {"run_id": rid, "task": "auto", "model_name": "LGBMClassifier" if clf else "GradientBoostingRegressor",
            "problem_type": "classification" if clf else "regression", "feature_cols": feats,
            "class_labels": ["no", "yes"] if clf else None, "sample": {c: PREVIEW[0][c] for c in feats},
            # Each category column's values in training, which the predict form lists.
            "feature_values": {c: sorted({row[c] for row in ROWS}) for c in feats if c not in NUMERIC}}


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


# ---- accounts (Backend/app/accounts.py): a code by email at every sign-in, a session cookie
COOKIE = "namtheg_session"
EMAIL = re.compile(r"[^@\s]{1,64}@[^@\s]+\.[^@\s.]{2,}")
OFF = "Accounts are not enabled on this server."
accounts: dict[str, dict] = {}  # email -> {"name", "email", "runs": [[run_id, added_at], ...]}
codes: dict[str, dict] = {}  # email -> the pending code and its tries
sessions: dict[str, str] = {}  # cookie value -> email


def signed_in(request):
    return accounts.get(sessions.get(request.cookies.get(COOKIE, ""), ""))


def link(request, rid):
    """As the backend does on upload and start: the run goes to the top of the signed-in user's runs."""
    user = signed_in(request) if CONFIG["auth"] else None
    if user:
        user["runs"] = [r for r in user["runs"] if r[0] != rid] + [[rid, time.time()]]


def account_json(user):
    return {"name": user["name"], "email": user["email"]}


@app.post("/__mock/account")
async def make_account(request: Request):
    body = await request.json()
    accounts[body["email"]] = {"name": body["name"], "email": body["email"],
                               "runs": [[rid, time.time() - 3600 * (len(body["runs"]) - i)] for i, rid in enumerate(body.get("runs", []))]}
    return {"ok": True}


@app.get("/__mock/last_code")
def last_code(email: str):
    pending = codes.get(email.strip().lower())
    return {"code": pending["code"] if pending else None}


@app.get("/auth/me")
def auth_me(request: Request):
    user = signed_in(request) if CONFIG["auth"] else None
    return {"enabled": CONFIG["auth"], "user": account_json(user) if user else None}


@app.post("/auth/code")
async def auth_code(request: Request):
    if not CONFIG["auth"]:
        return err(404, OFF)
    body = await request.json()
    email = str(body.get("email") or "").strip().lower()
    name = " ".join(str(body.get("name") or "").split())
    if len(email) > 254 or not EMAIL.fullmatch(email):
        return err(400, "Enter a valid email address.")
    if email not in accounts and not name:
        return {"sent": True}  # nothing is sent, and the answer doesn't say so
    pending, wait = codes.get(email), CONFIG.get("resend_seconds", 60)
    if pending and time.time() - pending["sent_at"] < wait:
        return err(429, f"A code was just sent. You can ask for another in {int(wait - (time.time() - pending['sent_at'])) + 1} seconds.")
    codes[email] = {"code": f"{secrets.randbelow(10 ** 6):06d}", "tries": 0, "sent_at": time.time(),
                    "signup_name": None if email in accounts else name[:80]}
    return {"sent": True}


@app.post("/auth/verify")
async def auth_verify(request: Request):
    if not CONFIG["auth"]:
        return err(404, OFF)
    body = await request.json()
    email = str(body.get("email") or "").strip().lower()
    pending = codes.get(email)
    if not pending or pending["tries"] >= 5:
        return err(400, "That code has expired. Ask for a new one.")
    if re.sub(r"\D", "", str(body.get("code") or "")) != pending["code"]:
        pending["tries"] += 1
        left = 5 - pending["tries"]
        return err(400, f"That code isn't right. {left} {'try' if left == 1 else 'tries'} left." if left
                   else "That code isn't right, and that was the last try. Ask for a new one.")
    del codes[email]
    user = accounts.setdefault(email, {"name": pending["signup_name"] or email.split("@")[0], "email": email, "runs": []})
    token = uuid.uuid4().hex
    sessions[token] = email
    response = JSONResponse({"user": account_json(user)})
    response.set_cookie(COOKIE, token, max_age=30 * 24 * 3600, httponly=True, samesite="lax", path="/")
    return response


@app.post("/auth/logout")
def auth_logout(request: Request):
    sessions.pop(request.cookies.get(COOKIE, ""), None)
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE, path="/")
    return response


@app.get("/auth/runs")
def auth_runs(request: Request):
    if not CONFIG["auth"]:
        return err(404, OFF)
    user = signed_in(request)
    if not user:
        return err(401, "Sign in to see your runs.")
    out = []
    for rid, added_at in reversed(user["runs"]):
        r = get_run(rid)
        if not r:
            continue
        res = result_for(rid, r) if r["status"] == "succeeded" else None
        out.append({"run_id": rid, "added_at": added_at, "filename": r.get("filename"), "status": r["status"],
                    "task": r.get("task"), "target": r.get("target"), "model_name": res["model_name"] if res else None,
                    "score": res["accuracy_score"] if res else None, "score_metric": res["score_metric"] if res else None})
    return {"runs": out}


@app.post("/auth/runs/{rid}")
def auth_save_run(rid: str, request: Request):
    if not CONFIG["auth"]:
        return err(404, OFF)
    if not signed_in(request):
        return err(401, "Sign in to save runs.")
    if not get_run(rid):
        return err(404, "run_id not found")
    link(request, rid)
    return {"ok": True}
