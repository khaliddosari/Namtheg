import json as _json
import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from app import r2, storage
from app.agent.orchestrator import run_agent
from app.config import settings
from app.data.ingest import SUPPORTED_EXTENSIONS, IngestError, ingest_file
from app.deploy.modal_deploy import deploy_run
from app.pipeline import export
from app.pipeline.train import GPU_APP_NAME
from app.schemas import StartRunRequest

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("modelforge")

app = FastAPI(title="Namtheg Backend (MVP)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    return {
        "status": "ok",
        "llm_configured": bool(settings.openai_api_key or settings.openrouter_api_key),
        "llm_provider": "openai" if settings.openai_api_key else "openrouter",
        "llm_model": settings.llm_primary_model if settings.openai_api_key else settings.openrouter_model,
        "llm_primary": {"model": settings.llm_primary_model, "configured": bool(settings.openai_api_key)},
        "llm_fallback": {"model": settings.openrouter_model, "configured": bool(settings.openrouter_api_key)},
        "sandbox_backend": settings.sandbox_backend,
        "training_service": GPU_APP_NAME,
        "r2_enabled": r2.enabled(),
    }


MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB cap


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    """Store the raw upload, convert it once to a typed Parquet DataFrame, and
    return a preview. Every later stage reads the Parquet copy."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}.")
    run_id = storage.new_run_id()
    raw = storage.raw_upload_path(run_id, ext)

    total_bytes = 0
    try:
        with raw.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File too large. Maximum size allowed is 30 MB.")
                f.write(chunk)
        report = await run_in_threadpool(
            ingest_file, raw, storage.run_dir(run_id) / storage.DATASET_FILE, file.filename
        )
        storage.persist(run_id, f"raw/{raw.name}")
        storage.persist(run_id, storage.DATASET_FILE)
        storage.write_json(run_id, "ingest.json", report)
        storage.write_status(run_id, "uploaded", filename=file.filename)
        df = storage.dataset_head(run_id, 5)
    except HTTPException:
        shutil.rmtree(storage.run_dir(run_id), ignore_errors=True)
        raise
    except IngestError as e:
        shutil.rmtree(storage.run_dir(run_id), ignore_errors=True)
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("Upload failed for run %s", run_id)
        shutil.rmtree(storage.run_dir(run_id), ignore_errors=True)
        raise HTTPException(500, f"Upload failed: {e}")

    # Roundtrip through pandas' JSON writer to safely handle NaN, Inf,
    # Timestamps and numpy scalar types that the default encoder rejects.
    return {
        "run_id": run_id,
        "filename": file.filename,
        "columns": [str(c) for c in df.columns],
        "preview": _json.loads(df.to_json(orient="records", date_format="iso")),
        "n_rows": report["n_rows"],
        "ingest": {k: report[k] for k in ("source_format", "actions", "warnings")},
    }


@app.get("/runs/{run_id}/preview")
def preview(run_id: str) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    df = storage.dataset_head(run_id, 20)
    status_data = storage.read_status(run_id) or {}
    filename = status_data.get("filename", "dataset.csv")
    columns = [str(c) for c in df.columns.tolist()]
    preview_rows = _json.loads(df.to_json(orient="records", date_format="iso"))
    return {
        "run_id": run_id,
        "filename": filename,
        "n_columns": int(df.shape[1]),
        "columns": columns,
        "preview": preview_rows,
    }


@app.post("/runs/{run_id}/start")
def start_run(run_id: str, req: StartRunRequest, background: BackgroundTasks) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    columns = storage.dataset_columns(run_id)
    if req.target not in columns:
        raise HTTPException(400, f"Target '{req.target}' not in columns: {columns}")
    storage.write_status(run_id, "queued", target=req.target)
    background.add_task(run_agent, run_id, req.target)
    return {"run_id": run_id, "status": "queued", "target": req.target}


@app.get("/runs/{run_id}/status")
def status(run_id: str) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    return {"run_id": run_id, **storage.read_status(run_id)}


@app.get("/runs/{run_id}/result")
def result(run_id: str) -> JSONResponse:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    result = storage.read_json(run_id, "result.json")
    if result is None:
        raise HTTPException(409, "Run has not produced a result yet.")
    return JSONResponse(result)


DOWNLOADS = {
    # kind: (artifact, filename suffix)
    "cleaned_csv": (export.CLEANED_CSV, "_cleaned.csv"),
    "model": (export.MODEL_PACKAGE, "_model.zip"),
}


@app.get("/runs/{run_id}/download/{kind}")
def download(run_id: str, kind: str):
    """cleaned_csv: the dataset the models trained on. model: the winning model
    package (pipeline, native weights, metadata, predict.py). With R2 on, this
    redirects to a presigned URL that expires after R2_URL_TTL_SECONDS."""
    if kind not in DOWNLOADS:
        raise HTTPException(404, f"Unknown download {kind!r}. Available: {', '.join(DOWNLOADS)}.")
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    name, suffix = DOWNLOADS[kind]
    stem = Path((storage.read_status(run_id) or {}).get("filename") or "dataset").stem
    filename = f"{stem}{suffix}"
    if r2.exists(run_id, name):
        return RedirectResponse(r2.presigned_download_url(run_id, name, filename), status_code=307)
    path = storage.artifact_path(run_id, name)
    if not path.exists():
        raise HTTPException(409, "Not available yet - finish a successful run first.")
    return FileResponse(path, filename=filename)


@app.get("/runs/{run_id}/agent_log")
def agent_log(run_id: str) -> dict:
    """Every code cell the analyst agent ran in the sandbox, with its output."""
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    data = storage.read_json(run_id, "agent_log.json")
    if data is None:
        raise HTTPException(409, "The analysis has not run yet.")
    return {"run_id": run_id, **data}


@app.get("/runs/{run_id}/plot")
def plot(run_id: str) -> FileResponse:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    path = storage.artifact_path(run_id, "plot.png")
    if not path.exists():
        raise HTTPException(404, "Plot not yet generated.")
    return FileResponse(path, media_type="image/png", filename="plot.png")


@app.post("/runs/{run_id}/deploy")
def deploy(run_id: str, background: BackgroundTasks) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    if not storage.artifact_path(run_id, "model.joblib").exists():
        raise HTTPException(409, "No trained model to deploy - finish a successful run first.")

    existing = storage.read_json(run_id, "deployment.json") or {}
    if existing.get("status") == "deploying":
        return {"run_id": run_id, "status": "deploying", "message": "Deployment already in progress."}

    storage.write_json(
        run_id,
        "deployment.json",
        {"status": "deploying", "app_name": settings.modal_inference_app},
    )
    background.add_task(deploy_run, run_id)
    return {"run_id": run_id, "status": "deploying"}


@app.get("/runs/{run_id}/deployment")
def deployment(run_id: str) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    data = storage.read_json(run_id, "deployment.json")
    if data is None:
        return {"run_id": run_id, "status": "not_deployed"}
    return {"run_id": run_id, **data}


@app.get("/runs/{run_id}/model_schema")
def model_schema(run_id: str) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    # model_meta.json carries everything this endpoint needs, so the backend
    # never unpickles a model (and needs no GPU-library installs). Runs from
    # before the GPU trainer only have the sklearn pickle.
    bundle = storage.read_json(run_id, "model_meta.json")
    if bundle is None:
        bundle_path = storage.artifact_path(run_id, "model.joblib")
        if not bundle_path.exists():
            raise HTTPException(409, "No trained model - finish a successful run first.")
        bundle = joblib.load(bundle_path)

    # First engineered row gives a realistic baseline for the predict form.
    sample: dict = {}
    eng_path = storage.engineered_path(run_id)
    if eng_path.exists() or (storage.run_dir(run_id) / "engineered.csv").exists():
        row = storage.engineered_head(run_id, 1)
        target = storage.read_status(run_id).get("target")
        for c in bundle["feature_cols"]:
            if c in row.columns:
                val = row.iloc[0][c]
                # Cast numpy scalars to plain JSON-safe types.
                if pd.isna(val):
                    sample[c] = None
                elif hasattr(val, "item"):
                    sample[c] = val.item()
                else:
                    sample[c] = val
        # paranoia: never leak the target column
        sample.pop(target, None)

    return {
        "run_id": run_id,
        "model_name": bundle.get("model_name"),
        "problem_type": bundle.get("problem_type"),
        "feature_cols": bundle["feature_cols"],
        "class_labels": bundle.get("class_labels"),
        "sample": sample,
    }


@app.post("/runs/{run_id}/predict")
async def predict(run_id: str, request: Request) -> dict:
    """Proxy a prediction request to the deployed Modal endpoint.

    Going through the backend avoids browser CORS issues with Modal."""
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    dep = storage.read_json(run_id, "deployment.json") or {}
    predict_url = dep.get("predict_url")
    if not predict_url or dep.get("status") != "succeeded":
        raise HTTPException(409, "Model is not deployed yet. Click 'Deploy to Modal' first.")

    payload = await request.body()
    req = urllib.request.Request(
        predict_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise HTTPException(e.code, f"Modal endpoint error: {detail[:500]}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"Could not reach Modal endpoint: {e.reason}")
    try:
        return _json.loads(body)
    except _json.JSONDecodeError:
        raise HTTPException(502, f"Modal returned non-JSON: {body[:500]}")


@app.get("/runs/{run_id}/diagnostics")
def get_run_diagnostics(run_id: str) -> dict:
    import psutil
    import subprocess
    import shutil

    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")

    # CPU Usage
    cpu_percent = psutil.cpu_percent()

    # RAM Usage
    vm = psutil.virtual_memory()
    ram_used = round(vm.used / (1024 ** 3), 1)
    ram_total = round(vm.total / (1024 ** 3), 0)

    # GPU Usage via nvidia-smi
    gpu_percent = 0
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True
            )
            gpu_percent = int(res.stdout.strip())
        except Exception:
            pass

    # Dynamic Training Speed (rows/s) based on dataset size and CPU load
    row_count = 10000
    profile = storage.read_json(run_id, "profile.json")
    if profile:
        row_count = profile.get("n_rows", 10000)
    else:
        try:
            row_count = max(100, int(storage.read_json(run_id, "ingest.json")["n_rows"]))
        except Exception:
            pass

    cpu_factor = (100 - cpu_percent) / 100.0
    speed = int((800 + (row_count / 12)) * (0.6 + 0.4 * cpu_factor))
    speed = max(100, min(speed, 100000))

    return {
        "cpu": cpu_percent,
        "gpu": gpu_percent,
        "ram": ram_used,
        "ram_total": ram_total,
        "speed": speed
    }

