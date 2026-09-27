"""Modal deployment of the Namtheg backend: the HTTP API plus a job function.

Deploy with:
    modal deploy app/deploy/backend_app.py

Or test locally with live reload:
    modal serve app/deploy/backend_app.py

- `fastapi_app` serves the HTTP API. Every request is short: starting a run
  only spawns a job and returns.
- `run_job` executes one run end to end in its own container, detached from
  any HTTP request, so runs are bounded by JOB_TIMEOUT_SECONDS rather than a
  request timeout. It waits on the GPU training service (app/training/gpu_app.py).
- Both mount the same Volume at /storage; the API reloads it on reads so it
  sees the job's progress.
"""
from pathlib import Path

import modal

from app.jobs import JOB_APP_NAME, JOB_FUNCTION_NAME, JOB_TIMEOUT_SECONDS

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BACKEND_DIR / ".env"
REQ_PATH = BACKEND_DIR / "requirements.txt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REQ_PATH))
    .add_local_python_source("app")
)

# Persistent storage volume for datasets, run outputs, plots, and models
storage_volume = modal.Volume.from_name("modelforge-storage", create_if_missing=True)

# Unconditionally define Secret so local and remote container definitions match exactly
secret = modal.Secret.from_dotenv(path=str(ENV_PATH))

app = modal.App(JOB_APP_NAME, image=image)


@app.function(
    image=image,
    volumes={"/storage": storage_volume},
    secrets=[secret],
    cpu=2.0,
    memory=2048,
    # Requests are short now; the long ones are uploads of large files.
    timeout=900,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    import os
    os.environ["STORAGE_DIR"] = "/storage"
    from app.main import app as _fastapi_app
    return _fastapi_app


@app.function(
    name=JOB_FUNCTION_NAME,
    image=image,
    volumes={"/storage": storage_volume},
    secrets=[secret],
    cpu=2.0,
    memory=4096,
    timeout=JOB_TIMEOUT_SECONDS,
    retries=0,
)
def run_job(run_id: str, spec: dict) -> None:
    import os
    os.environ["STORAGE_DIR"] = "/storage"
    from app import storage
    from app.agent.orchestrator import run_agent

    # Recorded here too: the API's own write of the call id can race with this job's first status write.
    storage.update_status(run_id, executor="modal", call_id=modal.current_function_call_id())
    run_agent(run_id, spec)
