"""Modal ASGI deployment entrypoint for the Namtheg FastAPI backend.

Deploy to Modal with:
    modal deploy app/deploy/backend_app.py

Or test locally with live reload:
    modal serve app/deploy/backend_app.py

This provides:
- Serverless scale-to-zero when idle, fast <2s wake-up
- Persistent storage via Modal Volume (mounted at /storage)
- $30/month free compute credits on Modal
- 2 vCPU and 2GB+ RAM to avoid 512MB OOM crashes on ML pipelines
"""
from pathlib import Path
import modal

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BACKEND_DIR / ".env"
REQ_PATH = BACKEND_DIR / "requirements.txt"

# Define container image with all backend dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REQ_PATH))
    .add_local_python_source("app")
)

# Persistent storage volume for datasets, run outputs, plots, and models
storage_volume = modal.Volume.from_name("modelforge-storage", create_if_missing=True)

# Unconditionally define Secret so local and remote container definitions match exactly
secret = modal.Secret.from_dotenv(path=str(ENV_PATH))

app = modal.App("namtheg-backend", image=image)


@app.function(
    image=image,
    volumes={"/storage": storage_volume},
    secrets=[secret],
    cpu=2.0,
    memory=2048,
    # Runs execute as background tasks inside this function and wait on the
    # GPU trainer (up to its own 30 min timeout), so allow a full hour.
    timeout=3600,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    import os
    os.environ["STORAGE_DIR"] = "/storage"
    from app.main import app as _fastapi_app
    return _fastapi_app
