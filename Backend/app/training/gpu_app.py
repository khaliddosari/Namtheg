"""GPU training service: every model the platform trains is fit here, on an
NVIDIA H200. The backend has no CPU or local training path.

Deploy (from Backend/), and again whenever app/training/ changes:

    modal deploy app/training/gpu_app.py
"""
import modal

GPU = "H200"
APP_NAME = "namtheg-train-gpu"
PYTHON_VERSION = "3.12"

# The pipeline pickled here is unpickled by app/deploy/inference_app.py, so the
# packages both images share must be pinned identically there
# (tests/test_training.py enforces it).
IMAGE_PIN = {
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "scikit-learn": "1.8.0",
    "pandas": "2.3.3",
    "joblib": "1.5.3",
    "xgboost": "3.4.1",
    "catboost": "1.2.10",
    "pyarrow": "21.0.0",
    "optuna": "5.0.0",
}

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*(f"{name}=={version}" for name, version in IMAGE_PIN.items()))
    .add_local_python_source("app")
)
app = modal.App(APP_NAME, image=image)


def _require_gpu() -> dict:
    """Refuse to train unless both libraries can see a CUDA GPU."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except Exception as e:
        raise RuntimeError(f"No NVIDIA GPU visible in the training container ({e}); refusing to train on CPU.")
    name, memory, driver = (s.strip() for s in out.splitlines()[0].split(","))

    import xgboost
    from catboost.utils import get_gpu_device_count

    if not xgboost.build_info().get("USE_CUDA"):
        raise RuntimeError("XGBoost in the training image has no CUDA support; refusing to train on CPU.")
    if get_gpu_device_count() < 1:
        raise RuntimeError("CatBoost can't see the GPU; refusing to train on CPU.")
    return {"gpu": name, "gpu_memory": memory, "driver": driver, "requested": GPU}


@app.function(gpu=GPU, timeout=1800, retries=0)
def train(data: bytes, target: str, problem_type: str, plan: dict) -> dict:
    hardware = _require_gpu()
    from app.training.core import run_training

    result = run_training(data, target, problem_type, plan, device="cuda")
    result["hardware"] = hardware
    return result
