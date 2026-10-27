"""GPU training service: every model the platform trains is fit here, on
NVIDIA H200s. The backend has no CPU or local training path.

Deploy (from Backend/), and again whenever app/training/ changes:

    modal deploy app/training/gpu_app.py

One image: cuML (runs scikit-learn's SVM/KNN/linear models on the GPU via
cuml.accel), XGBoost, CatBoost, Optuna. Classic machine learning only:
classification and regression.

The backend fans work out in parallel (one model group per call);
MAX_PARALLEL_GPUS caps concurrent H200s per function to bound cost.
"""
import modal

GPU = "H200"
APP_NAME = "namtheg-train-gpu"
PYTHON_VERSION = "3.12"
MAX_PARALLEL_GPUS = 3
CLASSIC_TIMEOUT = 2 * 3600

# Pinned together. The inference image (app/deploy/inference_app.py) must use
# the same versions of every package it shares with these (tests/test_training.py
# enforces it), because scikit-learn objects pickled here are loaded there.
SHARED_PIN = {
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "scikit-learn": "1.8.0",
    "pandas": "2.3.3",
    "joblib": "1.5.3",
    "pyarrow": "21.0.0",
    "xgboost": "3.4.1",
}
CLASSIC_PIN = {
    **SHARED_PIN,
    "catboost": "1.2.10",
    "optuna": "5.0.0",
    "cuml-cu13": "26.6.0",
}


def _pins(pins: dict) -> list[str]:
    return [f"{name}=={version}" for name, version in pins.items()]


classic_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*_pins(CLASSIC_PIN))
    .add_local_python_source("app")
)
app = modal.App(APP_NAME)


def _require_gpu() -> dict:
    """Refuse to run unless the libraries this image trains with can see a GPU."""
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


@app.function(image=classic_image, gpu=GPU, timeout=CLASSIC_TIMEOUT, max_containers=MAX_PARALLEL_GPUS, retries=0)
def train_classic(data: bytes, spec: dict) -> dict:
    """One group of tabular candidates."""
    # Must precede every scikit-learn import so its estimators are GPU-backed.
    import cuml.accel

    cuml.accel.install()
    hardware = _require_gpu()
    from app.training import tabular

    result = tabular.run(data, spec, device="cuda")
    result["hardware"] = hardware
    return result
