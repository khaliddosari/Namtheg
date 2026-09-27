"""GPU training service: every model the platform trains is fit here, on
NVIDIA H200s. The backend has no CPU or local training path.

Deploy (from Backend/), and again whenever app/training/ changes:

    modal deploy app/training/gpu_app.py

Two images:
- classic: cuML (runs scikit-learn's SVM/KNN/linear/clustering on the GPU via
  cuml.accel), XGBoost, CatBoost, Optuna, hdbscan. Tabular and clustering.
- deep: PyTorch, timm, sentence-transformers, Chronos. Forecasting, images,
  and text embeddings. Pretrained weights are baked in at build time.

The backend fans work out in parallel (one model group or candidate per
call); MAX_PARALLEL_GPUS caps concurrent H200s per function to bound cost.
"""
import modal

GPU = "H200"
APP_NAME = "namtheg-train-gpu"
PYTHON_VERSION = "3.12"
MAX_PARALLEL_GPUS = 3
CLASSIC_TIMEOUT = 2 * 3600
DEEP_TIMEOUT = 4 * 3600

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
    "hdbscan": "0.8.44",
    "cuml-cu13": "26.6.0",
}
DEEP_PIN = {
    **SHARED_PIN,
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "timm": "1.0.30",
    "sentence-transformers": "6.1.0",
    "chronos-forecasting": "2.3.2",
}

# Downloaded into the deep image at build time. Must match the model ids in
# app/training/{text,forecasting,vision}.py (tests/test_training.py checks).
PRETRAINED = {
    "text": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "chronos": "amazon/chronos-2",
    "timm": ("convnext_tiny.fb_in22k_ft_in1k", "efficientnet_b0.ra_in1k", "resnet50.a1_in1k"),
}


def _pins(pins: dict) -> list[str]:
    return [f"{name}=={version}" for name, version in pins.items()]


def _download_weights():
    from huggingface_hub import snapshot_download

    snapshot_download(PRETRAINED["text"])
    snapshot_download(PRETRAINED["chronos"])
    import timm

    for name in PRETRAINED["timm"]:
        timm.create_model(name, pretrained=True)


classic_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*_pins(CLASSIC_PIN))
    .add_local_python_source("app")
)
deep_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*_pins(DEEP_PIN))
    .run_function(_download_weights)
    .add_local_python_source("app")
)
app = modal.App(APP_NAME)


def _require_gpu(kind: str) -> dict:
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
    if kind == "classic":
        import xgboost
        from catboost.utils import get_gpu_device_count

        if not xgboost.build_info().get("USE_CUDA"):
            raise RuntimeError("XGBoost in the training image has no CUDA support; refusing to train on CPU.")
        if get_gpu_device_count() < 1:
            raise RuntimeError("CatBoost can't see the GPU; refusing to train on CPU.")
    else:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch can't see the GPU; refusing to train on CPU.")
    return {"gpu": name, "gpu_memory": memory, "driver": driver, "requested": GPU}


@app.function(image=classic_image, gpu=GPU, timeout=CLASSIC_TIMEOUT, max_containers=MAX_PARALLEL_GPUS, retries=0)
def train_classic(data: bytes, spec: dict) -> dict:
    """One group of tabular candidates, or clustering."""
    # Must precede every scikit-learn import so its estimators are GPU-backed.
    import cuml.accel

    cuml.accel.install()
    hardware = _require_gpu("classic")
    if spec["task"] == "clustering":
        from app.training import clustering as engine
    else:
        from app.training import tabular as engine
    result = engine.run(data, spec, device="cuda")
    result["hardware"] = hardware
    return result


@app.function(image=deep_image, gpu=GPU, cpu=8, timeout=DEEP_TIMEOUT, max_containers=MAX_PARALLEL_GPUS, retries=0)
def train_deep(data: bytes, spec: dict) -> dict:
    """One forecasting or image candidate."""
    hardware = _require_gpu("deep")
    if spec["task"] == "forecasting":
        from app.training import forecasting as engine
    else:
        from app.training import vision as engine
    result = engine.run(data, spec, device="cuda")
    result["hardware"] = hardware
    return result


@app.function(image=deep_image, gpu=GPU, timeout=1800, max_containers=MAX_PARALLEL_GPUS, retries=0)
def embed_text(columns: dict, encoder: str) -> dict:
    """Pretrained embeddings for free-text columns."""
    _require_gpu("deep")
    from app.training.text import embed_columns

    return embed_columns(columns, encoder)
