"""Pretrained text embeddings for free-text columns (runs on the GPU).

Each free-text column is embedded with a multilingual sentence encoder
(Arabic included), then projected onto its top principal components so a few
dense features replace one unusable high-cardinality category. The encoder
is pretrained and never sees the target, so embedding every row up front
does not leak labels.
"""
import numpy as np

from app.training.runtime import encode_text

ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Apache-2.0, 50+ languages
DIMS = 32
PCA_FIT_ROWS = 50_000


def _svd(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Singular values and right singular vectors, on the GPU when there is one."""
    import torch

    if torch.cuda.is_available():
        _, s, vt = torch.linalg.svd(torch.as_tensor(centered, device="cuda"), full_matrices=False)
        return s.cpu().numpy(), vt.cpu().numpy()
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    return s, vt


def embed_columns(columns: dict[str, list], encoder: str = ENCODER, dims: int = DIMS) -> dict:
    """-> {"encoder", "columns": {name: {"mean", "components", "explained_variance"}},
           "features": {name: float32 array bytes, shape (n, k)}}"""
    rng = np.random.default_rng(0)
    projections, features = {}, {}
    for name, values in columns.items():
        emb = encode_text(values, encoder)
        fit_rows = emb if len(emb) <= PCA_FIT_ROWS else emb[rng.choice(len(emb), PCA_FIT_ROWS, replace=False)]
        mean = fit_rows.mean(axis=0)
        k = int(min(dims, fit_rows.shape[1], max(1, len(fit_rows) - 1)))
        s, vt = _svd(fit_rows - mean)
        components = vt[:k]
        variance = (s ** 2) / max((s ** 2).sum(), 1e-12)
        projections[name] = {"mean": mean.tolist(), "components": components.tolist(),
                             "explained_variance": round(float(variance[:k].sum()), 4)}
        features[name] = ((emb - mean) @ components.T).astype(np.float32).tobytes()
    return {"encoder": encoder, "columns": projections, "features": features}
