"""Clustering (no target column) on the GPU.

Candidates, all run through cuml.accel on the GPU (`gpu_only` rejects any
CPU fallback):
- K-Means, with k chosen by silhouette over 2..MAX_K.
- HDBSCAN (hdbscan package), density-based, finds k and outliers itself.
- DBSCAN, eps taken from the k-distance distribution.
- Spectral clustering on a nearest-neighbour graph (small data only).
Agglomerative clustering and Gaussian mixtures are deliberately absent: they
have no GPU implementation.

Selection is by silhouette on non-noise points, requiring at least two
clusters and at most MAX_NOISE_SHARE outliers. There is no ground truth, so
the result is a description of structure, not a verified labelling.
"""
import io
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from app.training.common import RANDOM_STATE, gpu_only, library_versions
from app.training.runtime import BUNDLE_FORMAT
from app.training.tabular import build_preprocessor

CANDIDATES = ("K-Means", "HDBSCAN", "DBSCAN", "Spectral")
MAX_K = 12
MAX_DIMS = 50
SILHOUETTE_SAMPLE = 10_000
SPECTRAL_MAX_ROWS = 5_000
REFERENCE_POINTS = 5_000
PROJECTION_POINTS = 5_000
MAX_NOISE_SHARE = 0.5


def silhouette(Z: np.ndarray, labels: np.ndarray, device: str, rng) -> float | None:
    """Silhouette on non-noise points (sampled); None if fewer than 2 clusters."""
    idx = np.flatnonzero(labels >= 0)
    if len(np.unique(labels[idx])) < 2:
        return None
    if len(idx) > SILHOUETTE_SAMPLE:
        idx = rng.choice(idx, SILHOUETTE_SAMPLE, replace=False)
    if device.startswith("cuda"):
        from cuml.metrics.cluster import silhouette_score
    else:
        from sklearn.metrics import silhouette_score
    return float(silhouette_score(Z[idx], labels[idx]))


def _evaluate(name, params, labels, Z, device, rng) -> dict:
    labels = np.asarray(labels).astype(int)
    noise = float((labels < 0).mean())
    n_clusters = int(len(np.unique(labels[labels >= 0])))
    sil = silhouette(Z, labels, device, rng)
    valid = sil is not None and noise <= MAX_NOISE_SHARE
    return {"name": name, "params": params, "silhouette": None if sil is None else round(sil, 4),
            "n_clusters": n_clusters, "noise_share": round(noise, 4), "valid": valid, "labels": labels}


def _best(tries: list[dict]) -> dict | None:
    valid = [t for t in tries if t["valid"]]
    # Highest silhouette; on ties, fewer clusters.
    return max(valid, key=lambda t: (t["silhouette"], -t["n_clusters"])) if valid else None


def run_kmeans(Z, device, rng):
    from sklearn.cluster import KMeans

    tries, curve = [], []
    for k in range(2, min(MAX_K, max(2, len(Z) // 10)) + 1):
        with gpu_only(device):
            model = KMeans(n_clusters=k, n_init=3, random_state=RANDOM_STATE)
            labels = model.fit_predict(Z)
        t = _evaluate("K-Means", {"n_clusters": k}, labels, Z, device, rng)
        t["centers"] = np.asarray(model.cluster_centers_, dtype=np.float64)
        curve.append({"k": k, "silhouette": t["silhouette"], "inertia": round(float(model.inertia_), 4)})
        tries.append(t)
    return tries, curve


def run_hdbscan(Z, device, rng):
    import hdbscan

    n = len(Z)
    tries = []
    for m in sorted({max(5, n // 100), max(10, n // 50), max(15, n // 25)}):
        with gpu_only(device):
            labels = hdbscan.HDBSCAN(min_cluster_size=m).fit_predict(Z)
        tries.append(_evaluate("HDBSCAN", {"min_cluster_size": m}, labels, Z, device, rng))
    return tries


def run_dbscan(Z, device, rng):
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    min_samples = int(min(20, max(5, 2 * Z.shape[1])))
    sample = Z[rng.choice(len(Z), min(len(Z), SILHOUETTE_SAMPLE), replace=False)]
    with gpu_only(device):
        kdist = NearestNeighbors(n_neighbors=min_samples).fit(sample).kneighbors(sample)[0][:, -1]
    tries = []
    for q in (0.90, 0.95, 0.98):
        eps = float(np.quantile(kdist, q))
        if eps <= 0:
            continue
        with gpu_only(device):
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
        t = _evaluate("DBSCAN", {"eps": round(eps, 6), "min_samples": min_samples, "eps_quantile": q},
                      labels, Z, device, rng)
        t["radius"] = eps
        tries.append(t)
    return tries


def run_spectral(Z, k, device, rng):
    from sklearn.cluster import SpectralClustering

    with gpu_only(device):
        labels = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=min(10, len(Z) - 1),
                                    assign_labels="kmeans", random_state=RANDOM_STATE).fit_predict(Z)
    return [_evaluate("Spectral", {"n_clusters": k}, labels, Z, device, rng)]


def _reference_assignment(Z, labels, radius, device, rng) -> dict:
    """Sampled training points and labels; new rows take the nearest one's label
    within `radius` (or the 95th percentile nearest-neighbour distance inside clusters)."""
    idx = np.flatnonzero(labels >= 0)
    if len(idx) > REFERENCE_POINTS:
        idx = np.sort(rng.choice(idx, REFERENCE_POINTS, replace=False))
    points, ref_labels = Z[idx], labels[idx]
    if radius is None:
        from sklearn.neighbors import NearestNeighbors

        gaps = []
        for c in np.unique(ref_labels):
            members = points[ref_labels == c]
            if len(members) < 2:
                continue
            with gpu_only(device):
                d = NearestNeighbors(n_neighbors=2).fit(members).kneighbors(members)[0][:, 1]
            gaps.append(d)
        radius = float(np.quantile(np.concatenate(gaps), 0.95)) if gaps else 0.0
    return {"kind": "reference", "points": points.tolist(), "labels": ref_labels.tolist(), "radius": radius}


def run(data: bytes, spec: dict, device: str) -> dict:
    """spec: {"candidates": [...], "plan": {"text", "raw_feature_cols"}}"""
    started = time.monotonic()
    plan = spec.get("plan", {})
    X = pd.read_parquet(io.BytesIO(data))
    rng = np.random.default_rng(RANDOM_STATE)

    pre = build_preprocessor(X, "scaled")
    Z = np.asarray(pre.fit_transform(X), dtype=np.float64)
    reducer = None
    if Z.shape[1] > MAX_DIMS:
        from sklearn.decomposition import PCA

        with gpu_only(device):
            reducer = PCA(n_components=MAX_DIMS, random_state=RANDOM_STATE).fit(Z)
            Z = np.asarray(reducer.transform(Z), dtype=np.float64)

    candidates, all_tries, curve = [], [], []
    runners = {
        "K-Means": lambda: run_kmeans(Z, device, rng),
        "HDBSCAN": lambda: (run_hdbscan(Z, device, rng), None),
        "DBSCAN": lambda: (run_dbscan(Z, device, rng), None),
    }
    for name in spec["candidates"]:
        if name == "Spectral":
            continue  # needs K-Means' k; runs below
        try:
            tries, extra = runners[name]()
        except Exception as e:
            candidates.append({"name": name, "status": "failed", "reason": str(e)[:300]})
            continue
        if extra:
            curve = extra
        all_tries += tries
        best = _best(tries)
        candidates.append(_summary(name, best, tries))

    if "Spectral" in spec["candidates"]:
        kmeans = _best([t for t in all_tries if t["name"] == "K-Means"])
        if len(Z) > SPECTRAL_MAX_ROWS:
            candidates.append({"name": "Spectral", "status": "skipped",
                               "reason": f"skipped: spectral clustering is impractical above {SPECTRAL_MAX_ROWS:,} rows"})
        elif kmeans is None:
            candidates.append({"name": "Spectral", "status": "skipped", "reason": "skipped: no k from K-Means"})
        else:
            try:
                tries = run_spectral(Z, kmeans["n_clusters"], device, rng)
                all_tries += tries
                candidates.append(_summary("Spectral", _best(tries), tries))
            except Exception as e:
                candidates.append({"name": "Spectral", "status": "failed", "reason": str(e)[:300]})

    result = {"task": "clustering", "selection_metric": "silhouette", "candidates": candidates,
              "kmeans_curve": curve, "library_versions": library_versions(), "device": device}
    winner = _best(all_tries)
    if winner is None:
        result.update(best=None, seconds=round(time.monotonic() - started, 1))
        return result

    labels = winner["labels"]
    clustered = labels >= 0
    if winner["name"] == "K-Means":
        assignment = {"kind": "centroids", "centers": winner["centers"].tolist()}
    else:
        assignment = _reference_assignment(Z, labels, winner.get("radius"), device, rng)

    from sklearn.decomposition import PCA

    proj_idx = np.sort(rng.choice(len(Z), min(len(Z), PROJECTION_POINTS), replace=False))
    with gpu_only(device):
        coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Z[proj_idx])

    buf = io.BytesIO()
    joblib.dump({
        "format": BUNDLE_FORMAT, "task": "clustering", "problem_type": "clustering", "model_name": winner["name"],
        "feature_cols": plan.get("raw_feature_cols") or X.columns.tolist(), "class_labels": None,
        "preprocessor": pre, "reducer": reducer, "assignment": assignment, "text": plan.get("text"),
        "estimator": {"kind": "cluster-assignment"},
    }, buf)
    sizes = pd.Series(labels).value_counts().sort_index()
    result.update({
        "best": winner["name"],
        "model_name": winner["name"],
        "params": winner["params"],
        "cv_mean": winner["silhouette"],
        "labels": labels.tolist(),
        "metrics": {
            "silhouette": winner["silhouette"],
            "davies_bouldin": round(float(davies_bouldin_score(Z[clustered], labels[clustered])), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(Z[clustered], labels[clustered])), 4),
            "n_clusters": winner["n_clusters"],
            "noise_share": winner["noise_share"],
            "cluster_sizes": {str(int(k)): int(v) for k, v in sizes.items()},
        },
        "projection": {"x": coords[:, 0].tolist(), "y": coords[:, 1].tolist(), "labels": labels[proj_idx].tolist()},
        "feature_cols": plan.get("raw_feature_cols") or X.columns.tolist(),
        "bundle_bytes": buf.getvalue(),
        "seconds": round(time.monotonic() - started, 1),
    })
    return result


def _summary(name, best, tries) -> dict:
    tried = [{k: t[k] for k in ("params", "silhouette", "n_clusters", "noise_share")} for t in tries]
    if best is None:
        return {"name": name, "status": "no_structure", "tries": tried,
                "reason": "no setting produced 2+ clusters with at most half the rows as noise"}
    return {"name": name, "status": "ok", "cv_mean": best["silhouette"], "n_clusters": best["n_clusters"],
            "noise_share": best["noise_share"], "params": best["params"], "tries": tried}
