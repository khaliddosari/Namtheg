"""Per-run artifact storage.

The working copy lives on local disk (the Modal Volume at /storage in
production). When R2 is configured (app/r2.py), every artifact written through
this module is mirrored to R2, and any artifact missing locally is restored
from R2 on first read, so a run survives losing its local copy.

Write artifacts through write_json/save_engineered, or write the file and then
call persist(); anything written another way is not mirrored.
"""
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from app import r2
from app.config import settings

log = logging.getLogger(__name__)

DATASET_FILE = "dataset.parquet"
ENGINEERED_FILE = "engineered.parquet"

_RUN_ID = re.compile(r"[0-9a-f]{12}")

_storage_volume = None


def _get_volume():
    global _storage_volume
    if _storage_volume is None and str(settings.storage_dir) == "/storage":
        try:
            import modal
            _storage_volume = modal.Volume.from_name("modelforge-storage", create_if_missing=True)
        except Exception:
            pass
    return _storage_volume


def sync_reload() -> None:
    vol = _get_volume()
    if vol:
        try:
            vol.reload()
        except Exception:
            pass


def sync_commit() -> None:
    vol = _get_volume()
    if vol:
        try:
            vol.commit()
        except Exception:
            pass


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def run_dir(run_id: str) -> Path:
    p = settings.storage_dir / "runs" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _local(run_id: str, name: str) -> Path:
    """Path of an artifact, restored from the volume or R2 if it's missing."""
    p = run_dir(run_id) / name
    if not p.exists():
        sync_reload()
    if not p.exists():
        r2.download(run_id, name, p)
    return p


def persist(run_id: str, name: str) -> None:
    """Make a just-written artifact durable: commit the volume, mirror to R2."""
    sync_commit()
    path = run_dir(run_id) / name
    if path.exists():
        r2.upload(path, run_id, name)


def raw_upload_path(run_id: str, extension: str) -> Path:
    """Where the untouched upload is kept, for audit and re-ingestion.
    The artifact name to persist() is f"raw/source{extension}"."""
    p = run_dir(run_id) / "raw"
    p.mkdir(exist_ok=True)
    return p / f"source{extension}"


def dataset_path(run_id: str) -> Path:
    """The run's canonical dataset: a typed Parquet copy made at upload time.

    Runs uploaded before the Parquet switch only have dataset.csv; they are
    converted here on first access so every stage can assume Parquet.
    """
    p = _local(run_id, DATASET_FILE)
    if not p.exists():
        legacy = _local(run_id, "dataset.csv")
        if legacy.exists():
            from app.data.ingest import ingest_file
            write_json(run_id, "ingest.json", ingest_file(legacy, p))
            persist(run_id, DATASET_FILE)
    return p


def load_dataset(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(dataset_path(run_id))


def dataset_columns(run_id: str) -> list[str]:
    return pq.read_schema(dataset_path(run_id)).names


def dataset_head(run_id: str, n: int) -> pd.DataFrame:
    return _parquet_head(dataset_path(run_id), n)


def engineered_path(run_id: str) -> Path:
    return _local(run_id, ENGINEERED_FILE)


def save_engineered(run_id: str, df: pd.DataFrame) -> None:
    df.to_parquet(run_dir(run_id) / ENGINEERED_FILE, index=False, compression="zstd")
    persist(run_id, ENGINEERED_FILE)


def load_engineered(run_id: str) -> pd.DataFrame:
    p = engineered_path(run_id)
    legacy = run_dir(run_id) / "engineered.csv"
    if not p.exists() and legacy.exists():
        return pd.read_csv(legacy)
    return pd.read_parquet(p)


def engineered_head(run_id: str, n: int) -> pd.DataFrame:
    p = engineered_path(run_id)
    legacy = run_dir(run_id) / "engineered.csv"
    if not p.exists() and legacy.exists():
        return pd.read_csv(legacy, nrows=n)
    return _parquet_head(p, n)


def _parquet_head(path: Path, n: int) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    batch = next(pf.iter_batches(batch_size=n), None)
    if batch is None:
        return pf.schema_arrow.empty_table().to_pandas()
    return batch.to_pandas()


def artifact_path(run_id: str, name: str) -> Path:
    return _local(run_id, name)


def write_json(run_id: str, name: str, payload: Any) -> Path:
    path = run_dir(run_id) / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    persist(run_id, name)
    return path


def read_json(run_id: str, name: str) -> Any:
    path = artifact_path(run_id, name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(run_id: str, status: str, **extra) -> None:
    current = read_json(run_id, "status.json") or {}
    current.update({"status": status, **extra})
    write_json(run_id, "status.json", current)


def update_status(run_id: str, **fields) -> None:
    """Merge fields into the run status without changing the status itself."""
    current = read_json(run_id, "status.json") or {}
    current.update(fields)
    write_json(run_id, "status.json", current)


def read_status(run_id: str) -> dict:
    return read_json(run_id, "status.json") or {"status": "unknown"}


def run_exists(run_id: str) -> bool:
    # Run ids come from URLs; anything but our own format (e.g. "..") is rejected
    # before it can become a filesystem path or an R2 key.
    if not _RUN_ID.fullmatch(run_id or ""):
        return False
    p = settings.storage_dir / "runs" / run_id
    if p.exists():
        return True
    sync_reload()
    if p.exists():
        return True
    # Restore from R2: status.json is written for every run at upload time.
    if r2.exists(run_id, "status.json"):
        return r2.download(run_id, "status.json", run_dir(run_id) / "status.json")
    return False
