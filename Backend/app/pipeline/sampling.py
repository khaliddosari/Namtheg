"""The run's working set: the rows the analyst agent investigates and the models
train on.

Tables up to SAMPLE_ROWS rows are used whole. Larger ones are cut once, at the
start of the run, to a reproducible random sample of SAMPLE_ROWS rows that have
a target value, stratified by the target when it looks like classes so every
class keeps its share. Profiling and the data audit still see every row; the
analyst's sandbox, data exploration, feature engineering and training all use
the same sample, and the final model is trained on it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app import storage

SAMPLE_ROWS = 300_000
SAMPLE_FILE = "sample.parquet"
MAX_STRATA = 100
RANDOM_STATE = 42


def working_set(run_id: str, df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    """-> (the working rows, a description of how they were chosen). A drawn
    sample is saved as sample.parquet for the analyst's sandbox."""
    info = {"rows_total": int(len(df)), "rows_used": int(len(df)), "sampled": False, "stratified": False}
    if len(df) <= SAMPLE_ROWS:
        storage.write_json(run_id, "sample.json", info)
        return df, info

    # Positions only: copying a multi-million-row frame just to sample it would
    # double the job's memory.
    labelled = np.flatnonzero(df[target].notna().to_numpy())
    chosen = labelled
    if len(labelled) > SAMPLE_ROWS:
        y = df[target].to_numpy()[labelled]
        counts = pd.Series(y).value_counts()
        # train_test_split needs every class in both parts to stratify.
        stratify = len(counts) <= MAX_STRATA and counts.min() >= 2 and len(labelled) - SAMPLE_ROWS >= len(counts)
        chosen, _ = train_test_split(labelled, train_size=SAMPLE_ROWS, random_state=RANDOM_STATE,
                                     stratify=y if stratify else None)
        chosen = np.sort(chosen)  # keep the table's own row order
        info["stratified"] = bool(stratify)
    sample = df.iloc[chosen].reset_index(drop=True)
    sample.to_parquet(storage.run_dir(run_id) / SAMPLE_FILE, index=False, compression="zstd")
    storage.persist(run_id, SAMPLE_FILE)
    info.update(sampled=True, rows_used=int(len(sample)), rows_with_target=int(len(labelled)))
    storage.write_json(run_id, "sample.json", info)
    return sample, info


def working_path(run_id: str, info: dict) -> Path:
    """The file the analyst's sandbox loads as `df`: the sample, or the whole table."""
    return storage.artifact_path(run_id, SAMPLE_FILE) if info["sampled"] else storage.dataset_path(run_id)
