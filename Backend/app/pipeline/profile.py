import pandas as pd

from app import storage


def profile_dataset(run_id: str, df: pd.DataFrame | None = None) -> dict:
    """`df`: the full table, when the caller already holds it."""
    df = storage.load_dataset(run_id) if df is None else df
    columns = []
    for col in df.columns:
        s = df[col]
        columns.append({
            "name": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
            "sample": s.dropna().head(3).tolist(),
        })
    profile = {
        "run_id": run_id,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": columns,
    }
    storage.write_json(run_id, "profile.json", profile)
    return profile
