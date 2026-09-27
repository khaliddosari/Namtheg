from app import storage


def profile_dataset(run_id: str) -> dict:
    df = storage.load_dataset(run_id)
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
