from pandas.api.types import is_numeric_dtype

from app import storage


def detect_problem_type(run_id: str, target: str) -> dict:
    df = storage.load_dataset(run_id)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    col = df[target].dropna()
    dtype = str(col.dtype)
    unique = int(col.nunique())
    n = int(len(col))

    # Float columns are almost never discrete class labels.
    is_float = is_numeric_dtype(col) and col.dtype.kind == "f"
    # Cap at 30: more than 30 unique numeric values is virtually always regression.
    classification_threshold = min(max(15, int(0.02 * n)), 30)

    if not is_numeric_dtype(col):
        problem_type = "classification"
        reason = (
            f"Target '{target}' is non-numeric (dtype={dtype}) with {unique} unique "
            f"values, so the problem is classification."
        )
    elif is_float or unique > classification_threshold:
        problem_type = "regression"
        reason = (
            f"Target '{target}' is numeric (dtype={dtype}) with {unique} unique "
            f"values out of {n} rows, indicating a continuous target. Treating as "
            f"regression."
        )
    else:
        problem_type = "classification"
        reason = (
            f"Target '{target}' is numeric but only has {unique} unique values "
            f"out of {n} rows (threshold={classification_threshold}), indicating a "
            f"small discrete label set. Treating as classification."
        )

    result = {
        "problem_type": problem_type,
        "reason": reason,
        "target": target,
        "target_dtype": dtype,
        "target_unique": unique,
    }
    storage.write_json(run_id, "detection.json", result)
    return result
