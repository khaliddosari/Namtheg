"""Finds free-text columns worth embedding (reviews, notes, descriptions).

A text column holds several words per value and is mostly unique. Treated as
a category it is useless (one level per row) and the default pipeline would
drop it as identifier-like; embedding it with a pretrained encoder keeps its
meaning. Short labels like "Riyadh" or "yes" stay categorical.
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype

MIN_AVG_WORDS = 4.0
MIN_UNIQUE_SHARE = 0.3
SAMPLE_ROWS = 2_000


def detect_text_columns(df: pd.DataFrame, exclude: list[str]) -> dict[str, dict]:
    """-> {column: {"avg_words", "unique_share"}} for every free-text column."""
    found = {}
    for c in df.columns:
        if c in exclude or is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        values = df[c].dropna().astype(str)
        if values.empty:
            continue
        sample = values.sample(min(len(values), SAMPLE_ROWS), random_state=0)
        avg_words = float(sample.str.split().str.len().mean())
        unique_share = float(values.nunique() / len(values))
        if avg_words >= MIN_AVG_WORDS and unique_share >= MIN_UNIQUE_SHARE:
            found[c] = {"avg_words": round(avg_words, 1), "unique_share": round(unique_share, 3)}
    return found
