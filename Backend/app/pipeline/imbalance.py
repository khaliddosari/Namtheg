"""Class-imbalance assessment and the strategy the trainer must apply.

Decided before any model is fit, then enforced by the GPU trainer:
- Balanced class weights, computed from each training fold's own labels
  (never from validation or test rows), so minority classes carry equal total
  weight in the loss. CatBoost gets the equivalent auto_class_weights.
- Model selection and tuning by macro-F1 instead of accuracy, because accuracy
  rewards a model that ignores the minority class.
- Stratified train/test split and stratified CV folds (always on for
  classification).

Deliberately not used: resampling. Oversampling/SMOTE before the split leaks
copies of rows into validation folds, and SMOTE interpolates one-hot encoded
categories into values that don't exist. Class weights get the same effect
without inventing data.
"""
import pandas as pd

# Majority/minority count ratio above which the target counts as imbalanced
# (1.5 = a 60/40 binary split).
IMBALANCE_RATIO_THRESHOLD = 1.5


def assess(y: pd.Series, problem_type: str) -> dict:
    """Imbalance facts plus the strategy the trainer will apply."""
    if problem_type != "classification":
        return {"applies": False, "detected": False, "strategy": None, "selection_metric": "r2"}
    counts = y.dropna().value_counts()
    ratio = float(counts.max() / counts.min()) if len(counts) > 1 else 1.0
    detected = ratio > IMBALANCE_RATIO_THRESHOLD
    return {
        "applies": True,
        "detected": detected,
        "ratio": round(ratio, 2),
        "threshold": IMBALANCE_RATIO_THRESHOLD,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "minority_class": str(counts.idxmin()),
        "minority_share": round(float(counts.min() / counts.sum()), 4),
        "strategy": "balanced_class_weights" if detected else None,
        "selection_metric": "f1_macro" if detected else "accuracy",
    }
