from typing import Literal, Optional
from pydantic import BaseModel, Field


ProblemType = Literal["regression", "classification"]
RunStatus = Literal["uploaded", "running", "succeeded", "failed"]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing: int
    unique: int
    sample: list


class DatasetProfile(BaseModel):
    run_id: str
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]


class DetectionResult(BaseModel):
    problem_type: ProblemType
    reason: str
    target: str
    target_dtype: str
    target_unique: int


class EDAReport(BaseModel):
    target: str
    numeric_summary: dict
    target_distribution: dict
    correlations_with_target: dict


class FeatureEngineeringReport(BaseModel):
    dropped_columns: list[str]
    encoded_columns: list[str]
    final_feature_count: int


class TrainingMetrics(BaseModel):
    model_name: str
    score: float
    score_metric: str
    extra: dict


class RunResult(BaseModel):
    run_id: str
    status: RunStatus
    target: Optional[str] = None
    problem_type: Optional[ProblemType] = None
    accuracy_score: Optional[float] = None
    score_metric: Optional[str] = None
    plot_path: Optional[str] = None
    justification: Optional[str] = None
    error: Optional[str] = None


class StartRunRequest(BaseModel):
    task: Literal["auto", "classification", "regression", "clustering", "forecasting",
                  "image_classification"] = "auto"
    target: Optional[str] = Field(None, description="Target column; leave empty for clustering.")
    date_column: Optional[str] = Field(None, description="Forecasting: the timestamp column.")
    series_id_column: Optional[str] = Field(None, description="Forecasting: identifies each series, if several.")
    horizon: Optional[int] = Field(None, description="Forecasting: steps ahead; defaults by frequency.")


class DirectUploadRequest(BaseModel):
    filename: str
    size: int = Field(..., ge=1, description="File size in bytes.")
