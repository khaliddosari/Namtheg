/**
 * The six pipeline cells the training stage shows. The code and log lines are
 * illustrative of what each stage does; which cell is active comes from the
 * run's real stage (see STAGE_TO_STEP).
 */
export const STEPS = [
  {
    key: "profile",
    fn: "profile_dataset(run_id)",
    label: "Profile Dataset",
    code: `import pandas as pd
import numpy as np

# Load dataset and profile basic statistics
df = pd.read_csv("storage/runs/{run_id}/dataset.csv")
null_counts = df.isnull().sum()
dtypes = df.dtypes

print(f"Loaded {len(df)} rows across {len(df.columns)} columns.")
print("Missing values mapped.")`,
    outputLines: [
      "[SYSTEM] Initializing reader buffers...",
      "[INFO] Reading column metadata & binary bytes...",
      "[INFO] Scanning dtypes and distinct cardinality counts",
      "[INFO] Evaluating missing values frequency per feature",
      "[INFO] Collecting 5-row sample matrix for schema verification",
      "[SUCCESS] Profiling complete. Dataset parsed successfully."
    ],
  },
  {
    key: "detect",
    fn: "detect_problem_type(run_id, target)",
    label: "Detect Problem Type",
    code: `# Determine ML objective (Classification vs Regression)
target_col = df[target]
cardinality = target_col.nunique()

if target_col.dtype == 'object' or cardinality < 15:
    problem_type = "classification"
else:
    problem_type = "regression"

print(f"Target variable has {cardinality} classes. Selected: {problem_type}")`,
    outputLines: [
      "[SYSTEM] Fetching configured target column...",
      "[INFO] Inspecting target column data type and structure",
      "[INFO] Counting unique values in target labels",
      "[INFO] Evaluating cardinality threshold limit (threshold=15)",
      "[SUCCESS] Problem type successfully determined."
    ],
  },
  {
    key: "eda",
    fn: "run_eda(run_id, target)",
    label: "Explore the Data",
    code: `# Run Automated Exploratory Data Analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corrwith(df[target])

print("Top 5 Pearson correlations with target:")
print(correlations.sort_values(ascending=False).head(5))`,
    outputLines: [
      "[SYSTEM] Launching analytical profiler...",
      "[INFO] Computing numeric summaries (mean, std, min, max)",
      "[INFO] Analyzing target distribution & class balancing",
      "[INFO] Computing Pearson correlations with target label",
      "[INFO] Identifying top features with highest mutual correlation",
      "[SUCCESS] EDA statistical report saved successfully."
    ],
  },
  {
    key: "feature_engineer",
    fn: "feature_engineer(run_id, target)",
    label: "Engineer Features",
    code: `from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Clean missing values and encode text values
imputer = SimpleImputer(strategy='median')
encoder = OneHotEncoder(handle_unknown='ignore')

# Drop columns with > 50% nulls
df_clean = df.drop(columns=[col for col in df if df[col].isnull().mean() > 0.5])`,
    outputLines: [
      "[SYSTEM] Booting preprocessing transformers...",
      "[INFO] Dropping columns with high rate (>50%) missing values",
      "[INFO] Removing ID-like features (cardinality metric ≥95% unique)",
      "[INFO] Applying One-Hot encoding to low-cardinality categoricals",
      "[INFO] Applying Label-encoding to high-cardinality categoricals",
      "[SUCCESS] Engineered feature space saved to storage node."
    ],
  },
  {
    key: "train",
    fn: "train_model(run_id, target, problem_type)",
    label: "Train & Compare Models",
    code: `from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# Compile model candidates and score via 5-Fold Cross Validation
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

scores = {name: cross_val_score(model, X, y, cv=5).mean() for name, model in models.items()}`,
    outputLines: [
      "[SYSTEM] Allocating model candidates memory cache...",
      "[INFO] Splitting dataset → 80% train / 20% test subsets",
      "[INFO] Fitting Imputation Imputer on training samples only",
      "[INFO] Training RandomForestClassifier (n_estimators=200)...",
      "[INFO] Training ExtraTreesClassifier (n_estimators=200)...",
      "[INFO] Training GradientBoostingClassifier (early stopping enabled)...",
      "[INFO] Training LogisticRegression estimator...",
      "[INFO] Evaluating 5-Fold cross-validation scores on all estimators",
      "[INFO] Selecting champion architecture by Cross-Validation mean",
      "[SUCCESS] Training phase completed. Best model compiled."
    ],
  },
  {
    key: "visualize",
    fn: "generate_visualization(run_id, target, problem_type)",
    label: "Generate Results",
    code: `import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Compute evaluation plots and save matrix assets
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Purples)
plt.savefig("storage/runs/{run_id}/plot.png")`,
    outputLines: [
      "[SYSTEM] Rendering performance engine...",
      "[INFO] Loading held-out test predictions matrices",
      "[INFO] Computing Confusion Matrix elements & error residuals",
      "[INFO] Rendering plot visualizations to PNG asset",
      "[SUCCESS] Performance plot successfully saved to run directory."
    ],
  },
];

export type PipelineStep = (typeof STEPS)[number];

/** Backend stage names (Backend/app/agent/orchestrator.py) to the cell they run in. */
export const STAGE_TO_STEP: Record<string, number> = {
  starting: 0,
  profile: 0,
  detect: 1,
  audit: 1,
  analysis: 2,
  eda: 2,
  feature_engineering: 3,
  train: 4,
  visualize: 5,
  report: 5,
  package: 5,
};
