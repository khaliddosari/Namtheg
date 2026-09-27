# Namtheg Backend

AutoML backend: upload a tabular file, pick a target, get a validated model with an evidence-backed write-up. See [../Docs/PRD.md](../Docs/PRD.md) for the product spec and [../Docs/AGENTS.md](../Docs/AGENTS.md) for agent operating rules.

---
### For me
Backend:
cd "c:\Users\Khalid\Downloads\Coding Projects\AutoML\Backend"
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000

Frontend:
cd "C:\Users\Khalid\Downloads\Coding Projects\AutoML\Frontend"
npm run dev

---

## How a run works

1. **Upload → DataFrame, once.** `app/data/ingest.py` parses CSV/TSV/TXT, Excel (`.xlsx`/`.xls`), Parquet, JSON or JSONL into a typed Parquet file. Every later stage loads that file instead of re-parsing the upload. Nothing is silently reinterpreted; every change (renamed headers, dropped empty rows, mixed-type columns stored as text) is listed in `ingest.json`. It also runs standalone: `python -m app.data.ingest input.xlsx output.parquet`.
2. **Deterministic facts.** Profile, problem-type detection, and a data audit (`pipeline/audit.py`): leakage suspects, duplicates, class imbalance, identifier and temporal columns, numbers stored as text.
3. **Analyst agent** (`agent/analyst.py`). GPT-6 Sol investigates the data by writing pandas code that runs in an isolated sandbox, then submits a structured analysis: problem type, columns to drop (with evidence), findings, and open questions it refuses to guess. The submission is validated in code; findings whose numbers appear in no tool output are marked `verified: false`.
4. **Imbalance plan, before modelling** (`pipeline/imbalance.py`). If the largest class has more than 1.5x the rows of the smallest, training uses balanced class weights computed inside each training fold and selects models by macro F1 instead of accuracy. No resampling: it would leak copies across folds or invent rows.
5. **Training on an H200 GPU only** (`training/gpu_app.py` runs `training/core.py`). Three GPU-native candidates (XGBoost depth-wise, XGBoost leaf-wise, CatBoost) are cross-validated with preprocessing fitted inside each fold, next to a feature-blind baseline on the same split; the best is tuned with Optuna (selected by CV mean, never the test set), fit once, and scored on the held-out 20%. There is no CPU or local fallback: if the GPU service fails, the run fails.
6. **Grounded report** (`agent/report.py`). Every number in the written justification must match a computed value; otherwise it's regenerated once, then replaced by a deterministic template.

### LLMs

| Role | Model | API |
|------|-------|-----|
| Primary | `gpt-6-sol` (`LLM_PRIMARY_MODEL`) | OpenAI Responses API (`OPENAI_API_KEY`) |
| Fallback | `deepseek/deepseek-v4-flash` (`OPENROUTER_MODEL`) | OpenRouter Chat Completions (`OPENROUTER_API_KEY`) |

A run switches to the fallback when the primary fails, and stays on it for the rest of that run. Calls use `store=False`. Check both keys with `python -m scripts.check_llm`.

### Downloads and R2

The result offers the cleaned dataset as CSV (the exact rows and columns the models trained on) and a model package: `model.joblib` (fitted preprocessing + model), the native weights (`model_weights.json` for XGBoost, `.cbm` for CatBoost), `metadata.json`, pinned `requirements.txt`, and a working `predict.py`. Models are stored in their native formats rather than pickled, because XGBoost pickles don't load across builds or operating systems.

With `R2_*` set, every artifact is mirrored to a private Cloudflare R2 bucket, runs missing locally are restored from it, and downloads redirect to presigned URLs that expire after 15 minutes. Without R2, the backend streams the files itself.

### Sandbox

`SANDBOX_BACKEND=modal` (default) runs the agent's code in a Modal Sandbox: gVisor isolation, **no network**, no secrets, no volumes, CPU/memory/time limits. The backend pushes in only the run's Parquet file; the sandbox is never given a URL or credential to fetch data. `local` runs an unisolated subprocess for development and tests; `none` disables code execution.

## Run it

```bash
cd Backend
uv venv --python 3.11 .venv
.venv\Scripts\activate          # PowerShell:  .venv\Scripts\Activate.ps1
uv pip install -r requirements-dev.txt
copy .env.example .env          # add OPENAI_API_KEY and OPENROUTER_API_KEY
uvicorn app.main:app --reload --port 8000
python -m pytest tests          # uses the local sandbox and a scripted LLM; no keys needed
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | Health, primary/fallback LLM config, sandbox backend |
| `POST` | `/upload` (multipart `file`) | Upload a dataset (≤ 30 MB) → `run_id`, columns, 5-row preview, ingest report |
| `GET`  | `/runs/{run_id}/preview` | First 20 rows + column list |
| `POST` | `/runs/{run_id}/start` body `{"target": "..."}` | Kick off the run in the background |
| `GET`  | `/runs/{run_id}/status` | `uploaded` → `queued` → `running` (with `stage`) → `succeeded`/`failed` |
| `GET`  | `/runs/{run_id}/result` | Final JSON: scores, baseline, audit, analysis, grounded justification |
| `GET`  | `/runs/{run_id}/agent_log` | Every code cell the analyst ran in the sandbox, with its output |
| `GET`  | `/runs/{run_id}/download/cleaned_csv` | The cleaned dataset (CSV, UTF-8 with BOM) |
| `GET`  | `/runs/{run_id}/download/model` | The winning model package (zip) |
| `GET`  | `/runs/{run_id}/plot` | Returns the generated PNG |

## Layout

```
Backend/
  app/
    main.py                       FastAPI routes
    config.py                     env-driven settings
    llm.py                        primary/fallback LLM client (Responses + Chat Completions)
    storage.py                    run storage (Modal Volume in prod, ./storage locally), R2 mirror
    r2.py                         Cloudflare R2 client: mirror, restore, presigned downloads
    data/ingest.py                any upload → typed Parquet DataFrame (also a CLI)
    sandbox/
      session.py                  Modal / local sandbox sessions (host side)
      driver.py                   code-cell REPL that runs inside the sandbox
    pipeline/
      profile.py, detect.py, eda.py
      audit.py                    deterministic data-quality and leakage checks (model-free)
      imbalance.py                class-imbalance assessment and strategy
      feature_engineering.py      structural drops (incl. the analyst's)
      train.py                    sends training to the GPU service, stores the results
      export.py                   cleaned CSV and model package
      visualize.py                predicted-vs-actual or confusion matrix
    training/
      gpu_app.py                  Modal H200 training service
      core.py                     training engine: CV, baseline, Optuna tuning, portable bundle
    agent/
      orchestrator.py             runs the stages in order
      analyst.py                  data-scientist agent (sandboxed code)
      report.py                   grounded justification
  scripts/check_llm.py            live tool-calling check for each LLM key
  tests/
```

## Per-run artifacts

Every run produces these files under `storage/runs/<run_id>/`:

- `raw/source.<ext>` - the untouched upload
- `dataset.parquet` - the canonical DataFrame every stage reads; `ingest.json` says how it was made
- `engineered.parquet` - after feature engineering; `cleaned.csv` is its download copy
- `imbalance.json` - the imbalance assessment and strategy applied
- `model.joblib`, `model_weights.json|.cbm`, `model_meta.json` - the trained model; `model_package.zip` bundles them for download
- `profile.json`, `detection.json`, `audit.json`, `analysis.json`, `eda.json`, `feature_engineering.json`, `metrics.json`, `visualization.json`
- `agent_log.json` - the analyst's code cells and outputs
- `plot.png` - the result graph
- `status.json` - current lifecycle state and stage
- `result.json` - final output
- `y_test.npy`, `y_pred.npy` - held-out predictions for the plot

Runs created before the Parquet switch (only `dataset.csv`) are converted on first access.

## Deploying inference to Modal (shared app)

The "Deploy" button on the result page uploads each trained model to a single
shared Modal app (`modelforge-inference`), instead of creating a brand new app
per run. This keeps Modal's free tier viable when many people use the project:

- One image build (cached after first deploy), not one per upload.
- One deployment slot used forever, not one per upload.
- Models live in a `modelforge-models` Modal Volume — uploading a new one is
  just a file copy, no `modal deploy` per run.

### One-time setup (do this once per workspace)

1. `pip install modal` and `modal token new` if you haven't already.
2. Add two lines to `Backend/.env`:

   ```
   MODAL_WORKSPACE=your-modal-username
   ```

   Find it by running `modal app list` — it's the workspace name shown at the
   top, or the prefix of any existing app URL (`{workspace}--…modal.run`).

3. Deploy the shared inference app **once**:

   ```bash
   modal deploy app/deploy/inference_app.py
   ```

   Modal builds the image (sklearn + pandas + fastapi pinned to the versions
   in `IMAGE_PIN`) and registers the `modelforge-inference` app. Subsequent
   user "Deploy" clicks just upload model files into the shared volume — no
   image build, no new app.

### When to re-deploy `inference_app.py`

Only when:
- You bump a pinned library version in `IMAGE_PIN` (must match the training
  environment so unpickled models load cleanly).
- You change the `Predictor` class (new endpoint, new preprocessing, etc.).

Per-user model updates don't need a re-deploy.

### URL shape

After deploy, prediction and schema endpoints look like:

```
https://{MODAL_WORKSPACE}--modelforge-inference-predictor-predict.modal.run/?run_id=<id>
https://{MODAL_WORKSPACE}--modelforge-inference-predictor-schema.modal.run/?run_id=<id>
```

The FastAPI backend proxies `/runs/<id>/predict` to these, so the frontend
never has to know the URL or worry about CORS.

## Notes & known gaps

- **Dataset contents reach the LLM provider.** The analyst sees 3 sample values per column and whatever its sandbox code prints. Calls use `store=False`, but no PII redaction exists yet (PRD §6.3 / §7); add it before handling regulated data.
- **Grounding checks numbers, not attribution.** A justification can't cite a number the pipeline didn't compute, but it could attach a real number to the wrong claim.
- **Random train/test split only.** The audit flags temporal columns; time-based splits aren't implemented yet.
- **GPU needs a Modal payment method.** Modal refuses every GPU function (even L4) on accounts without one, so training fails until it's added.
- **Date/time columns are dropped**, not turned into features, until time-aware splits exist.
- **Binary decision threshold is fixed at 0.5.** Imbalanced runs use class weights, but the threshold isn't tuned for a target precision/recall.
- **In-process background tasks**, not a job queue. Swap later when concurrency matters.
