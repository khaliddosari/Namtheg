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

## What it does

| Task | When | Models (all trained on NVIDIA H200 GPUs) | Selected by |
|------|------|------------------------------------------|-------------|
| Classification / regression | a target column | XGBoost (depth-wise and leaf-wise), CatBoost, SVM, KNN, logistic/ridge regression | CV accuracy, macro F1 (imbalanced), or R² |

Classical supervised machine learning only: no clustering, neural networks, pretrained models or text embeddings. A free-text column is treated like any other text column (dropped as an identifier when nearly every value is unique).

## How a run works

1. **Upload → DataFrame, once.** `app/data/ingest.py` parses CSV/TSV/TXT, Excel, Parquet, JSON or JSONL into a typed Parquet file. Nothing is silently reinterpreted; every change is listed in `ingest.json`. Tables also convert standalone: `python -m app.data.ingest input.xlsx output.parquet`.
2. **Task, validated up front** (`pipeline/tasks.py`): bad combinations (e.g. a missing target, a non-numeric regression target) fail in the request, before any GPU time.
3. **Background job.** The run executes as its own Modal job (`run_job`, 6-hour limit), detached from HTTP, and can be cancelled (`/cancel` also stops its GPU calls).
4. **Deterministic facts, from one read.** The job reads the full table once; profile, detection, and an audit (leakage suspects via model-free held-out checks, duplicates, imbalance, identifiers, temporal columns) see every row. Tables over 300,000 rows are then cut to one reproducible random sample of 300,000 rows, stratified by the target's classes (`pipeline/sampling.py`). The analyst, data exploration, feature engineering and training all use that same sample, and the final model is trained on it.
5. **Analyst agent** (`agent/analyst.py`). GPT-6 Sol investigates the working set with pandas code in an isolated sandbox and submits an analysis: a data dictionary (what a row is, what the target and key columns mean, and whether each meaning rests on the data or only on the name), the problem type, columns to drop, and cleaning: codes that mean missing (-999, "?"), target labels written several ways, number codes that are really categories, and date columns to turn into year, month, day, weekday or hour. Each decision is checked against the data in code. The cleaning is applied to training and, by the model itself (`training/runtime.py`), to every prediction, so predictions take raw values. Text whose numbers appear in no tool output is marked `verified: false`.
6. **Imbalance plan, before modelling** (`pipeline/imbalance.py`): balanced class weights inside each training fold and macro-F1 selection when the largest class exceeds 1.5x the smallest. No resampling.
7. **Parallel GPU training** (`pipeline/train.py` → `training/gpu_app.py`). Model groups run at the same time on separate H200 containers. Fast optimisation: early stopping for boosted trees, Optuna TPE with median pruning (weak trials stop after a fold or two) that ends once 5 trials in a row fail to beat the best by more than its CV noise. cuML runs scikit-learn's SVM/KNN/linear models on the GPU, and a profiler check rejects any call that fell back to CPU. There is no local or CPU fallback anywhere.
8. **Grounded report** (`agent/report.py`). Every number in the justification must match a computed value, or the text is replaced by a deterministic template.

### LLMs

| Role | Model | API |
|------|-------|-----|
| Primary | `gpt-6-sol` (`LLM_PRIMARY_MODEL`) | OpenAI Responses API (`OPENAI_API_KEY`) |
| Fallback | `deepseek/deepseek-v4-flash` (`OPENROUTER_MODEL`) | OpenRouter Chat Completions (`OPENROUTER_API_KEY`) |

A run switches to the fallback when the primary fails, and stays on it for the rest of that run. Calls use `store=False`. Check both keys with `python -m scripts.check_llm`.

### Downloads and R2

Every run offers the cleaned data as CSV and a model package: `model.joblib`, `runtime.py` (the same loader the live endpoint uses), a `predict.py` for every task type, `metadata.json`, and a `requirements.txt` listing only what that model needs. Models are stored portably (XGBoost JSON, CatBoost `.cbm`, numpy arrays), never as library pickles that break across builds.

With `R2_*` set, every artifact is mirrored to a private Cloudflare R2 bucket, runs missing locally are restored from it, downloads redirect to presigned URLs that expire after 15 minutes, and files over 30 MB upload straight to the bucket.

### Sandbox

`SANDBOX_BACKEND=modal` (default) runs the agent's code in a Modal Sandbox: gVisor isolation, **no network**, no secrets, no volumes, CPU/memory/time limits. The backend pushes in only the run's Parquet file. `local` runs an unisolated subprocess for development and tests; `none` disables code execution.

### Accounts (optional)

`app/accounts.py`: an account is a name and an email, with no password. Every sign-in and sign-up is confirmed with a 6-digit code sent by email over SMTP (`app/mailer.py`; any provider, set up in [../Docs/DEPLOYMENT.md](../Docs/DEPLOYMENT.md)). A signed-in user's uploads and starts are listed under the account ("My runs"); run links stay open to anyone who has them, so signing in is never required. The feature is off, with no sign-in shown, unless `AUTH_SECRET` is set and email can be sent.

- Codes expire after 10 minutes and allow 5 tries. An address gets at most one code a minute and five an hour, and `AUTH_MAX_CODES_PER_DAY` caps the site's total.
- Asking for a code answers the same whether or not the address has an account; an address without one gets no email unless a name comes with it.
- The session is a signed HttpOnly cookie (30 days). The browser reaches the API through the frontend's same-origin proxy, which passes the cookie through, and CORS allows no credentials.
- Files under `storage/accounts/` are named by a hash of the address. Codes are kept only as keyed hashes; account records are mirrored to R2 when it's on.
- Locally, `AUTH_DEV_LOG_CODES=true` writes codes to the server log instead of sending them.

## Run it

```bash
cd Backend
uv venv --python 3.12 .venv
.venv\Scripts\activate          # PowerShell:  .venv\Scripts\Activate.ps1
uv pip install -r requirements-dev.txt
copy .env.example .env          # add OPENAI_API_KEY and OPENROUTER_API_KEY
uvicorn app.main:app --reload --port 8000
python -m pytest tests          # engines run in-process on CPU as a harness; no keys or GPU needed
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health` | Health, LLM config, sandbox backend, training service, R2 |
| `POST` | `/upload` (multipart `file`) | Upload a table (≤ 30 MB) → `run_id`, columns, preview, ingest report |
| `POST` | `/uploads/direct` → `/uploads/{run_id}/complete` | Larger files (≤ 2 GB): presigned PUT straight to R2; `/complete` answers 202 and reads the file in the background, and the run's status turns `uploaded` (or `failed` with the reason) |
| `GET`  | `/runs/{run_id}/preview` | First 20 rows + column list |
| `POST` | `/runs/{run_id}/start` | Body `{"task", "target"}`; `{"target": ...}` alone auto-detects classification or regression |
| `POST` | `/runs/{run_id}/cancel` | Stop the job and its GPU work |
| `GET`  | `/runs/{run_id}/status` | `queued` → `running` (with `stage`) → `succeeded` / `failed` / `cancelled` |
| `GET`  | `/runs/{run_id}/diagnostics` | Stage, task, GPU, GPU containers, elapsed time (recorded state only) |
| `GET`  | `/runs/{run_id}/result` | Final JSON: scores, baseline, audit, analysis, grounded justification |
| `GET`  | `/runs/{run_id}/agent_log` | Every code cell the analyst ran in the sandbox, with its output |
| `GET`  | `/runs/{run_id}/download/{cleaned_csv,model}` | Downloads |
| `GET`  | `/runs/{run_id}/plot` | Confusion matrix or predicted vs actual |
| `GET`  | `/auth/me` | Whether accounts are on, and who is signed in |
| `POST` | `/auth/code` | Body `{"email", "name"?}`: emails a code; a name creates the account |
| `POST` | `/auth/verify` | Body `{"email", "code"}`: sets the session cookie |
| `POST` | `/auth/logout` | Clears the session cookie |
| `GET`  | `/auth/runs` | The signed-in user's runs, newest first |
| `POST` | `/auth/runs/{run_id}` | Adds a run to the signed-in user's list (the one open when they signed in) |

## Layout

```
Backend/
  app/
    main.py                       FastAPI routes
    accounts.py                   optional sign-in by emailed code, sessions, each account's runs
    mailer.py                     sends the codes over SMTP
    jobs.py                       where runs execute (Modal job or local background task)
    config.py                     env-driven settings
    llm.py                        primary/fallback LLM client (Responses + Chat Completions)
    storage.py                    run storage (Modal Volume in prod, ./storage locally), R2 mirror
    r2.py                         Cloudflare R2: mirror, restore, presigned uploads and downloads
    data/ingest.py                any table → typed Parquet DataFrame (also a CLI)
    sandbox/                      the analyst's isolated code sandbox
    pipeline/
      tasks.py                    task resolution and validation
      profile.py, detect.py, eda.py
      audit.py                    data-quality and leakage checks (model-free)
      imbalance.py                class-imbalance assessment and strategy
      feature_engineering.py      structural drops (incl. the analyst's)
      train.py                    fans training out to the GPU service, stores results
      export.py                   cleaned CSV, model package
      visualize.py                the result plot for each task
    training/                     runs inside the GPU containers
      gpu_app.py                  Modal service: H200 function, image, pins
      tabular.py                  classification/regression engine
      runtime.py                  loads and runs every model type (also shipped in packages)
      common.py                   shared helpers
    agent/
      orchestrator.py             runs the stages for each task
      analyst.py                  data-scientist agent (sandboxed code)
      report.py                   grounded justification
    deploy/                       Modal apps: backend (API + jobs), inference, model upload
  scripts/check_llm.py            live tool-calling check for each LLM key
  tests/
```

## Per-run artifacts

Every run produces these files under `storage/runs/<run_id>/`:

- `raw/source.<ext>` - the untouched upload
- `dataset.parquet` - the canonical DataFrame every stage reads; `ingest.json` says how it was made
- `sample.parquet`, `sample.json` - the working set for tables over 300,000 rows, and how it was drawn
- `engineered.parquet` - the modelling data; `cleaned.csv` is its download copy
- `imbalance.json`, `audit.json`, `analysis.json`, `feature_engineering.json`, `metrics.json`
- `model.joblib`, `model_meta.json` (+ `model_weights.*` for tables) - the trained model; `model_package.zip` bundles it for download
- `agent_log.json` - the analyst's code cells and outputs
- `plot.png`, `status.json`, `result.json`

Runs created before the Parquet switch (only `dataset.csv`) are converted on first access.

## Deploying inference to Modal (shared app)

The "Deploy" button on the result page uploads each trained model to a single
shared Modal app (`modelforge-inference`), instead of creating a brand new app
per run. This keeps Modal's free tier viable when many people use the project:

- One image build (cached after first deploy), not one per upload.
- One deployment slot used forever, not one per upload.
- Models live in a `modelforge-models` Modal Volume, so uploading a new one is
  just a file copy, no `modal deploy` per run.

### One-time setup (do this once per workspace)

1. `pip install modal` and `modal token new` if you haven't already.
2. Add two lines to `Backend/.env`:

   ```
   MODAL_WORKSPACE=your-modal-username
   ```

   Find it by running `modal app list`: it's the workspace name shown at the
   top, or the prefix of any existing app URL (`{workspace}--...modal.run`).

3. Deploy the shared inference app **once**:

   ```bash
   modal deploy app/deploy/inference_app.py
   ```

   Modal builds the image (sklearn + pandas + fastapi pinned to the versions
   in `IMAGE_PIN`) and registers the `modelforge-inference` app. Subsequent
   user "Deploy" clicks just upload model files into the shared volume, with
   no image build and no new app.

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

- **Classical machine learning only.** Forecasting, image classification and free-text embeddings were removed with every deep learning model; see [Docs/PRD.md](../Docs/PRD.md) §8.
- **Dataset contents reach the LLM provider.** The analyst sees 3 sample values per column and whatever its sandbox code prints. Calls use `store=False`, but no PII redaction exists yet.
- **Grounding checks numbers, not attribution.** A justification can't cite a number the pipeline didn't compute, but it could attach a real number to the wrong claim.
- **Tables use a random train/test split.** Date columns in tables are dropped rather than featurised until time-aware splits exist.
- **Binary decision threshold is fixed at 0.5.** Imbalanced runs use class weights, but the threshold isn't tuned for a target precision or recall.
- **The frontend's running page is still simulated** (scripted notebook cells and timings). The backend now exposes the real stages and the analyst's actual code cells (`/agent_log`) for the redesign.
