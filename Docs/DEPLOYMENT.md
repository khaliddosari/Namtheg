# Deployment

Namtheg runs across two primary cloud platforms:

| Tier | Platform | Config |
| --- | --- | --- |
| Next.js frontend | **Vercel** | `Frontend/vercel.json` |
| FastAPI backend | **Modal** (Serverless ASGI) | `Backend/app/deploy/backend_app.py` |
| Serverless predictors | **Modal** (Shared inference app) | `Backend/app/deploy/inference_app.py` |

---

## Why Modal replaces Render for the Backend

Deploying the backend to Modal solves all Render free-tier constraints:
1. **No Hour Pool Exhaustion**: Modal provides **$30/month in free compute credits** per account, scaling to zero when idle instead of consuming 744+ hours/month against a 750h pool.
2. **Persistent Storage**: Uses a persistent `modal.Volume("modelforge-storage")` mounted at `/storage`, preserving CSV uploads, EDA reports, and model plots across container restarts.
3. **No Keep-Alive Pingers**: Modal starts in ~1–2 seconds on incoming requests without requiring artificial pingers.
4. **Adequate RAM**: Allocates 2+ GB RAM and 2 vCPUs, preventing out-of-memory crashes on scikit-learn / pandas data processing (Render free was capped at 512 MB).

---

## 1. Backend on Modal (Recommended)

### One-Time Setup
1. Ensure your Modal CLI is authenticated:
   ```bash
   cd Backend
   pip install modal
   modal token new
   ```
2. Make sure `Backend/.env` contains your configuration:
   ```ini
   OPENAI_API_KEY=your_openai_key          # primary: gpt-6-sol
   OPENROUTER_API_KEY=your_openrouter_key  # fallback: deepseek/deepseek-v4-flash
   SANDBOX_BACKEND=modal
   MODAL_WORKSPACE=your-modal-username
   ```
   Verify both keys before deploying: `python -m scripts.check_llm`.

### Deploy
```bash
modal deploy app/deploy/backend_app.py     # HTTP API + the run_job function
modal deploy app/training/gpu_app.py       # GPU training service; again whenever app/training/ changes
modal deploy app/deploy/inference_app.py   # live predictions; again when its pins or app/training/runtime.py change
modal app stop modelforge-train            # the old 8-CPU trainer, no longer used
```

**Runs are background jobs.** `/runs/{id}/start` spawns `run_job` (in `backend_app.py`), which runs the
whole pipeline in its own container with a 6-hour limit, independent of any HTTP request, so there is
no request timeout to hit. The frontend polls `/status`; `/runs/{id}/cancel` stops the job and any GPU
work it started. Locally (`uvicorn`), runs use in-process background tasks instead.

**All model training runs on NVIDIA H200s** (`app/training/gpu_app.py`); there is no CPU or local
training path, so runs fail until this service is deployed. Modal only allows GPU functions on
accounts with a **payment method** on file (Settings → Billing), even when paid from free credits.
H200 is billed per second at about $4.54/hr. The service has two images:

| Function | Image | Runs |
|----------|-------|------|
| `train_classic` | cuML, XGBoost, CatBoost, Optuna, hdbscan | tabular model groups (boosted trees; SVM/KNN/linear) and clustering |
| `train_deep` | PyTorch, timm, sentence-transformers, Chronos | one forecaster or one image CNN per call |
| `embed_text` | same as `train_deep` | pretrained embeddings for free-text columns |

A run fans out across GPUs in parallel (2 containers for a table, up to 5 for forecasting, 3 for images).
`MAX_PARALLEL_GPUS` in `gpu_app.py` caps concurrent H200s per function (default 3) to bound cost.
Pretrained weights (text encoder, Chronos-2, three CNNs) are baked into the deep image at build time.

The GPU and inference images pin identical library versions (`tests/test_training.py`
enforces it); change them together and redeploy both.

The analyst agent's sandbox needs no deploy step: the backend creates a short-lived
Modal Sandbox per run under the `namtheg-sandbox` app. The first run builds its image
(~30 s); later runs reuse it.

### R2 storage (optional)
1. Cloudflare dashboard → R2 → create a bucket (e.g. `namtheg-runs`). Keep it private.
2. R2 → Manage API tokens → create a token with **Object Read & Write** on that bucket only.
3. Add `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` to `Backend/.env`
   and redeploy the backend. `/health` reports `"r2_enabled": true` when it's on.

Every run artifact is then mirrored to the bucket, runs missing from the Modal Volume are
restored from it, and the result page's downloads redirect to presigned URLs that expire
after 15 minutes.

R2 also enables **large uploads** (over 30 MB, up to 2 GB, e.g. image zips): the browser PUTs the file
straight to the bucket with a short-lived URL, skipping the frontend proxy and the backend. That needs
one CORS rule on the bucket (R2 → bucket → Settings → CORS policy):
```json
[{"AllowedOrigins": ["https://namtheg.khalid-ai.dev"], "AllowedMethods": ["PUT"], "AllowedHeaders": ["*"], "MaxAgeSeconds": 3600}]
```
Downloads need no CORS rule: they are plain browser navigations. Without R2, uploads are capped at 30 MB.

Modal will output your permanent public URL, for example:
```
https://<your-workspace>--namtheg-backend-fastapi-app.modal.run
```

Test health:
```bash
curl https://<your-workspace>--namtheg-backend-fastapi-app.modal.run/health
```

---

## 2. Serverless Predictors on Modal

Deploy the shared inference app once per workspace:
```bash
cd Backend
modal deploy app/deploy/inference_app.py
```

---

## 3. Frontend on Vercel

1. Import the repo at [vercel.com/new](https://vercel.com/new).
2. **Set Root Directory to `Frontend`.** (Required because the repository is a monorepo).
3. Add Environment Variables (Production + Preview):
   - `BACKEND_URL` = Your Modal backend URL from Step 1 (e.g. `https://<workspace>--namtheg-backend-fastapi-app.modal.run`)
   - `NEXT_PUBLIC_SITE_URL` (optional, defaults to `https://namtheg.khalid-ai.dev`) = the production domain. Link-preview images are served from this host, so it must actually serve the app.
4. Deploy (or click "Redeploy" if already connected to update `BACKEND_URL`).

No frontend code changes are needed: `Frontend/next.config.ts` automatically proxies `/api/backend/*` to `BACKEND_URL`.

---

## 4. Decommissioned: Render & Keep-Alive Pingers

The backend is on Modal now, which starts in ~1-2s on incoming requests and needs no keep-alive
pinging. The GitHub Actions workflow and Modal cron that used to ping the old Render backend
(`.github/workflows/keep-alive.yml`, `app/keepalive.py`) have been removed from the repo. If your
Modal workspace still has a `namtheg-keepalive` app deployed from before, stop it:
```bash
modal app stop namtheg-keepalive
```
And delete or suspend the old backend service in your Render dashboard if one is still running.

---

## Legacy / Fallback: Backend on Render

If you ever need to run on Render instead:
- Blueprint is defined in `render.yaml`.
- Requires Render free tier with `sync: false` variables.
- Note: Keep-alive pinging against Render free burns ~744 hours/month and has ephemeral disk storage.

