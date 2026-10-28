# Deployment

Namtheg runs across two primary cloud platforms:

| Tier | Platform | Config |
| --- | --- | --- |
| Next.js frontend | **Vercel** | `Frontend/vercel.json` |
| FastAPI backend | **Modal** (Serverless ASGI) | `Backend/app/deploy/backend_app.py` |
| GPU training | **Modal** (NVIDIA H200) | `Backend/app/training/gpu_app.py` |
| Serverless predictors | **Modal** (Shared inference app) | `Backend/app/deploy/inference_app.py` |

Pushing `main` deploys only the frontend. The three Modal apps run the code uploaded at their last
`modal deploy`, so a backend, training or inference change is live only once that app is deployed.
Until then the dashboard falls back to what the older backend offers: for example one plot instead of
the graph pager, and a text box instead of a list in the prediction form.

---

## Why Modal replaces Render for the Backend

Deploying the backend to Modal solves all Render free-tier constraints:
1. **No Hour Pool Exhaustion**: Modal provides **$30/month in free compute credits** per account, scaling to zero when idle instead of consuming 744+ hours/month against a 750h pool.
2. **Persistent Storage**: Uses a persistent `modal.Volume("modelforge-storage")` mounted at `/storage`, preserving uploads, reports, graphs and models across container restarts.
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
modal deploy app/deploy/backend_app.py     # HTTP API + the run_job function; again after any backend change
modal deploy app/training/gpu_app.py       # GPU training service; again whenever app/training/ changes
modal deploy app/deploy/inference_app.py   # live predictions; again when its pins or app/training/runtime.py change
modal app stop modelforge-train            # the old 8-CPU trainer, no longer used
```

**Runs are background jobs.** `/runs/{id}/start` spawns `run_job` (in `backend_app.py`), which runs the
whole pipeline in its own container with a 6-hour limit, independent of any HTTP request, so there is
no request timeout to hit. The frontend polls `/status`; `/runs/{id}/cancel` stops the job and any GPU
work it started. Locally (`uvicorn`), runs use in-process background tasks instead. `run_job` asks for
8 CPU cores and 8 GB: it reads the full table once and holds it through the audit, then works on at
most 300,000 rows (a random sample of larger tables) for the rest of the run.

**Deploy order when a change spans the apps** (for example `app/training/runtime.py`, the bundle the
trainer writes, or what it returns): deploy `gpu_app.py` and `inference_app.py` first, then
`backend_app.py`, so no model is built or served by an older runtime than the backend expects. A
redeploy takes effect within a minute; requests in that minute can still reach the old version.

**All model training runs on NVIDIA H200s** (`app/training/gpu_app.py`); there is no CPU or local
training path, so runs fail until this service is deployed. Modal only allows GPU functions on
accounts with a **payment method** on file (Settings → Billing), even when paid from free credits.
H200 is billed per second at about $4.54/hr. The service has one function:

| Function | Image | Runs |
|----------|-------|------|
| `train_classic` | cuML, XGBoost, CatBoost, Optuna | one model group per call: boosted trees, or SVM/KNN/linear |

A run trains its two model groups in parallel, on 2 GPU containers.
`MAX_PARALLEL_GPUS` in `gpu_app.py` caps concurrent H200s per function (default 3) to bound cost.

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

R2 also enables **large uploads** (over 30 MB, up to 2 GB): the browser PUTs the file
straight to the bucket with a short-lived URL, skipping the frontend proxy and the backend. That needs
one CORS rule on the bucket (R2 → bucket → Settings → CORS policy):
```json
[{"AllowedOrigins": ["https://namtheg.khalid-ai.dev"], "AllowedMethods": ["PUT"], "AllowedHeaders": ["*"], "MaxAgeSeconds": 3600}]
```
Downloads need no CORS rule: they are plain browser navigations. Without R2, uploads are capped at 30 MB.

### Sign-in by email (optional)
Accounts are a name and an email, with no passwords: every sign-in and sign-up is confirmed with a
6-digit code sent by email. A signed-in user gets "My runs", a list of the runs they upload or start.
Run links keep working for anyone who has them, signed in or not. Until the steps below are done, the
site shows no sign-in button.

1. Generate a secret and add it to `Backend/.env`:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```
   ```ini
   AUTH_SECRET=<the output>
   ```
   It signs the session cookies. Changing it later signs everyone out; accounts and their runs are kept.
2. Choose an email sender. Any SMTP service works; two that cost nothing at this scale:
   - **Resend** (recommended). The free plan sends 3,000 emails a month, at most 100 a day, from one
     domain. At the cap, sending pauses until the next day; nothing is billed. Add a domain you own
     (e.g. `khalid-ai.dev`) in Resend, add the DNS records it lists in Cloudflare, then create an
     API key with sending access:
     ```ini
     SMTP_HOST=smtp.resend.com
     SMTP_PORT=465
     SMTP_USERNAME=resend
     SMTP_PASSWORD=<the Resend API key>
     SMTP_FROM=Namtheg <login@khalid-ai.dev>
     ```
   - **Gmail**, with no domain to set up. It needs 2-Step Verification on the account and an
     [app password](https://myaccount.google.com/apppasswords):
     ```ini
     SMTP_HOST=smtp.gmail.com
     SMTP_PORT=465
     SMTP_USERNAME=you@gmail.com
     SMTP_PASSWORD=<the 16-character app password>
     SMTP_FROM=Namtheg <you@gmail.com>
     ```
3. Redeploy the backend: `.env` reaches Modal only at deploy time.
   ```bash
   modal deploy app/deploy/backend_app.py
   ```
   `GET /auth/me` then answers `"enabled": true`, and the sign-in button appears in the header.

Every code is an email, so `AUTH_MAX_CODES_PER_DAY` (default 100, Resend's free daily cap) stops the
site from sending more than that in a day. Each address is also limited to one code a minute and five
an hour, and a code allows five tries and expires after 10 minutes.

Accounts are stored next to the runs on the Modal Volume (`/storage/accounts`) and mirrored to R2
when R2 is on. Each account record holds the name, the email and its list of runs. Codes are stored
only as keyed hashes, and are never mirrored.

For a local backend with no email set up, `AUTH_DEV_LOG_CODES=true` (with any `AUTH_SECRET`) writes
each code to the server log instead of sending it. Never set it on Modal.

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

