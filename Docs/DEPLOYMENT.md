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
All model training runs on an NVIDIA **H200** (`app/training/gpu_app.py`); there is no CPU
or local training path, so runs fail until this service is deployed. Modal only allows GPU
functions on accounts with a **payment method** on file (Settings → Billing), even when
paid from free credits. H200 is billed per second at about $4.54/hr.

```bash
modal deploy app/deploy/backend_app.py
modal deploy app/training/gpu_app.py       # whenever app/training/ changes
modal deploy app/deploy/inference_app.py   # whenever its pins or app/training/core.py change
modal app stop modelforge-train            # the old 8-CPU trainer, no longer used
```
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
after 15 minutes. No bucket CORS setup is needed: downloads are plain browser navigations.

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

## 4. Decommissioning Render & Keep-Alive Pingers

Once your Modal backend is live and pointed to by Vercel:
1. **Render**: Delete or suspend the old backend service in your Render dashboard.
2. **GitHub Actions Keep-Alive**: Disable the `.github/workflows/keep-alive.yml` workflow in the GitHub Actions tab (or remove the `KEEPALIVE_BACKEND_URL` variable).
3. **Modal Keep-Alive**: If you previously deployed `app/keepalive.py`, stop it with:
   ```bash
   modal app stop namtheg-keepalive
   ```

---

## Legacy / Fallback: Backend on Render

If you ever need to run on Render instead:
- Blueprint is defined in `render.yaml`.
- Requires Render free tier with `sync: false` variables.
- Note: Keep-alive pinging against Render free burns ~744 hours/month and has ephemeral disk storage.

