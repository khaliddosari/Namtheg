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
   OPENROUTER_API_KEY=your_key_here
   OPENROUTER_MODEL=deepseek/deepseek-v4-flash
   MODAL_WORKSPACE=your-modal-username
   ```

### Deploy
Deploy the FastAPI backend:
```bash
modal deploy app/deploy/backend_app.py
```

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

