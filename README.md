# Namtheg: Agentic AutoML 🛠️

Namtheg is a premium, end-to-end agentic AutoML platform. It automates the entire machine learning pipeline—from raw data upload (CSV, Excel, Parquet, JSON), profiling, target selection, and feature engineering to model training, evaluation plotting, and instant serverless API deployment.

An **analyst agent (GPT-6 Sol, with DeepSeek V4 Flash as fallback)** investigates each dataset the way a data scientist would, running its own pandas code in an isolated, network-blocked sandbox. Every number it reports is checked against what the pipeline actually computed.

---

## 🏗️ Architecture & Monorepo Layout

This repository is structured as a modern monorepo separating frontend UI from backend execution:

```
├── Backend/                 # Python FastAPI Backend
│   ├── app/
│   │   ├── agent/           # Orchestrator, analyst agent, tuning, grounded report
│   │   ├── data/            # Any upload → typed Parquet DataFrame
│   │   ├── sandbox/         # Isolated code execution for the agent (Modal Sandbox)
│   │   ├── training/        # H200 GPU training service and engine
│   │   ├── pipeline/        # Profiling, audit, feature engineering, training, visualization
│   │   ├── deploy/          # Modal serverless deployment logic
│   │   ├── llm.py           # Primary/fallback LLM client
│   │   ├── storage.py       # Run storage (Modal Volume in production)
│   │   └── main.py          # FastAPI application routes
│   ├── requirements.txt     # Backend Python dependencies
│   └── .env.example         # Example local backend environment variables
│
├── Frontend/                # Next.js 15 Frontend
│   ├── app/                 # Next.js App Router (Upload, Preview, Running, Result, Inference UI)
│   ├── components/          # Reusable UI component library (TailwindCSS + Framer Motion)
│   ├── lib/                 # Frontend API client library
│   ├── package.json         # Node.js dependencies and scripts
│   ├── vercel.json          # Vercel build config (set Root Directory to Frontend)
│   └── .env.local.example   # Example local frontend environment variables
│
├── Docs/                    # Product specs and agent rules
│   ├── PRD.md               # Product Requirements Document
│   ├── DEPLOYMENT.md        # Vercel + Render + Modal deployment guide
│   └── AGENTS.md            # Agent Operational guidelines
│
├── render.yaml              # Render Blueprint config (backend only)
└── insurance.csv            # Sample dataset for demonstration
```

---

## 🧠 AutoML Pipeline Flow

When you select a target column and click **Start AutoML**:

1. **Ingest (at upload)**: the file is converted once into a typed Parquet DataFrame; every later step reads that, never the raw file.
2. **Profile, detect, audit**: schema and missing values, regression vs classification, and deterministic checks for leakage suspects, duplicates, class imbalance, identifiers and temporal columns.
3. **Analyst agent**: GPT-6 Sol writes and runs pandas code in a Modal Sandbox (gVisor, no network, no secrets) to investigate what the audit can't settle, then submits validated decisions: columns to drop with evidence, findings, and open questions it won't guess at.
4. **Imbalance plan**: decided before any model is fit. Imbalanced targets train with balanced class weights (computed inside each fold) and are judged by macro F1, not accuracy.
5. **Train & tune on an H200 GPU**: XGBoost (depth-wise and leaf-wise) and CatBoost, cross-validated with preprocessing fitted inside each fold, next to a feature-blind baseline; the best is tuned with Optuna by CV mean, never the test set. No CPU or local training path exists.
6. **Visualize**: Predicted-vs-Actual (regression) or a confusion matrix (classification).
7. **Grounded justification**: a short write-up whose every number is verified against computed results, with a deterministic fallback.
8. **Downloads**: the cleaned dataset as CSV, and the winning model package (weights in native XGBoost/CatBoost format, metadata, and a working `predict.py`). Optionally stored in Cloudflare R2 and served through expiring links.

---

## 🚀 Local Quickstart

### 1. Prerequisites
- **Python 3.10+** installed
- **Node.js 18+** installed

---

### 2. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd Backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Configure your environment variables:
   ```bash
   copy .env.example .env
   ```
   Open `.env` and fill in:
   - `OPENAI_API_KEY`: primary model (GPT-6 Sol).
   - `OPENROUTER_API_KEY`: fallback model (DeepSeek V4 Flash), from [OpenRouter](https://openrouter.ai/).
   - `SANDBOX_BACKEND`: `modal` (default; needs `modal token new`) or `local` for unisolated development.
   - `MODAL_WORKSPACE`: Your Modal username.

5. Start the backend:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

---

### 3. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd Frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Set up local environment variables:
   ```bash
   copy .env.local.example .env.local
   ```
4. Start the Next.js development server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000` to interact with the UI.

---

## 🌐 Deploying to Production

Namtheg is fully ready for multi-tier production deployment:

Full instructions live in **[Docs/DEPLOYMENT.md](Docs/DEPLOYMENT.md)**. In short:

### 1. FastAPI backend on Modal (Recommended)
Deploy the FastAPI backend serverlessly using Modal ASGI:
```bash
cd Backend
modal deploy app/deploy/backend_app.py
```
- Provisions the backend on Modal with persistent `modal.Volume` storage at `/storage`.
- Scales to zero when idle, wakes in 1–2s, and uses Modal's $30/month free compute credits without Render's 750h quota limits or 512MB RAM constraints.

*(Alternatively, legacy Render Blueprint is available in `render.yaml`).*

### 2. Next.js frontend on Vercel
Import the repo at [vercel.com/new](https://vercel.com/new), **set the Root Directory to `Frontend`**, and add `BACKEND_URL` (your Modal backend's public URL, e.g. `https://<workspace>--namtheg-backend-fastapi-app.modal.run`) plus `NEXT_PUBLIC_SITE_URL`. The frontend proxies the browser's calls to the backend through its own `/api/backend/*` rewrite, so there is no CORS setup.


### 3. Serverless Predictors on Modal
When a user clicks "Deploy to Modal" on their successfully trained model:
- The backend leverages **Modal** serverless volumes (`namtheg-models`) and the shared app (`namtheg-inference`).
- **Zero-cold-start uploads**: The model is saved directly to a mounted persistent volume rather than redeploying containers.
- Interactive serverless prediction endpoints are served dynamically!

#### One-Time Modal Setup:
Ensure you deploy the core serverless inference wrapper once to your Modal space:
```bash
cd Backend
pip install modal
modal token new
modal deploy app/deploy/inference_app.py
```

---

## 📄 License

Copyright (c) 2026 Khalid. **All Rights Reserved.** See [LICENSE](LICENSE).

This project is source-available for viewing only. No use, copying, modification, or distribution is permitted without prior written permission from the copyright holder.
