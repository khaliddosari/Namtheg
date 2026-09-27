# Namtheg: Agentic AutoML 🛠️

Namtheg is a premium, end-to-end agentic AutoML platform. It automates the entire machine learning pipeline, from raw data upload (CSV, Excel, Parquet, JSON), profiling, target selection, and feature engineering to model training, evaluation plotting, and instant serverless API deployment.

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

Upload a table (CSV, Excel, Parquet, JSON) or a `.zip` of images, then choose what to learn:

| Task | Models (all trained on NVIDIA H200 GPUs, in parallel) |
|------|--------------------------------------------------------|
| **Predict a column** (classification / regression) | XGBoost, CatBoost, SVM, KNN, logistic / ridge regression |
| **Find groups** (clustering, no target) | K-Means, HDBSCAN, DBSCAN, Spectral |
| **Forecast over time** (planned for removal, see [Docs/PRD.md](Docs/PRD.md) §1) | Chronos-2 (pretrained), LSTM and GRU (RNNs), TCN (1-D CNN), XGBoost on lags |
| **Image classification** | ConvNeXt, EfficientNet, ResNet (pretrained, fine-tuned) |

Every run: the upload becomes a typed DataFrame once; deterministic checks (leakage, duplicates, imbalance, time-series gaps) run first; for tables an analyst agent (GPT-6 Sol) investigates with code in an isolated sandbox; imbalance is handled before modelling; models train as a background job on GPUs with early stopping and pruned Optuna search, next to a naive baseline; free-text columns are embedded with a pretrained multilingual encoder; and the written summary may only cite numbers the pipeline computed. Downloads: the cleaned data (CSV), the model package (weights, `predict.py`), and forecasts.

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
- The backend leverages **Modal** serverless volumes (`modelforge-models`) and the shared app (`modelforge-inference`).
- **No redeploy per model**: the trained model is saved directly to a mounted persistent volume rather than rebuilding a container for each upload.
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
