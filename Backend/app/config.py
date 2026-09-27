from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Primary LLM: OpenAI directly, via the Responses API (GPT-6 Sol only
    # supports tool calling with reasoning on /v1/responses).
    openai_api_key: str = ""
    llm_primary_model: str = "gpt-6-sol"
    # none | low | medium | high | xhigh | max
    llm_primary_reasoning_effort: str = "medium"
    llm_timeout_seconds: float = 120.0

    # Fallback LLM: DeepSeek V4 Flash via OpenRouter (Chat Completions).
    # Used whenever the primary is unconfigured or errors out.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "deepseek/deepseek-v4-flash"
    openrouter_referer: str = "http://localhost:8000"
    openrouter_app_title: str = "Namtheg"

    # Where the analyst agent's code runs. "modal" = Modal Sandbox (gVisor,
    # no network, production). "local" = plain subprocess on this machine, NOT
    # isolated, dev/tests only. "none" = agent runs without a code tool.
    sandbox_backend: str = "modal"
    sandbox_app_name: str = "namtheg-sandbox"
    sandbox_cpu: float = 2.0
    sandbox_memory_mb: int = 2048
    # Hard ceiling on the sandbox's whole lifetime, and per code cell.
    sandbox_lifetime_seconds: int = 900
    sandbox_exec_timeout_seconds: int = 60

    # Analyst agent budget. When either runs out the agent is forced to submit.
    agent_max_tool_calls: int = 12
    agent_time_budget_seconds: int = 240

    # GPU training (app/training/gpu_app.py). Hyperparameter search stops at
    # whichever limit comes first.
    tuning_trials: int = 20
    tuning_timeout_seconds: int = 300

    # Cloudflare R2 (S3-compatible). Off unless all four are set. When on,
    # every run artifact is mirrored to R2, runs missing locally are restored
    # from it, and downloads are served as time-limited presigned URLs.
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""
    r2_prefix: str = ""
    r2_url_ttl_seconds: int = 900

    storage_dir: Path = Path("/storage" if Path("/storage").is_dir() else "./storage")
    log_level: str = "INFO"

    # Modal workspace username (visible in `modal app list` output). Used to
    # construct shared-inference endpoint URLs without shelling out. Set via
    # MODAL_WORKSPACE in .env.
    modal_workspace: str = "swager2014"
    # Name of the deployed shared inference app — must match the `modal.App`
    # name in app/deploy/inference_app.py.
    modal_inference_app: str = "modelforge-inference"
    # Modal Volume that holds per-run model bundles. Created on first upload.
    modal_models_volume: str = "modelforge-models"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
if Path("/storage").is_dir():
    settings.storage_dir = Path("/storage")
settings.storage_dir.mkdir(parents=True, exist_ok=True)
