"""Where runs execute.

On Modal (production) every run is a detached call to the `run_job` function
in app/deploy/backend_app.py: it survives the HTTP request, has its own
JOB_TIMEOUT_SECONDS budget, and can be cancelled. Locally (uvicorn), runs use
FastAPI background tasks in the API process; training still goes to the GPU
service either way.

This module is imported by the Modal deployment file, so it keeps imports light.
"""
import logging

log = logging.getLogger(__name__)

JOB_APP_NAME = "namtheg-backend"
JOB_FUNCTION_NAME = "run_job"
JOB_TIMEOUT_SECONDS = 6 * 3600


def on_modal() -> bool:
    import modal

    return not modal.is_local()


def start(run_id: str, spec: dict, background) -> dict:
    """Start a run. Returns fields to merge into the run status."""
    if on_modal():
        import modal

        call = modal.Function.from_name(JOB_APP_NAME, JOB_FUNCTION_NAME).spawn(run_id, spec)
        log.info("Run %s spawned as Modal call %s", run_id, call.object_id)
        return {"executor": "modal", "call_id": call.object_id}
    from app.agent.orchestrator import run_agent

    background.add_task(run_agent, run_id, spec)
    return {"executor": "local"}


def cancel(status: dict) -> bool:
    """Cancel a Modal job and the GPU calls it spawned (those are independent
    calls that would otherwise keep running, and billing). Local background
    tasks can't be cancelled."""
    if status.get("executor") != "modal" or not status.get("call_id"):
        return False
    import modal

    for call_id in [status["call_id"], *(status.get("gpu_calls") or [])]:
        try:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        except Exception as e:  # already finished, or never started
            log.warning("Could not cancel Modal call %s: %s", call_id, e)
    return True
