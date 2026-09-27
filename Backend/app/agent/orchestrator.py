"""Runs one AutoML job end to end.

Division of labour:
- Deterministic, tested code computes every number: profile, problem-type
  detection, data audit, imbalance plan, and feature engineering here; the
  baseline, cross-validation, tuning, and final fit on the GPU service.
- The LLM investigates the data (the analyst agent, running its code in the
  sandbox) and writes the summary. Both outputs are validated in code before
  the pipeline uses them.
"""
import logging
from concurrent.futures import Future, ThreadPoolExecutor

from app import storage
from app.agent import analyst, report
from app.llm import LLMClient
from app.pipeline import audit, detect, eda, export, feature_engineering, imbalance, profile, train, visualize
from app.sandbox import open_sandbox

log = logging.getLogger(__name__)

# The champion must beat the feature-blind baseline's CV score by at least this
# much (in the selection metric) to count as having learned something.
BASELINE_MARGIN = 0.01


def _stage(run_id: str, target: str, stage: str) -> None:
    storage.write_status(run_id, "running", target=target, stage=stage)
    log.info("Run %s: %s", run_id, stage)


def _close_when_ready(future: Future | None) -> None:
    """Close the sandbox as soon as it exists, without blocking on its startup."""
    if future is None:
        return

    def _close(f: Future) -> None:
        try:
            sandbox = f.result()
        except Exception:
            return
        if sandbox is not None:
            sandbox.close()

    future.add_done_callback(_close)


def _facts(target, problem_type, fe_report, metrics, beats_baseline, imbalance_plan, audit_report, analysis) -> dict:
    """Every number the written summary is allowed to cite."""
    extra = metrics["extra"]
    plan = analysis["plan"]
    critical = [{"column": f["column"], "detail": f["detail"]}
                for f in audit_report["findings"] if f["severity"] == "critical"]
    critical += [{"column": None, "detail": f["finding"]}
                 for f in plan["findings"] if f["severity"] == "critical" and f.get("verified")]
    return {
        "target": target,
        "problem_type": problem_type,
        "rows_used": fe_report.get("final_row_count"),
        "features_used": fe_report.get("final_feature_count"),
        "model": {
            "name": metrics["model_name"],
            "metric": metrics["score_metric"],
            "cv_score": metrics["score"],
            "test_score": extra.get("test_score"),
            "train_score": extra.get("train_score"),
            "overfit_gap": extra.get("overfit_gap"),
        },
        "other_test_metrics": extra.get("test_metrics", {}),
        "baseline": extra["baseline"],
        "beats_baseline": beats_baseline,
        "candidates": extra.get("all_models", []),
        "class_imbalance": (
            {k: imbalance_plan[k] for k in ("ratio", "minority_class", "minority_share", "strategy", "selection_metric")}
            if imbalance_plan.get("detected") else None
        ),
        "trained_on": (extra.get("hardware") or {}).get("gpu"),
        "critical_findings": critical,
        "warnings": [f["detail"] for f in audit_report["findings"] if f["severity"] == "warning"][:5],
        "analysis_summary": plan["summary"] if plan.get("summary_verified") else None,
        "dropped_by_analysis": [d["column"] for d in plan["drop_columns"]],
        "open_questions": plan["open_questions"],
    }


def run_agent(run_id: str, target: str) -> dict:
    storage.write_status(run_id, "running", target=target, stage="starting")
    llm = LLMClient()
    executor: ThreadPoolExecutor | None = None
    sandbox_future: Future | None = None
    if llm.configured:
        # Start the sandbox now so its cold start overlaps profiling and the audit.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sandbox")
        sandbox_future = executor.submit(open_sandbox, storage.dataset_path(run_id))
    try:
        _stage(run_id, target, "profile")
        prof = profile.profile_dataset(run_id)

        _stage(run_id, target, "detect")
        detection = detect.detect_problem_type(run_id, target)

        _stage(run_id, target, "audit")
        audit_report = audit.audit_dataset(run_id, target, detection["problem_type"])

        _stage(run_id, target, "analysis")
        analysis = analyst.run_analyst(run_id, target, prof, detection, audit_report, llm, sandbox_future)
        _close_when_ready(sandbox_future)
        sandbox_future = None
        storage.write_json(run_id, "analysis.json", {k: v for k, v in analysis.items() if k != "cells"})
        storage.write_json(run_id, "agent_log.json", {"cells": analysis["cells"], "notes": analysis["notes"]})
        plan = analysis["plan"]
        problem_type = plan["problem_type"]
        if problem_type != detection["problem_type"]:
            audit_report = audit.audit_dataset(run_id, target, problem_type)

        _stage(run_id, target, "eda")
        eda.run_eda(run_id, target)

        _stage(run_id, target, "feature_engineering")
        fe_report = feature_engineering.feature_engineer(run_id, target, extra_drops=plan["drop_columns"])
        export.export_cleaned_csv(run_id)

        # Decided on exactly the rows that will be trained on, before any model is fit.
        imbalance_plan = imbalance.assess(storage.load_engineered(run_id)[target], problem_type)
        storage.write_json(run_id, "imbalance.json", imbalance_plan)

        _stage(run_id, target, "train")
        metrics = train.train_model(run_id, target, problem_type, imbalance_plan)
        extra = metrics["extra"]
        beats_baseline = float(metrics["score"]) > extra["baseline"]["cv_mean"] + BASELINE_MARGIN

        _stage(run_id, target, "visualize")
        viz_info = visualize.generate_visualization(run_id, target, problem_type)

        _stage(run_id, target, "report")
        facts = _facts(target, problem_type, fe_report, metrics, beats_baseline, imbalance_plan, audit_report, analysis)
        names = [c["name"] for c in prof["columns"]] + [m["name"] for m in facts["candidates"]] + [target]
        justification, grounding = report.write_justification(llm, facts, names)

        extra.update({
            "beats_baseline": beats_baseline,
            "imbalance": imbalance_plan,
            "audit": audit_report,
            "analysis": {k: analysis.get(k) for k in ("status", "notes", "code_runs", "seconds")} | {
                "problem_type_source": plan["problem_type_source"],
                "summary": plan["summary"],
                "findings": plan["findings"],
                "drop_columns": plan["drop_columns"],
                "open_questions": plan["open_questions"],
            },
            "grounding": grounding,
            "llm": llm.summary(),
        })
        final = {
            "run_id": run_id,
            "status": "succeeded",
            "target": target,
            "problem_type": problem_type,
            "accuracy_score": metrics["score"],
            "score_metric": metrics["score_metric"],
            "plot_path": viz_info.get("plot_path", ""),
            "justification": justification,
            "model_name": metrics["model_name"],
            "extra": extra,
        }
        _stage(run_id, target, "package")
        export.build_model_package(run_id, final)
        final["downloads"] = ["cleaned_csv", "model"]
        storage.write_json(run_id, "result.json", final)
        storage.write_status(run_id, "succeeded", stage="done")
        log.info("Run %s succeeded.", run_id)
        return final
    except Exception as e:
        log.exception("Agent run failed for %s", run_id)
        err = {
            "run_id": run_id,
            "status": "failed",
            "target": target,
            "error": str(e),
        }
        storage.write_json(run_id, "result.json", err)
        storage.write_status(run_id, "failed", error=str(e))
        return err
    finally:
        _close_when_ready(sandbox_future)
        if executor is not None:
            executor.shutdown(wait=False)
