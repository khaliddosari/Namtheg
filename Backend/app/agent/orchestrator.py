"""Runs one AutoML job end to end.

Division of labour:
- Deterministic, tested code computes every number: profiling, detection, the
  data audit, time-series preparation and naive baselines, the imbalance plan,
  and feature engineering here; cross-validation, tuning and final fits on the
  GPU service, in parallel.
- The LLM investigates tabular data (the analyst agent, running its code in
  the sandbox) and writes the summary. Both are validated in code first.

Three flows share the ending (plot, grounded report, downloads):
tables (classification, regression, clustering), forecasting, and images.
"""
import logging
from concurrent.futures import Future, ThreadPoolExecutor

from app import storage
from app.agent import analyst, report
from app.llm import LLMClient
from app.pipeline import (
    audit, detect, eda, export, feature_engineering, imbalance, profile, text_features, timeseries, train, visualize,
)
from app.sandbox import open_sandbox

log = logging.getLogger(__name__)

# The champion must beat the feature-blind baseline by at least this much (in
# the selection metric's units) to count as having learned something.
BASELINE_MARGIN = 0.01


def _stage(run_id: str, stage: str) -> None:
    storage.write_status(run_id, "running", stage=stage)
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


def beats_baseline(metrics: dict) -> bool | None:
    base = (metrics["extra"].get("baseline") or {}).get("cv_mean")
    if base is None:
        return None
    score = float(metrics["score"])
    return score < base - BASELINE_MARGIN if metrics.get("higher_is_better") is False else score > base + BASELINE_MARGIN


def _facts(task, target, metrics, beats, *, fe_report=None, audit_report=None, analysis=None,
           imbalance_plan=None) -> dict:
    """Every number the written summary is allowed to cite."""
    extra = metrics["extra"]
    critical, warnings = [], []
    if audit_report:
        critical = [{"column": f["column"], "detail": f["detail"]}
                    for f in audit_report["findings"] if f["severity"] == "critical"]
        warnings = [f["detail"] for f in audit_report["findings"] if f["severity"] == "warning"][:5]
    plan = (analysis or {}).get("plan") or {}
    critical += [{"column": None, "detail": f["finding"]}
                 for f in plan.get("findings", []) if f["severity"] == "critical" and f.get("verified")]
    facts = {
        "task": task,
        "target": target,
        "model": {
            "name": metrics["model_name"],
            "metric": metrics["score_metric"],
            "higher_is_better": metrics.get("higher_is_better", True),
            "score": metrics["score"],
            "test_score": extra.get("test_score"),
            "train_score": extra.get("train_score"),
            "overfit_gap": extra.get("overfit_gap"),
        },
        "other_test_metrics": extra.get("test_metrics") or extra.get("backtest_metrics") or {},
        "baseline": extra.get("baseline"),
        "beats_baseline": beats,
        "candidates": extra.get("all_models", []),
        "trained_on": (extra.get("hardware") or {}).get("gpu"),
        "critical_findings": critical,
        "warnings": warnings,
        "analysis_summary": plan.get("summary") if plan.get("summary_verified") else None,
        "dropped_by_analysis": [d["column"] for d in plan.get("drop_columns", [])],
        "open_questions": plan.get("open_questions", []),
    }
    if fe_report:
        facts.update(rows_used=fe_report.get("final_row_count"), features_used=fe_report.get("final_feature_count"))
    if imbalance_plan and imbalance_plan.get("detected"):
        facts["class_imbalance"] = {k: imbalance_plan[k] for k in
                                    ("ratio", "minority_class", "minority_share", "strategy", "selection_metric")}
    if task == "clustering":
        facts["clusters"] = extra.get("cluster_metrics")
        facts["silhouette_guide"] = report.silhouette_guide(metrics["score"])
    if task == "forecasting":
        facts["forecast"] = extra.get("timeseries")
        facts["baselines"] = extra.get("baselines")
    return facts


def _finish(run_id, task, target, problem_type, metrics, facts, names, llm, downloads, extra_updates) -> dict:
    _stage(run_id, "report")
    justification, grounding = report.write_justification(llm, facts, names)
    extra = metrics["extra"]
    extra.update({**extra_updates, "beats_baseline": facts["beats_baseline"], "grounding": grounding,
                  "llm": llm.summary()})
    final = {
        "run_id": run_id,
        "status": "succeeded",
        "task": task,
        "target": target,
        "problem_type": problem_type,
        "accuracy_score": metrics["score"],
        "score_metric": metrics["score_metric"],
        "higher_is_better": metrics.get("higher_is_better", True),
        "plot_path": str(storage.run_dir(run_id) / "plot.png"),
        "justification": justification,
        "model_name": metrics["model_name"],
        "extra": extra,
    }
    _stage(run_id, "package")
    export.build_model_package(run_id, final)
    final["downloads"] = downloads
    return final


def _tables(run_id: str, spec: dict, llm: LLMClient) -> dict:
    """Classification, regression, or clustering (no target)."""
    target = spec.get("target")
    clustering = spec["task"] == "clustering"
    executor, sandbox_future = None, None
    if llm.configured:
        # Start the sandbox now so its cold start overlaps profiling and the audit.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sandbox")
        sandbox_future = executor.submit(open_sandbox, storage.dataset_path(run_id))
    try:
        _stage(run_id, "profile")
        prof = profile.profile_dataset(run_id)

        _stage(run_id, "detect")
        if clustering:
            detection = {"problem_type": "clustering", "reason": "No target column was chosen.", "target": None}
        else:
            detection = detect.detect_problem_type(run_id, target)
            if spec["task"] in ("classification", "regression"):
                detection = {**detection, "problem_type": spec["task"], "reason": "Chosen by the user."}

        _stage(run_id, "audit")
        audit_report = audit.audit_dataset(run_id, target, detection["problem_type"])

        _stage(run_id, "analysis")
        analysis = analyst.run_analyst(run_id, target, prof, detection, audit_report, llm, sandbox_future)
    finally:
        _close_when_ready(sandbox_future)
        if executor is not None:
            executor.shutdown(wait=False)
    storage.write_json(run_id, "analysis.json", {k: v for k, v in analysis.items() if k != "cells"})
    storage.write_json(run_id, "agent_log.json", {"cells": analysis["cells"], "notes": analysis["notes"]})
    plan = analysis["plan"]
    if spec["task"] != "auto":
        # The user chose the task; the analyst's other decisions still apply.
        plan["problem_type"], plan["problem_type_source"] = detection["problem_type"], "user"
    problem_type = plan["problem_type"]
    if problem_type != detection["problem_type"]:
        audit_report = audit.audit_dataset(run_id, target, problem_type)

    if not clustering:
        _stage(run_id, "eda")
        eda.run_eda(run_id, target)

    _stage(run_id, "feature_engineering")
    dropped = [d["column"] for d in plan["drop_columns"]]
    text_cols = list(text_features.detect_text_columns(storage.load_dataset(run_id),
                                                       exclude=[c for c in [target] + dropped if c]))
    fe_report = feature_engineering.feature_engineer(run_id, target, extra_drops=plan["drop_columns"],
                                                     text_columns=text_cols)
    text_cols = fe_report["text_columns"]

    imbalance_plan = None
    _stage(run_id, "train")
    if clustering:
        metrics = train.train_clustering(run_id, text_cols)
    else:
        # Decided on exactly the rows that will be trained on, before any model is fit.
        imbalance_plan = imbalance.assess(storage.load_engineered(run_id)[target], problem_type)
        storage.write_json(run_id, "imbalance.json", imbalance_plan)
        metrics = train.train_tabular(run_id, target, problem_type, imbalance_plan, text_cols)
    export.export_cleaned_csv(run_id, problem_type)

    _stage(run_id, "visualize")
    visualize.generate_visualization(run_id, target, problem_type)

    beats = beats_baseline(metrics)
    facts = _facts(problem_type, target, metrics, beats, fe_report=fe_report, audit_report=audit_report,
                   analysis=analysis, imbalance_plan=imbalance_plan)
    names = [c["name"] for c in prof["columns"]] + [m["name"] for m in facts["candidates"]] + [c for c in [target] if c]
    return _finish(run_id, problem_type, target, problem_type, metrics, facts, names, llm,
                   ["cleaned_csv", "model"], {
                       "imbalance": imbalance_plan,
                       "audit": audit_report,
                       "analysis": {k: analysis.get(k) for k in ("status", "notes", "code_runs", "seconds")} | {
                           "problem_type_source": plan["problem_type_source"],
                           "summary": plan["summary"],
                           "findings": plan["findings"],
                           "drop_columns": plan["drop_columns"],
                           "open_questions": plan["open_questions"],
                       },
                   })


def _forecasting(run_id: str, spec: dict, llm: LLMClient) -> dict:
    target = spec["target"]
    _stage(run_id, "prepare")
    ts = timeseries.prepare(run_id, spec["date_column"], target, spec.get("series_id_column"), spec.get("horizon"))
    _stage(run_id, "train")
    metrics = train.train_forecasting(run_id, ts)
    export.export_cleaned_csv(run_id, "forecasting")
    _stage(run_id, "visualize")
    visualize.generate_visualization(run_id, target, "forecasting")
    facts = _facts("forecasting", target, metrics, beats_baseline(metrics))
    names = [target, spec["date_column"]] + [m["name"] for m in facts["candidates"]]
    return _finish(run_id, "forecasting", target, "forecasting", metrics, facts, names, llm,
                   ["cleaned_csv", "forecast", "model"], {
                       "timeseries_preparation": ts,
                       "analysis": {"status": "skipped", "notes": [
                           "Forecasting runs use the deterministic time-series checks; the analyst agent "
                           "reviews tables."]},
                   })


def _images(run_id: str, spec: dict, llm: LLMClient) -> dict:
    manifest = storage.load_dataset(run_id)
    _stage(run_id, "audit")
    imbalance_plan = imbalance.assess(manifest["label"].astype(str), "classification")
    storage.write_json(run_id, "imbalance.json", imbalance_plan)
    _stage(run_id, "train")
    metrics = train.train_images(run_id, imbalance_plan)
    export.export_cleaned_csv(run_id, "image_classification")
    _stage(run_id, "visualize")
    visualize.generate_visualization(run_id, "label", "classification")
    facts = _facts("image_classification", "label", metrics, beats_baseline(metrics), imbalance_plan=imbalance_plan)
    names = [str(c) for c in manifest["label"].unique()] + [m["name"] for m in facts["candidates"]]
    return _finish(run_id, "image_classification", "label", "classification", metrics, facts, names, llm,
                   ["cleaned_csv", "model"], {
                       "imbalance": imbalance_plan,
                       "analysis": {"status": "skipped", "notes": [
                           "Image runs are checked deterministically at upload; the analyst agent reviews "
                           "tables."]},
                   })


def run_agent(run_id: str, spec: dict | str) -> dict:
    """spec: from app.pipeline.tasks.resolve. A bare string is a target column
    (task "auto"), as older callers pass."""
    if isinstance(spec, str):
        spec = {"task": "auto", "target": spec}
    storage.write_status(run_id, "running", task=spec["task"], target=spec.get("target"), stage="starting")
    llm = LLMClient()
    runner = {"forecasting": _forecasting, "image_classification": _images}.get(spec["task"], _tables)
    try:
        final = runner(run_id, spec, llm)
        storage.write_json(run_id, "result.json", final)
        storage.write_status(run_id, "succeeded", stage="done")
        log.info("Run %s succeeded.", run_id)
        return final
    except Exception as e:
        log.exception("Agent run failed for %s", run_id)
        err = {"run_id": run_id, "status": "failed", "task": spec["task"], "target": spec.get("target"),
               "error": str(e)}
        storage.write_json(run_id, "result.json", err)
        storage.write_status(run_id, "failed", error=str(e))
        return err
