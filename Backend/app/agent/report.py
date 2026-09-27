"""Writes the run's justification and verifies it against computed facts.

Every number in LLM-written text must match a number the pipeline actually
computed (within the rounding the text used, and allowing percent form). Text
that cites anything else is regenerated once with the offending numbers named,
then replaced by a deterministic template if it still fails. The result records
which path produced the final text.
"""
import json
import logging
import re

from app.llm import LLMClient, LLMUnavailable

log = logging.getLogger(__name__)

# Integers this small are overwhelmingly structural ("5-fold", "top 3", "1-2
# sentences") rather than claims about the data.
_FREE_INTEGERS = set(range(0, 11))
_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")

JUSTIFICATION_SYSTEM_PROMPT = """You write the result summary for نَمذِج, an AutoML platform, read by
data analysts and business leads.

Write 2-3 sentences covering: which model won and how its score compares with the naive baseline
(when FACTS has one); what that score means in plain terms; and the single most important caveat
from the facts (a critical audit or analysis finding if there is one, otherwise the most relevant
limitation).

Rules:
- Use only numbers that appear in the FACTS JSON. Round them if you like, but never compute,
  estimate, or introduce new ones.
- Respect the metric's direction: model.higher_is_better is false for errors like MASE, where
  lower is better.
- If the model does not clearly beat the baseline, say so plainly.
- Clustering has no ground truth: describe the strength of the structure using silhouette_guide,
  never claim the clusters are correct or meaningful segments.
- Never call the model production-ready.
- Plain text only: no markdown, no lists. Never use an em dash or en dash; use commas or semicolons.
- Refer to the platform as نَمذِج, not as an agent or AI."""


def _collect_numbers(obj, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, str):
        # Numbers quoted inside computed strings (e.g. audit details) count too.
        for m in _NUMBER.finditer(obj):
            value = _parse_number(m.group())
            if value is not None:
                out.append(value[0])


def _parse_number(token: str) -> tuple[float, int, bool] | None:
    """-> (value, decimals written, was a percent)"""
    percent = token.endswith("%")
    raw = token.rstrip("%").replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return value, decimals, percent


def _strip_names(text: str, names: list[str]) -> str:
    # Column and model names often contain digits ("q3_sales", "col_2019").
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        text = text.replace(name, " ")
    return text


def ungrounded_numbers(text: str, facts: dict, names: list[str] | None = None) -> list[str]:
    """Numbers in `text` that no value in `facts` accounts for."""
    known: list[float] = []
    _collect_numbers(facts, known)
    candidates = set()
    for v in known:
        candidates.update((abs(v), abs(v) * 100))
    bad = []
    for m in _NUMBER.finditer(_strip_names(text, names or [])):
        parsed = _parse_number(m.group())
        if parsed is None:
            continue
        value, decimals, _ = parsed
        value = abs(value)
        if value.is_integer() and int(value) in _FREE_INTEGERS:
            continue
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(value - c) <= tolerance for c in candidates):
            bad.append(m.group())
    return bad


def _clean(text: str) -> str:
    return text.strip().replace("—", ", ").replace("–", "-")


EVALUATION = {
    "classification": "a cross-validated", "regression": "a cross-validated", "clustering": "a",
    "forecasting": "a rolling-backtest", "image_classification": "a validation",
}
# Kaufman & Rousseeuw's reading of the average silhouette width.
SILHOUETTE_LEVELS = ((0.7, "strong"), (0.5, "reasonable"), (0.25, "weak"))


def silhouette_guide(value: float) -> dict:
    level = next((name for cut, name in SILHOUETTE_LEVELS if value >= cut), "no substantial")
    return {"value": value, "structure": level, "thresholds": {"strong": 0.7, "reasonable": 0.5, "weak": 0.25}}


def template_justification(facts: dict) -> str:
    """Deterministic fallback: built only from facts, so it is always grounded."""
    m = facts["model"]
    task = facts.get("task", "classification")
    text = f"{m['name']} achieved {EVALUATION.get(task, 'a')} {m['metric']} of {m['score']:.4f}"
    if m.get("test_score") is not None and task != "clustering":
        text += f" and {m['test_score']:.4f} on the held-out test set"
    base = facts.get("baseline")
    if base and base.get("cv_mean") is not None:
        direction = " (lower is better)" if m.get("higher_is_better") is False else ""
        text += f", against {base['cv_mean']:.4f} for a naive baseline ({base['name'].lower()}){direction}"
    text += "."
    if base and facts.get("beats_baseline") is False:
        text += " The model does not meaningfully beat that baseline, so its predictions should not be relied on."
    if task == "clustering" and facts.get("silhouette_guide"):
        text += (f" That silhouette indicates {facts['silhouette_guide']['structure']} cluster structure; "
                 "there is no ground truth, so the clusters are a description of the data, not verified segments.")
    caveat = next((f["detail"] for f in facts.get("critical_findings", [])), None)
    if caveat:
        text += f" Caveat: {caveat}"
    return _clean(text)


def write_justification(llm: LLMClient, facts: dict, names: list[str]) -> tuple[str, dict]:
    """-> (justification, grounding record)"""
    if not llm.configured:
        return template_justification(facts), {"source": "template", "reason": "no LLM configured"}

    user = "FACTS:\n" + json.dumps(facts, indent=2, default=str)
    transcript = [{"role": "user", "content": user}]
    for attempt in (1, 2):
        try:
            turn = llm.complete(JUSTIFICATION_SYSTEM_PROMPT, transcript, effort="low")
        except LLMUnavailable as e:
            log.warning("Justification LLM call failed: %s", e)
            return template_justification(facts), {"source": "template", "reason": "LLM unavailable"}
        text = _clean(turn.text)
        bad = ungrounded_numbers(text, facts, names)
        if text and not bad:
            return text, {"source": "llm", "model": turn.model, "attempts": attempt, "verified": True}
        log.warning("Justification attempt %d cited unverifiable numbers: %s", attempt, bad)
        transcript += [
            {"role": "assistant", "turn": turn},
            {"role": "user", "content": (
                f"These numbers do not appear in FACTS: {', '.join(bad) or '(empty answer)'}. "
                "Rewrite using only numbers from FACTS."
            )},
        ]
    return template_justification(facts), {
        "source": "template",
        "reason": "LLM text cited numbers not in the computed facts",
        "rejected_numbers": bad,
    }
