from app.agent.report import template_justification, ungrounded_numbers, write_justification
from app.llm import AssistantTurn

FACTS = {
    "task": "regression",
    "model": {"name": "RandomForest", "metric": "r2", "score": 0.8734, "test_score": 0.8612,
              "higher_is_better": True},
    "baseline": {"name": "Mean of target", "cv_mean": -0.0021},
    "rows_used": 1338,
    "critical_findings": [{"detail": "This column alone predicts the target with cross-validated R2 0.9912."}],
}


def test_rounded_and_percent_forms_are_grounded():
    text = "RandomForest scored 0.87 (87.3%) on 1,338 rows, vs 0.86 held out; one feature alone reached 0.991."
    assert ungrounded_numbers(text, FACTS) == []


def test_invented_numbers_are_caught():
    assert ungrounded_numbers("It reaches 0.95 accuracy and saves 40% of costs.", FACTS) == ["0.95", "40%"]


def test_small_integers_and_names_with_digits_are_ignored():
    text = "Across 5 folds, q2023_sales mattered most."
    assert ungrounded_numbers(text, FACTS, names=["q2023_sales"]) == []


class _FakeLLM:
    configured = True

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, transcript, tools=None, tool_choice=None, effort=None):
        self.prompts.append(transcript[-1]["content"])
        return AssistantTurn(text=self.replies.pop(0), tool_calls=[], provider="fake", model="fake")


def test_hallucination_is_retried_then_replaced_by_template():
    llm = _FakeLLM(["Scores 0.95.", "Still 0.95."])
    text, grounding = write_justification(llm, FACTS, [])
    assert grounding["source"] == "template" and grounding["rejected_numbers"] == ["0.95"]
    assert "0.95" in llm.prompts[1]  # the retry named the offending number
    assert ungrounded_numbers(text, FACTS) == []


def test_retry_that_fixes_the_number_is_accepted():
    llm = _FakeLLM(["Scores 0.95.", "RandomForest scores 0.873 in cross-validation."])
    text, grounding = write_justification(llm, FACTS, [])
    assert grounding == {"source": "llm", "model": "fake", "attempts": 2, "verified": True}


def test_template_is_always_grounded_and_flags_weak_models():
    weak = {**FACTS, "beats_baseline": False}
    text = template_justification(weak)
    assert "does not meaningfully beat" in text
    assert ungrounded_numbers(text, weak) == []


def test_template_states_direction_for_error_metrics():
    facts = {"task": "forecasting", "model": {"name": "TCN", "metric": "mase", "score": 1.08,
                                              "higher_is_better": False},
             "baseline": {"name": "Seasonal naive", "cv_mean": 0.9}, "beats_baseline": False}
    text = template_justification(facts)
    assert "rolling-backtest mase of 1.0800" in text and "lower is better" in text
    assert "does not meaningfully beat" in text and ungrounded_numbers(text, facts) == []


def test_clustering_template_never_claims_verified_segments():
    from app.agent.report import silhouette_guide

    facts = {"task": "clustering", "model": {"name": "K-Means", "metric": "silhouette", "score": 0.31},
             "silhouette_guide": silhouette_guide(0.31)}
    text = template_justification(facts)
    assert "weak cluster structure" in text and "not verified segments" in text
    assert ungrounded_numbers(text, facts) == []
