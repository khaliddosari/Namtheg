"""Large tables: one working sample shared by the analyst and training. The
analyst's understanding and cleaning: checked against the data, applied to
training, and applied again by the model to every raw input at prediction."""
import json
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest

from app import storage
from app.agent import analyst, orchestrator
from app.main import model_schema
from app.pipeline import sampling
from app.training.runtime import build_new_column, load_bundle, new_column_sources
from tests.conftest import INSURANCE_CSV, ScriptedLLM, make_run


def _use_llm(monkeypatch, llm):
    monkeypatch.setattr(orchestrator, "LLMClient", lambda: llm)


def _submission(**overrides) -> dict:
    return {
        "data_dictionary": {"row_meaning": "One insured person.", "target_meaning": "Yearly medical charges.",
                            "columns": [{"column": "bmi", "meaning": "Body mass index.", "basis": "both"}]},
        "problem_type": "regression", "problem_type_rationale": "charges is an amount.",
        "drop_columns": [], "findings": [], "open_questions": [], "summary": "Smokers are charged more.",
        **overrides,
    }


# -- sampling ---------------------------------------------------------------------------------

def test_large_tables_get_one_reproducible_stratified_sample(monkeypatch):
    monkeypatch.setattr(sampling, "SAMPLE_ROWS", 500)
    df = pd.read_csv(INSURANCE_CSV)
    df.loc[:9, "smoker"] = None  # rows without a target are never sampled
    run_id = make_run(df)
    data = storage.load_dataset(run_id)

    first, info = sampling.working_set(run_id, data, "smoker")
    again, _ = sampling.working_set(run_id, data, "smoker")
    assert len(first) == 500 and first["smoker"].notna().all()
    pd.testing.assert_frame_equal(first, again)
    assert info == {"rows_total": 1338, "rows_used": 500, "sampled": True, "stratified": True, "rows_with_target": 1328}
    share = (data["smoker"] == "yes").sum() / data["smoker"].notna().sum()
    assert abs((first["smoker"] == "yes").mean() - share) < 0.01  # every class keeps its share
    pd.testing.assert_frame_equal(pd.read_parquet(storage.artifact_path(run_id, sampling.SAMPLE_FILE)), first)

    whole, info = sampling.working_set(run_id, data.head(400), "smoker")
    assert len(whole) == 400 and info["sampled"] is False


def test_the_analyst_and_training_share_the_sample(monkeypatch):
    monkeypatch.setattr(sampling, "SAMPLE_ROWS", 600)
    llm = ScriptedLLM([[("run_python", {"code": "print(len(df))"})], [("submit_analysis", _submission())]])
    _use_llm(monkeypatch, llm)
    run_id = make_run(pd.read_csv(INSURANCE_CSV))

    result = orchestrator.run_agent(run_id, "charges")
    assert result["status"] == "succeeded", result.get("error")
    assert llm.analyst_transcripts[1][-1]["content"].strip() == "600"  # the sandbox held the sample
    context = json.loads(llm.analyst_transcripts[0][0]["content"].split("\n\n", 1)[1])
    assert context["working_data"]["rows_used"] == 600 and context["profile"]["n_rows"] == 1338
    assert storage.read_json(run_id, "feature_engineering.json")["final_row_count"] == 600
    assert len(pd.read_csv(storage.artifact_path(run_id, "cleaned.csv"), encoding="utf-8-sig")) == 600
    assert result["extra"]["sample"] == {"rows_total": 1338, "rows_used": 600, "sampled": True,
                                         "stratified": False, "rows_with_target": 1338}
    assert result["extra"]["analysis"]["data_dictionary"]["row_meaning"] == "One insured person."


def test_the_report_says_when_a_sample_was_used(monkeypatch):
    monkeypatch.setattr(sampling, "SAMPLE_ROWS", 600)
    result = orchestrator.run_agent(make_run(pd.read_csv(INSURANCE_CSV)), "charges")  # no LLM: the template
    assert "random sample of 600 of the 1,338 rows" in result["justification"]


# -- the analyst's cleaning ---------------------------------------------------------------------

def messy_frame(n=400) -> pd.DataFrame:
    """Store codes that are categories, -999 meaning missing, "?" plans, dates
    whose weekday matters, and one class spelled three ways."""
    rng = np.random.default_rng(0)
    store = rng.choice([101, 205, 333, 470], size=n)
    visit = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, size=n), unit="D")
    churned = (store == 333) | (pd.DatetimeIndex(visit).dayofweek >= 5)
    income = rng.normal(50_000, 8_000, size=n).round()
    income[rng.choice(n, 40, replace=False)] = -999
    return pd.DataFrame({
        "store": store,
        "income": income,
        "plan": rng.choice(["basic", "pro", "?"], size=n),
        "visit": visit.strftime("%Y-%m-%d"),
        "churn": np.where(churned, rng.choice(["yes", "Yes", "Y"], size=n), "no"),
    })


CLEANING = _submission(
    data_dictionary={
        "row_meaning": "One customer visit.",
        "target_meaning": "Whether the customer churned.",
        "columns": [{"column": "store", "meaning": "The store, as a code.", "basis": "data"},
                    {"column": "ghost", "meaning": "Not a column.", "basis": "data"}],
        "relationships": [{"finding": "Store 333 churns most.", "evidence": "churn rate by store"}],
    },
    problem_type="classification", problem_type_rationale="churn is yes or no.",
    missing_value_codes=[{"column": "income", "values": [-999, 12345], "evidence": "-999 is not an income."},
                         {"column": "plan", "values": ["?"], "evidence": "? marks no plan."},
                         {"column": "churn", "values": ["no"], "evidence": "x"}],
    label_merges=[{"labels": ["yes", "Y"], "into": "Yes", "evidence": "three spellings of one class"}],
    categorical_columns=[{"column": "store", "evidence": "four store codes"},
                         {"column": "plan", "evidence": "x"}],
    date_features=[{"column": "visit", "parts": ["year", "month", "day_of_week"], "evidence": "visit dates"},
                   {"column": "plan", "parts": ["month"], "evidence": "x"}],
)


def test_cleaning_is_checked_against_the_data():
    work = messy_frame()
    stats = {"numeric": False, "unique": 4, "rows": len(work)}
    detection = {"problem_type": "classification"}
    plan, notes = analyst.validate_analysis(CLEANING, "churn", detection, list(work.columns), stats, work)

    assert analyst.cleaning_spec(plan) == {
        "missing_codes": {"income": [-999], "plan": ["?"]},
        "label_merges": [{"labels": ["Y", "yes"], "into": "Yes"}],
        "categorical": ["store"],
        "date_parts": {"visit": ["month", "day_of_week"]},  # the year is the same in every row
        "new_columns": [],  # none asked for
    }
    assert [c["column"] for c in plan["data_dictionary"]["columns"]] == ["store"]
    assert any("[12345]" in n and "not in the data" in n for n in notes)
    assert any("target column" in n for n in notes)
    assert any("'plan' as categories" in n for n in notes)
    assert any("['year']" in n for n in notes)
    assert any("date features for 'plan'" in n for n in notes)

    # Merging every label away, or into a label that doesn't exist, is refused.
    bad, notes = analyst.validate_analysis(
        _submission(problem_type="classification",
                    label_merges=[{"labels": ["yes", "Y", "Yes"], "into": "no", "evidence": "x"},
                                  {"labels": ["no"], "into": "maybe", "evidence": "x"}]),
        "churn", detection, list(work.columns), stats, work)
    assert bad["cleaning"]["label_merges"] == []
    assert any("fewer than 2 classes" in n for n in notes) and any("'maybe'" in n for n in notes)


def test_cleaning_reaches_training_and_every_prediction(tmp_path, monkeypatch):
    _use_llm(monkeypatch, ScriptedLLM([[("submit_analysis", CLEANING)]]))
    df = messy_frame()
    run_id = make_run(df)
    result = orchestrator.run_agent(run_id, "churn")
    assert result["status"] == "succeeded", result.get("error")
    assert len(result["extra"]["cleaning_applied"]) == 5

    engineered = storage.load_engineered(run_id)
    assert "visit" not in engineered and {"visit__month", "visit__day_of_week"} <= set(engineered.columns)
    assert engineered["income"].isna().sum() == 40 and set(engineered["churn"]) == {"Yes", "no"}
    assert set(engineered["store"]) == {"101", "205", "333", "470"}

    # Predictions take the raw columns; the model cleans them the way training did.
    meta = storage.read_json(run_id, "model_meta.json")
    assert {"visit", "store", "income", "plan"} == set(meta["feature_cols"])
    assert "visit__month" in meta["model_cols"] and meta["cleaning"]["categorical"] == ["store"]
    bundle = load_bundle(joblib.load(storage.artifact_path(run_id, "model.joblib")))
    model, raw = bundle["model"], df.drop(columns=["churn"]).head(25)
    expected = model.estimator.predict(model.preprocessor.transform(engineered[model.model_cols].head(25)))
    assert np.array_equal(np.asarray(model.predict(raw)).ravel(), np.asarray(expected).ravel())
    assert bundle["class_labels"] == ["Yes", "no"]

    zipfile.ZipFile(storage.artifact_path(run_id, "model_package.zip")).extractall(tmp_path / "pkg")
    raw.to_csv(tmp_path / "in.csv", index=False)
    subprocess.run([sys.executable, "predict.py", str(tmp_path / "in.csv"), str(tmp_path / "out.csv")],
                   cwd=tmp_path / "pkg", check=True, capture_output=True)
    assert set(pd.read_csv(tmp_path / "out.csv")["prediction"]) <= {"Yes", "no"}

    schema = model_schema(run_id)  # the "Try it" form asks for raw values
    assert set(schema["feature_cols"]) == set(meta["feature_cols"]) and schema["sample"]["visit"] == df["visit"][0]


# -- a new column the user asked for ----------------------------------------------------------

def test_a_new_column_formula_is_parsed_never_run():
    X = pd.DataFrame({"age": [20, 40, 0], "bmi": [25.0, 35.0, 40.0], "smoker": ["yes", "yes ", None],
                      "total charges": ["100", "200", "x"]})

    def build(formula):
        return build_new_column(X, {"name": "new", "formula": formula}).tolist()

    assert build("bmi / age")[:2] == [1.25, 0.875] and np.isnan(build("bmi / age")[2])  # 40 / 0 is missing
    assert build('smoker == "yes" and bmi >= 30')[:2] == [0.0, 1.0] and np.isnan(build('smoker == "yes"')[2])
    assert build("`total charges` / 100")[:2] == [1.0, 2.0] and np.isnan(build("`total charges` / 100")[2])
    assert new_column_sources({"formula": "`total charges` * 2 + log(bmi) - age"}) == ["total charges", "bmi", "age"]
    bands = {"name": "band", "bins": {"column": "bmi", "edges": [30.0], "labels": ["under 30", "30 and over"]}}
    assert build_new_column(X, bands).tolist() == ["under 30", "30 and over", "30 and over"]
    for formula in ["__import__('os').system('echo hi')", "age.real", "[age][0]", "(lambda: 1)()",
                    "age if bmi else 0", "1 < age < 3", "open('x')", "eval('1')"]:
        with pytest.raises(ValueError):
            new_column_sources({"formula": formula})


def test_new_columns_are_checked_against_the_data():
    work = pd.read_csv(INSURANCE_CSV)
    stats = {"numeric": True, "unique": int(work["charges"].nunique()), "rows": len(work)}
    detection = {"problem_type": "regression"}

    def check(*new_columns, drops=()):
        submission = _submission(new_columns=list(new_columns),
                                 drop_columns=[{"column": c, "reason": "other", "evidence": "x"} for c in drops])
        return analyst.validate_analysis(submission, "charges", detection, list(work.columns), stats, work)

    bands = {"column": "bmi", "edges": [18.5, 25, 30], "labels": ["under", "healthy", "over", "obese"]}
    plan, notes = check({"name": "bmi band", "bins": bands, "description": "x"},
                        {"name": "smoker_obese", "formula": 'smoker == "yes" and bmi >= 30', "description": "x"})
    assert analyst.cleaning_spec(plan)["new_columns"] == [
        {"name": "bmi band", "bins": {**bands, "edges": [18.5, 25.0, 30.0]}},
        {"name": "smoker_obese", "formula": 'smoker == "yes" and bmi >= 30'},
    ] and not notes

    refused = [
        ({"name": "per_year", "formula": "charges / age"}, "reads the target"),
        ({"name": "southeast", "formula": 'region == "southeast"'}, "doesn't have"),
        ({"name": "bmi", "formula": "bmi * 2"}, "name is taken"),
        ({"name": "both", "formula": "bmi", "bins": {"column": "bmi", "edges": [30], "labels": ["a", "b"]}},
         "one of the two"),
        ({"name": "old", "formula": "age > 1000"}, "same in every row"),
        ({"name": "nothing", "formula": "1 + 2"}, "no column"),
        ({"name": "backwards", "bins": {"column": "bmi", "edges": [30, 25], "labels": ["a", "b", "c"]}}, "increasing"),
        ({"name": "few", "bins": {"column": "bmi", "edges": [25, 30], "labels": ["a", "b"]}}, "3 in all"),
        ({"name": "code", "formula": "__import__('os')"}, "may only use"),
    ]
    for item, reason in refused:
        plan, notes = check(item, drops=["region"])
        assert plan["cleaning"]["new_columns"] == [] and any(reason in n for n in notes), (item, notes)


def test_a_new_column_is_only_offered_when_asked_for():
    assert "new_columns" not in analyst.submit_analysis_tool(None).parameters["properties"]
    assert "new_columns" in analyst.submit_analysis_tool(None, "BMI bands").parameters["properties"]
    assert "new_columns" in analyst.SUBMIT_ANALYSIS.parameters["properties"]  # the shared spec is left as it was


def test_a_new_column_reaches_training_and_every_prediction(tmp_path, monkeypatch):
    bands = {"name": "bmi_band", "bins": {"column": "bmi", "edges": [18.5, 25, 30],
                                          "labels": ["under", "healthy", "over", "obese"]}, "description": "BMI band."}
    llm = ScriptedLLM([[("submit_analysis", _submission(new_columns=[bands]))]])
    _use_llm(monkeypatch, llm)
    df = pd.read_csv(INSURANCE_CSV)
    run_id = make_run(df)
    asked = "BMI bands: under 18.5, 18.5 to 25, 25 to 30, 30 and over"
    result = orchestrator.run_agent(run_id, {"task": "auto", "target": "charges", "new_column": asked})
    assert result["status"] == "succeeded", result.get("error")
    assert f'"new_column": "{asked}"' in llm.analyst_transcripts[0][0]["content"]
    assert result["extra"]["analysis"]["new_column"] == asked
    assert any(c.startswith("Added the column bmi_band you asked for") for c in result["extra"]["cleaning_applied"])

    engineered = storage.load_engineered(run_id)
    assert set(engineered["bmi_band"]) == {"under", "healthy", "over", "obese"}
    meta = storage.read_json(run_id, "model_meta.json")
    assert "bmi_band" not in meta["feature_cols"] and "bmi" in meta["feature_cols"]  # predictions take raw columns
    assert "bmi_band" in meta["model_cols"] and meta["cleaning"]["new_columns"][0]["name"] == "bmi_band"
    model = load_bundle(joblib.load(storage.artifact_path(run_id, "model.joblib")))["model"]
    raw = df.drop(columns=["charges"]).head(25)
    expected = model.estimator.predict(model.preprocessor.transform(engineered[model.model_cols].head(25)))
    assert np.allclose(np.asarray(model.predict(raw)).ravel(), np.asarray(expected).ravel())

    zipfile.ZipFile(storage.artifact_path(run_id, "model_package.zip")).extractall(tmp_path / "pkg")
    raw.to_csv(tmp_path / "in.csv", index=False)
    subprocess.run([sys.executable, "predict.py", str(tmp_path / "in.csv"), str(tmp_path / "out.csv")],
                   cwd=tmp_path / "pkg", check=True, capture_output=True)
    assert len(pd.read_csv(tmp_path / "out.csv")) == 25

    # The "Try it" form lists each category column's values.
    assert model_schema(run_id)["feature_values"] == {
        "sex": ["female", "male"], "smoker": ["no", "yes"],
        "region": ["northeast", "northwest", "southeast", "southwest"],
    }


def test_a_new_column_that_isnt_built_says_why(monkeypatch):
    llm = ScriptedLLM([[("submit_analysis", _submission(new_columns=[{"name": "x", "formula": "charges * 2"}]))]])
    _use_llm(monkeypatch, llm)
    result = orchestrator.run_agent(make_run(pd.read_csv(INSURANCE_CSV)),
                                    {"task": "auto", "target": "charges", "new_column": "twice the charges"})
    assert result["status"] == "succeeded", result.get("error")
    finding = next(f for f in result["extra"]["analysis"]["findings"] if "wasn't added" in f["finding"])
    assert "reads the target" in finding["evidence"]
    assert not any("Added the column" in c for c in result["extra"]["cleaning_applied"])
