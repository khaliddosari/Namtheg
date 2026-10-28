"""Every graph and table a finished run can show: drawn after training, kept
with the run, served one by one and as a zip, and drawn for older runs the
first time their result is read."""
import io
import shutil
import zipfile

import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image

from app import storage
from app.agent import orchestrator
from app.pipeline import visualize
from tests.conftest import INSURANCE_CSV, make_run


def _run(target: str) -> tuple[str, dict]:
    run_id = make_run(pd.read_csv(INSURANCE_CSV))
    result = orchestrator.run_agent(run_id, target)  # no LLM: the deterministic path
    assert result["status"] == "succeeded", result.get("error")
    return run_id, result


def _client():
    from app.main import app
    return TestClient(app)


def test_a_regression_run_draws_every_graph_and_keeps_them():
    run_id, result = _run("charges")
    graphs = result["extra"]["graphs"]
    keys = [g["key"] for g in graphs]
    assert keys[:4] == ["predicted_vs_actual", "residuals", "error_distribution", "test_metrics"]
    assert {"feature_importance", "model_comparison"} <= set(keys) and "graphs" in result["downloads"]
    for g in graphs:
        assert g["title"] and g["caption"] and g["kind"] in ("image", "table")
        if g["kind"] == "image":
            assert storage.artifact_path(run_id, f"graphs/{g['key']}.png").exists()
        else:
            assert g["columns"] and g["rows"] and storage.artifact_path(run_id, f"graphs/{g['key']}.csv").exists()
    metrics = dict(next(g for g in graphs if g["key"] == "test_metrics")["rows"])
    assert {"R²", "RMSE", "MAE", "Median absolute error", "Rows in the test set"} <= set(metrics)
    assert storage.read_json(run_id, visualize.MANIFEST) == graphs

    # One by one for the dashboard, transparent; all of them in a zip, on a dark background.
    client = _client()
    r = client.get(f"/runs/{run_id}/graphs/residuals")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).mode == "RGBA"
    assert client.get(f"/runs/{run_id}/graphs/nope").status_code == 404
    assert client.get(f"/runs/{run_id}/graphs/..%2Fresult").status_code == 404
    r = client.get(f"/runs/{run_id}/download/graphs")
    assert r.status_code == 200 and 'filename="data_graphs.zip"' in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = z.namelist()
        assert names == [f"{i:02d}-{g['key']}.{'png' if g['kind'] == 'image' else 'csv'}" for i, g in enumerate(graphs, 1)]
        assert Image.open(io.BytesIO(z.read(names[0]))).mode == "RGB"
        assert z.read("04-test_metrics.csv").decode("utf-8-sig").startswith("Metric,Value")
    assert client.get(f"/runs/{run_id}/plot").status_code == 200  # still the first graph, for older clients


def test_a_classification_run_draws_its_curves_and_class_table():
    run_id, result = _run("smoker")
    keys = [g["key"] for g in result["extra"]["graphs"]]
    assert keys[:6] == ["confusion_matrix", "confusion_matrix_share", "class_metrics", "roc_curve", "precision_recall",
                        "class_counts"]
    table = next(g for g in result["extra"]["graphs"] if g["key"] == "class_metrics")
    assert table["columns"] == ["Class", "Precision", "Recall", "F1", "Rows"]
    assert [row[0] for row in table["rows"]] == ["no", "yes", "Macro average", "Weighted average"]
    assert storage.artifact_path(run_id, "y_score.npy").exists()


def test_an_older_run_gets_its_graphs_when_its_result_is_read():
    run_id, result = _run("smoker")
    # As a run saved before every graph was drawn: one plot, no probabilities, no graphs in the result.
    shutil.rmtree(storage.run_dir(run_id) / visualize.GRAPHS_DIR)
    for name in (visualize.GRAPHS_ZIP, visualize.MANIFEST, "y_score.npy"):
        (storage.run_dir(run_id) / name).unlink()
    old = {**result, "downloads": ["cleaned_csv", "model"],
           "extra": {k: v for k, v in result["extra"].items() if k != "graphs"}}
    storage.write_json(run_id, "result.json", old)

    client = _client()
    upgraded = client.get(f"/runs/{run_id}/result").json()
    keys = [g["key"] for g in upgraded["extra"]["graphs"]]
    assert "confusion_matrix" in keys and "roc_curve" not in keys  # no probabilities were saved back then
    assert "graphs" in upgraded["downloads"] and storage.read_json(run_id, "result.json") == upgraded
    assert client.get(f"/runs/{run_id}/download/graphs").status_code == 200
