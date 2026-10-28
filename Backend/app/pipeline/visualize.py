"""Every graph and table a finished run can show, drawn once after training
and kept with the run for later.

Each graph is a PNG at graphs/<key>.png, transparent so it sits on the
dashboard's dark glass. Each table is in the manifest (graphs.json), which
the result carries, with a CSV beside it. graphs.zip holds all of them for
download, numbered in the dashboard's order, the graphs laid on the
dashboard's dark background so they read in any image viewer. plot.png stays
a copy of the first graph for older clients.

They come from the held-out test split the trainer saved (y_test.npy,
y_pred.npy, and y_score.npy with each row's class probabilities, when the
model gives them) and from metrics.json, so runs from before this module
drew more than one plot can be drawn again (see ensure_graphs).
"""
import csv
import io
import shutil
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    max_error,
    median_absolute_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from app import storage

GRAPHS_DIR = "graphs"
MANIFEST = "graphs.json"
GRAPHS_ZIP = "graphs.zip"
DPI = 150
BACKGROUND = "#0b1220"  # what the downloaded graphs are laid on
MAX_SCATTER_POINTS = 5_000
MAX_CURVE_CLASSES = 10  # ROC and precision-recall draw one line per class, up to this many
MAX_BARS = 15
MAX_MATRIX_VALUES = 15  # classes past which a confusion matrix drops its numbers

# Liquid Glass palette - dark, transparent plots that float on the frosted-glass
# surfaces of the frontend (cyan -> blue accent, off-white text).
ACCENT = "#4fc3f7"       # cyan
ACCENT_DARK = "#0288d1"  # deep ocean blue
TEXT = "#e8e8ed"         # off-white (primary text)
MUTED = "#9999a8"        # cool grey (labels, ticks)
GRID = "#2a2a35"         # faint hairline grid / spines
SERIES = ["#4fc3f7", "#9fa8da", "#ce93d8", "#80cbc4", "#ffcc80", "#f48fb1", "#a5d6a7", "#90caf9", "#ffab91", "#e6ee9c"]
# Low counts fade into the dark page; high counts glow cyan.
BRAND_CMAP = LinearSegmentedColormap.from_list("namtheg_cyan", ["#0a1620", "#4fc3f7"])

METRIC_NAMES = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "f1_macro": "Macro F1",
    "roc_auc": "ROC AUC", "pr_auc": "Average precision", "r2": "R²", "rmse": "RMSE", "mae": "MAE",
}


# -- drawing -------------------------------------------------------------------------

def _recolor_cm_text(disp) -> None:
    """Force legible confusion-matrix counts: dark text on bright (cyan) cells,
    off-white text on dark cells, independent of the colormap's auto threshold."""
    cm = disp.confusion_matrix
    vmax = cm.max() or 1
    if disp.text_ is None:
        return
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            t = disp.text_[i, j]
            if t is not None:
                t.set_color("#0a1620" if cm[i, j] >= 0.5 * vmax else "#e8e8ed")


def _style_dark(fig, ax) -> None:
    """Make the figure transparent and recolor text/ticks/spines for the dark UI."""
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_alpha(0.0)
        legend.get_frame().set_edgecolor(GRID)
        for text in legend.get_texts():
            text.set_color(TEXT)


def _grid(ax) -> None:
    ax.grid(True, color=MUTED, linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def _sample(*arrays):
    """At most MAX_SCATTER_POINTS rows, the same random rows from each array."""
    n = len(arrays[0])
    if n <= MAX_SCATTER_POINTS:
        return arrays
    keep = np.random.default_rng(0).choice(n, MAX_SCATTER_POINTS, replace=False)
    return tuple(a[keep] for a in arrays)


def _confusion(y_test, y_pred, labels, target, normalized: bool):
    n = len(labels)
    size = min(12.0, max(5.5, 0.4 * n + 2))
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=labels, ax=ax, colorbar=False, cmap=BRAND_CMAP,
        normalize="true" if normalized else None, values_format=".0%" if normalized else "d",
        include_values=n <= MAX_MATRIX_VALUES,
    )
    _recolor_cm_text(disp)
    if n > 8:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{'Share of each actual class' if normalized else 'Confusion matrix'}, {target}")
    return fig, ax


def _positive_scores(scores, labels):
    """The score for the second class of a two-class target, from probabilities or a margin."""
    return scores if scores.ndim == 1 else scores[:, 1]


def _roc(y_test, scores, labels, target):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if len(labels) == 2:
        fpr, tpr, _ = roc_curve(y_test == labels[1], _positive_scores(scores, labels))
        ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"{labels[1]} (AUC {auc(fpr, tpr):.3f})")
    else:
        for i, label in enumerate(labels[:MAX_CURVE_CLASSES]):
            truth = y_test == label
            if truth.all() or not truth.any():
                continue
            fpr, tpr, _ = roc_curve(truth, scores[:, i])
            ax.plot(fpr, tpr, color=SERIES[i % len(SERIES)], linewidth=1.6, label=f"{label} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    _grid(ax)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve, {target}")
    ax.legend(loc="lower right", fontsize=8)
    return fig, ax


def _precision_recall(y_test, scores, labels, target):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if len(labels) == 2:
        truth, score = y_test == labels[1], _positive_scores(scores, labels)
        precision, recall, _ = precision_recall_curve(truth, score)
        ax.plot(recall, precision, color=ACCENT, linewidth=2,
                label=f"{labels[1]} (AP {average_precision_score(truth, score):.3f})")
        ax.axhline(truth.mean(), color=MUTED, linestyle="--", linewidth=1)
    else:
        for i, label in enumerate(labels[:MAX_CURVE_CLASSES]):
            truth = y_test == label
            if truth.all() or not truth.any():
                continue
            precision, recall, _ = precision_recall_curve(truth, scores[:, i])
            ax.plot(recall, precision, color=SERIES[i % len(SERIES)], linewidth=1.6,
                    label=f"{label} (AP {average_precision_score(truth, scores[:, i]):.3f})")
    _grid(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision and recall, {target}")
    ax.legend(loc="lower left", fontsize=8)
    return fig, ax


def _class_counts(y_test, y_pred, labels, target):
    actual = [int((y_test == c).sum()) for c in labels]
    predicted = [int((y_pred == c).sum()) for c in labels]
    shown = sorted(range(len(labels)), key=lambda i: -actual[i])[:MAX_BARS]
    x = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=(max(5.5, 0.5 * len(shown) + 2), 4.5))
    ax.bar(x - 0.2, [actual[i] for i in shown], width=0.4, color=ACCENT, label="Actual")
    ax.bar(x + 0.2, [predicted[i] for i in shown], width=0.4, color=SERIES[1], label="Predicted")
    ax.set_xticks(x, [str(labels[i]) for i in shown], rotation=45 if len(shown) > 6 else 0,
                  ha="right" if len(shown) > 6 else "center")
    _grid(ax)
    ax.set_ylabel("Rows in the test set")
    ax.set_title(f"Actual and predicted classes, {target}")
    ax.legend(fontsize=8)
    return fig, ax


def _predicted_vs_actual(y_test, y_pred, target):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    actual, predicted = _sample(y_test, y_pred)
    ax.scatter(actual, predicted, s=14, alpha=0.75, color=ACCENT, edgecolor=ACCENT_DARK, linewidth=0.4)
    lo = float(min(np.min(y_test), np.min(y_pred)))
    hi = float(max(np.max(y_test), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color=ACCENT, linestyle="--", linewidth=1.2)
    _grid(ax)
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"Predicted vs actual, {target}")
    return fig, ax


def _residuals(y_test, y_pred, target):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    predicted, residual = _sample(y_pred, y_test - y_pred)
    ax.scatter(predicted, residual, s=14, alpha=0.75, color=ACCENT, edgecolor=ACCENT_DARK, linewidth=0.4)
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1.2)
    _grid(ax)
    ax.set_xlabel(f"Predicted {target}")
    ax.set_ylabel("Actual minus predicted")
    ax.set_title(f"Residuals, {target}")
    return fig, ax


def _error_distribution(y_test, y_pred, target):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(y_test - y_pred, bins=40, color=ACCENT, edgecolor=ACCENT_DARK, linewidth=0.4)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1.2)
    _grid(ax)
    ax.set_xlabel("Actual minus predicted")
    ax.set_ylabel("Rows in the test set")
    ax.set_title(f"Error distribution, {target}")
    return fig, ax


def _feature_importance(features):
    top = features[:MAX_BARS][::-1]
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.32 * len(top) + 1.2)))
    ax.barh([str(f["feature"]) for f in top], [f["importance"] for f in top], color=ACCENT)
    _grid(ax)
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance")
    return fig, ax


def _model_comparison(models, champion, metric):
    shown = models[:MAX_BARS][::-1]
    fig, ax = plt.subplots(figsize=(5.5, max(3.0, 0.38 * len(shown) + 1.2)))
    ax.barh([m["name"] for m in shown], [m["cv_mean"] for m in shown], xerr=[m["cv_std"] for m in shown],
            color=[ACCENT if m["name"] == champion else SERIES[1] for m in shown], ecolor=MUTED, capsize=3)
    _grid(ax)
    ax.set_xlabel(f"Cross-validated {METRIC_NAMES.get(metric, metric)}")
    ax.set_title("Model comparison")
    return fig, ax


def _tuning(trials, metric, lower_is_better: bool):
    scored = [t for t in trials if t.get("score") is not None]
    x = [t["trial"] for t in scored]
    y = [t["score"] for t in scored]
    best = (np.minimum if lower_is_better else np.maximum).accumulate(y)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x, y, s=22, color=SERIES[1], label="Trial", zorder=3)
    ax.step(x, best, where="post", color=ACCENT, linewidth=2, label="Best so far")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _grid(ax)
    ax.set_xlabel("Trial (0 is the default settings)")
    ax.set_ylabel(f"Cross-validated {METRIC_NAMES.get(metric, metric)}")
    ax.set_title("Tuning progress")
    ax.legend(fontsize=8)
    return fig, ax


# -- tables --------------------------------------------------------------------------

def _class_metrics(y_test, y_pred, labels) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
    rows = [[str(c), round(float(p), 4), round(float(r), 4), round(float(f), 4), int(s)]
            for c, p, r, f, s in zip(labels, precision, recall, f1, support)]
    total = int(support.sum()) or 1
    rows.append(["Macro average", round(float(precision.mean()), 4), round(float(recall.mean()), 4),
                 round(float(f1.mean()), 4), int(support.sum())])
    rows.append(["Weighted average", *(round(float((v * support).sum() / total), 4) for v in (precision, recall, f1)),
                 int(support.sum())])
    return {"columns": ["Class", "Precision", "Recall", "F1", "Rows"], "rows": rows}


def _test_metrics(metrics: dict, y_test, y_pred, regression: bool) -> dict:
    tested = (metrics.get("extra") or {}).get("test_metrics") or {}
    rows = [[METRIC_NAMES.get(k, k), round(float(v), 4)] for k, v in tested.items() if isinstance(v, (int, float))]
    if regression:
        errors = np.abs(y_test - y_pred)
        rows += [["Median absolute error", round(float(median_absolute_error(y_test, y_pred)), 4)],
                 ["Largest error", round(float(max_error(y_test, y_pred)), 4)],
                 ["Rows within 10% of actual", round(float(np.mean(errors <= 0.1 * np.abs(y_test))), 4)]]
    rows.append(["Rows in the test set", int(len(y_test))])
    return {"columns": ["Metric", "Value"], "rows": rows}


# -- the set ---------------------------------------------------------------------------

def _load(run_id: str, name: str):
    path = storage.artifact_path(run_id, name)
    return np.load(path, allow_pickle=True) if path.exists() else None


def _items(run_id: str, target: str, problem_type: str) -> list[tuple[dict, object]]:
    """(manifest entry, figure or table) for every graph and table this run's data allows, in order."""
    y_test, y_pred = _load(run_id, "y_test.npy"), _load(run_id, "y_pred.npy")
    scores = _load(run_id, "y_score.npy")
    metrics = storage.read_json(run_id, "metrics.json") or {}
    extra = metrics.get("extra") or {}
    items: list[tuple[dict, object]] = []

    def graph(key, title, caption, draw):
        items.append(({"key": key, "title": title, "caption": caption, "kind": "image"}, draw))

    def table(key, title, caption, data):
        items.append(({"key": key, "title": title, "caption": caption, "kind": "table", **data}, None))

    if problem_type == "classification":
        meta = storage.read_json(run_id, "model_meta.json") or {}
        labels = [str(c) for c in meta.get("class_labels") or sorted({*map(str, y_test), *map(str, y_pred)})]
        y_test, y_pred = y_test.astype(str), y_pred.astype(str)
        graph("confusion_matrix", "Confusion matrix", "Diagonal cells are correct predictions.",
              lambda: _confusion(y_test, y_pred, labels, target, normalized=False))
        graph("confusion_matrix_share", "Confusion matrix, by actual class",
              "Each row adds up to 100%: where the rows of each actual class were predicted.",
              lambda: _confusion(y_test, y_pred, labels, target, normalized=True))
        table("class_metrics", "Per-class metrics", "Precision, recall and F1 of each class on held-out data.",
              _class_metrics(y_test, y_pred, labels))
        if scores is not None and len(scores) == len(y_test) and (scores.ndim == 1 or scores.shape[1] == len(labels)):
            graph("roc_curve", "ROC curve", "True against false positives at every threshold. AUC is the area under it.",
                  lambda: _roc(y_test, scores, labels, target))
            graph("precision_recall", "Precision and recall",
                  "Precision against recall at every threshold. The dashed line is a guess with no model.",
                  lambda: _precision_recall(y_test, scores, labels, target))
        graph("class_counts", "Actual and predicted classes", "How often each class is in the test set, and predicted.",
              lambda: _class_counts(y_test, y_pred, labels, target))
    else:
        y_test, y_pred = y_test.astype(float), y_pred.astype(float)
        graph("predicted_vs_actual", "Predicted vs actual", "Points close to the diagonal are accurate predictions.",
              lambda: _predicted_vs_actual(y_test, y_pred, target))
        graph("residuals", "Residuals", "Each prediction's error against its value. An even band around zero is good.",
              lambda: _residuals(y_test, y_pred, target))
        graph("error_distribution", "Error distribution", "How far predictions land from the actual value.",
              lambda: _error_distribution(y_test, y_pred, target))

    table("test_metrics", "Held-out metrics", "Every score on the test set the models never trained on.",
          _test_metrics(metrics, y_test, y_pred, problem_type != "classification"))
    features = extra.get("top_features") or []
    if features:
        graph("feature_importance", "Feature importance", "The columns that drive the prediction most.",
              lambda: _feature_importance(features))
    models = extra.get("all_models") or []
    if len(models) > 1:
        graph("model_comparison", "Model comparison", "Each candidate's cross-validated score, with its spread.",
              lambda: _model_comparison(models, metrics.get("model_name"), metrics.get("score_metric")))
    trials = [t for t in extra.get("tuning_trials") or [] if t.get("score") is not None]
    if len(trials) > 1:
        graph("tuning", "Tuning progress", "Each tuning trial's score, and the best so far.",
              lambda: _tuning(trials, metrics.get("score_metric"), metrics.get("higher_is_better") is False))
    return items


def generate_visualization(run_id: str, target: str, problem_type: str) -> dict:
    """Draw and save every graph and table, then zip them. Returns the manifest under "graphs"."""
    folder = storage.run_dir(run_id) / GRAPHS_DIR
    folder.mkdir(exist_ok=True)
    manifest = []
    for entry, content in _items(run_id, target, problem_type):
        name = f"{GRAPHS_DIR}/{entry['key']}.{'png' if entry['kind'] == 'image' else 'csv'}"
        if entry["kind"] == "image":
            fig, ax = content()
            _style_dark(fig, ax)
            fig.tight_layout()
            fig.savefig(storage.run_dir(run_id) / name, dpi=DPI, transparent=True)
            plt.close(fig)
        else:
            with open(storage.run_dir(run_id) / name, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(entry["columns"])
                writer.writerows(entry["rows"])
        storage.persist(run_id, name)
        manifest.append(entry)

    first = next(e for e in manifest if e["kind"] == "image")
    shutil.copyfile(folder / f"{first['key']}.png", storage.run_dir(run_id) / "plot.png")
    storage.persist(run_id, "plot.png")
    storage.write_json(run_id, MANIFEST, manifest)
    _zip(run_id, manifest)
    info = {"plot_kind": first["key"], "plot_path": str(storage.run_dir(run_id) / "plot.png"), "graphs": manifest}
    storage.write_json(run_id, "visualization.json", {k: v for k, v in info.items() if k != "graphs"})
    return info


def _zip(run_id: str, manifest: list[dict]) -> None:
    """graphs.zip: each graph on the dashboard's dark background, each table as CSV, numbered in order."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, entry in enumerate(manifest, 1):
            stem = f"{i:02d}-{entry['key']}"
            if entry["kind"] == "image":
                image = Image.open(storage.artifact_path(run_id, f"{GRAPHS_DIR}/{entry['key']}.png")).convert("RGBA")
                flat = Image.alpha_composite(Image.new("RGBA", image.size, BACKGROUND), image).convert("RGB")
                png = io.BytesIO()
                flat.save(png, "PNG", optimize=True)
                z.writestr(f"{stem}.png", png.getvalue())
            else:
                z.write(storage.artifact_path(run_id, f"{GRAPHS_DIR}/{entry['key']}.csv"), f"{stem}.csv")
    (storage.run_dir(run_id) / GRAPHS_ZIP).write_bytes(buf.getvalue())
    storage.persist(run_id, GRAPHS_ZIP)


def ensure_graphs(run_id: str, result: dict) -> dict:
    """A finished run's result with its graphs. Runs from before every graph
    was drawn get them now, from the test split they saved, once; a run with
    nothing to draw them from is returned as it was."""
    extra = result.get("extra") or {}
    if result.get("status") != "succeeded" or "graphs" in extra or _load(run_id, "y_test.npy") is None:
        return result
    info = generate_visualization(run_id, result.get("target") or "target", result.get("problem_type") or "regression")
    result = {**result, "extra": {**extra, "graphs": info["graphs"]},
              "downloads": [*dict.fromkeys([*(result.get("downloads") or []), "graphs"])]}
    storage.write_json(run_id, "result.json", result)
    return result
