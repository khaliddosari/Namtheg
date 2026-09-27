import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay

from app import storage

# Liquid Glass palette - dark, transparent plots that float on the frosted-glass
# surfaces of the redesigned frontend (cyan -> blue accent, off-white text).
ACCENT = "#4fc3f7"       # cyan
ACCENT_DARK = "#0288d1"  # deep ocean blue
TEXT = "#e8e8ed"         # off-white (primary text)
MUTED = "#9999a8"        # cool grey (labels, ticks)
GRID = "#2a2a35"         # faint hairline grid / spines
# Low counts fade into the dark page; high counts glow cyan.
BRAND_CMAP = LinearSegmentedColormap.from_list(
    "namtheg_cyan", ["#0a1620", "#4fc3f7"]
)


def _recolor_cm_text(disp) -> None:
    """Force legible confusion-matrix counts: dark text on bright (cyan) cells,
    off-white text on dark cells — independent of the colormap's auto threshold."""
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


# Categorical palette for clusters, legible on the dark UI; noise is grey.
CLUSTER_COLORS = ["#4fc3f7", "#f48fb1", "#ffcc80", "#a5d6a7", "#ce93d8", "#80cbc4",
                  "#fff59d", "#ef9a9a", "#90caf9", "#bcaaa4", "#e6ee9c", "#b39ddb"]
NOISE_COLOR = "#5f5f6e"
HISTORY_POINTS = 120


def _supervised(ax, run_id: str, target: str, problem_type: str) -> str:
    y_test = np.load(storage.artifact_path(run_id, "y_test.npy"), allow_pickle=True)
    y_pred = np.load(storage.artifact_path(run_id, "y_pred.npy"), allow_pickle=True)
    if problem_type == "classification":
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, colorbar=False, cmap=BRAND_CMAP
        )
        _recolor_cm_text(disp)
        ax.set_title(f"Confusion Matrix - target: {target}")
        return "confusion_matrix"
    ax.scatter(y_test, y_pred, alpha=0.75, color=ACCENT, edgecolor=ACCENT_DARK, linewidth=0.4)
    lo = float(min(np.min(y_test), np.min(y_pred)))
    hi = float(max(np.max(y_test), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color=ACCENT, linestyle="--", linewidth=1.2)
    ax.grid(True, color=MUTED, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"Predicted vs Actual - target: {target}")
    return "predicted_vs_actual"


def _clusters(ax, run_id: str) -> str:
    proj = storage.read_json(run_id, "projection.json")
    x, y, labels = np.asarray(proj["x"]), np.asarray(proj["y"]), np.asarray(proj["labels"])
    for cid in np.unique(labels):
        mask = labels == cid
        color = NOISE_COLOR if cid < 0 else CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)]
        ax.scatter(x[mask], y[mask], s=10, alpha=0.8, color=color, linewidth=0,
                   label="noise" if cid < 0 else f"cluster {int(cid)}")
    ax.legend(loc="best", fontsize=7, frameon=False, labelcolor=TEXT, markerscale=1.6)
    ax.grid(True, color=MUTED, linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Clusters (2-D projection)")
    return "cluster_projection"


def _forecast(ax, run_id: str, target: str) -> str:
    bt = storage.read_json(run_id, "backtest.json")
    series = storage.load_engineered(run_id)
    sid = (bt.get("backtest") or {}).get("series_id") or series["series_id"].iloc[0]
    hist = series[series["series_id"] == sid].tail(HISTORY_POINTS)
    ax.plot(pd.to_datetime(hist["timestamp"]), hist["value"], color=MUTED, linewidth=1.2, label="history")
    if bt.get("backtest"):
        b = bt["backtest"]
        ax.plot(pd.to_datetime(b["timestamps"]), b["forecast"], color=ACCENT, linewidth=1.2, linestyle="--",
                label="backtest forecast")
    fc = pd.DataFrame(bt["forecast"])
    fc = fc[fc["series_id"] == sid]
    t = pd.to_datetime(fc["timestamp"])
    ax.fill_between(t, fc["lower"], fc["upper"], color=ACCENT, alpha=0.18, linewidth=0, label="80% interval")
    ax.plot(t, fc["forecast"], color=ACCENT, linewidth=1.8, label="forecast")
    ax.legend(loc="best", fontsize=7, frameon=False, labelcolor=TEXT)
    ax.grid(True, color=MUTED, linewidth=0.6, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel(target)
    ax.set_title(f"Forecast - {target}" + (f" ({sid})" if series["series_id"].nunique() > 1 else ""))
    ax.figure.autofmt_xdate()
    return "forecast"


def generate_visualization(run_id: str, target: str | None, problem_type: str) -> dict:
    # PERF: figsize 5.5x4.5 + dpi 100 below ≈ same on-screen size as the old
    # 6x5 @ 120 but with ~30% fewer pixels to rasterize and encode.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if problem_type == "clustering":
        plot_kind = _clusters(ax, run_id)
    elif problem_type == "forecasting":
        plot_kind = _forecast(ax, run_id, target)
    else:
        plot_kind = _supervised(ax, run_id, target, problem_type)

    _style_dark(fig, ax)
    fig.tight_layout()
    out = storage.run_dir(run_id) / "plot.png"
    fig.savefig(out, dpi=100, transparent=True)
    plt.close(fig)
    storage.persist(run_id, "plot.png")

    info = {"plot_kind": plot_kind, "plot_path": str(out)}
    storage.write_json(run_id, "visualization.json", info)
    return info
