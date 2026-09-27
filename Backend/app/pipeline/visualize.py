import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def generate_visualization(run_id: str, target: str, problem_type: str) -> dict:
    y_test = np.load(storage.run_dir(run_id) / "y_test.npy", allow_pickle=True)
    y_pred = np.load(storage.run_dir(run_id) / "y_pred.npy", allow_pickle=True)

    # PERF: figsize 5.5x4.5 + dpi 100 below ≈ same on-screen size as the old
    # 6x5 @ 120 but with ~30% fewer pixels to rasterize and encode.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    if problem_type == "classification":
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, colorbar=False, cmap=BRAND_CMAP
        )
        _recolor_cm_text(disp)
        ax.set_title(f"Confusion Matrix - target: {target}")
        plot_kind = "confusion_matrix"
    else:
        ax.scatter(y_test, y_pred, alpha=0.75, color=ACCENT, edgecolor=ACCENT_DARK, linewidth=0.4)
        lo = float(min(np.min(y_test), np.min(y_pred)))
        hi = float(max(np.max(y_test), np.max(y_pred)))
        ax.plot([lo, hi], [lo, hi], color=ACCENT, linestyle="--", linewidth=1.2)
        ax.grid(True, color=MUTED, linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"Predicted vs Actual - target: {target}")
        plot_kind = "predicted_vs_actual"

    _style_dark(fig, ax)
    fig.tight_layout()
    out = storage.artifact_path(run_id, "plot.png")
    fig.savefig(out, dpi=100, transparent=True)
    plt.close(fig)
    storage.persist(run_id, "plot.png")

    info = {"plot_kind": plot_kind, "plot_path": str(out)}
    storage.write_json(run_id, "visualization.json", info)
    return info
