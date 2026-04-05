# Publication-quality plotting for TPC_RP active learning experiments.
# Targets IEEE single-column format (3.5in width, serif font, size 9).

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from sklearn.manifold import TSNE


# Global style 

STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.dpi": 150,
}

# Colorblind-friendly palette
COLORS = {
    "TypiClust":  "#2196F3",
    "Random":     "#9E9E9E",
    "Modified":   "#FF9800",
}

# Fallback order for methods not in the palette
_EXTRA_COLORS = ["#4CAF50", "#E91E63", "#9C27B0", "#00BCD4", "#795548"]

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (7.0, 2.5)


def _color_for(method: str, idx: int = 0) -> str:
    if method in COLORS:
        return COLORS[method]
    return _EXTRA_COLORS[idx % len(_EXTRA_COLORS)]


def _save(fig: plt.Figure, save_path: str | Path) -> None:
    # Save as both PDF (for LaTeX) and PNG (for preview).
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".pdf"))
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {save_path.with_suffix('.pdf')}  /  {save_path.with_suffix('.png')}")



#  1. Accuracy vs Budget


def plot_accuracy_vs_budget(
    results_dict: dict[str, dict],
    save_path: str | Path = "plots/accuracy_vs_budget",
) -> None:
    # results_dict: {method: {"labelled_counts": [...], "accuracies": [...],
    #                         optional "std": [...]}}
    # "accuracies" can be mean values; "std" provides shaded error band.
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=SINGLE_COL)

        extra_idx = 0
        for method, data in results_dict.items():
            color = _color_for(method, extra_idx)
            if method not in COLORS:
                extra_idx += 1

            x = np.asarray(data["labelled_counts"])
            y = np.asarray(data["accuracies"]) * 100  # convert to %

            ax.plot(x, y, marker="o", label=method, color=color)

            if "std" in data:
                se = np.asarray(data["std"]) * 100
                ax.fill_between(x, y - se, y + se, alpha=0.18, color=color)

        ax.set_xlabel("Cumulative Budget")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Active Learning: Accuracy vs. Label Budget")
        ax.legend(loc="lower right")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        _save(fig, save_path)



#  2. Comparison Bar Chart


def plot_comparison_bars(
    results_dict: dict[str, dict],
    budget_level: int,
    save_path: str | Path = "plots/comparison_bars",
) -> None:
    # Compare methods at a specific budget level.
    # results_dict: {method: {"labelled_counts": [...], "accuracies": [...],
    #                         optional "std": [...]}}
    with plt.rc_context(STYLE):
        methods, accs, errs = [], [], []
        extra_idx = 0

        for method, data in results_dict.items():
            counts = np.asarray(data["labelled_counts"])

            # Find the round closest to budget_level
            match = np.argmin(np.abs(counts - budget_level))
            acc = data["accuracies"][match] * 100
            se = data["std"][match] * 100 if "std" in data else 0.0

            methods.append(method)
            accs.append(acc)
            errs.append(se)

        colors = [_color_for(m, i) for i, m in enumerate(methods)]

        fig, ax = plt.subplots(figsize=SINGLE_COL)
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, accs, yerr=errs, capsize=4,
                      color=colors, edgecolor="black", linewidth=0.5, width=0.55)

        # Value labels on top of each bar
        for bar, acc, err in zip(bars, accs, errs):
            y_top = bar.get_height() + err + 0.5
            ax.text(bar.get_x() + bar.get_width() / 2, y_top,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(f"Method Comparison at Budget = {budget_level}")
        ax.set_ylim(bottom=0)

        _save(fig, save_path)



#  3. t-SNE Visualisation of Selection


def plot_tsne_selection(
    features: np.ndarray,
    selected_indices: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path = "plots/tsne_selection",
    perplexity: float = 30.0,
    seed: int = 42,
) -> None:
    # Compute 2D t-SNE embedding
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                init="pca", learning_rate="auto")
    emb = tsne.fit_transform(features)

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Use a perceptually distinct colourmap
    cmap = plt.cm.get_cmap("tab10", max(n_classes, 10))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=SINGLE_COL)

        # Background: all points, coloured by class (light)
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[cmap(i)], alpha=0.25, s=6, label=f"Class {lab}")

        # Overlay: selected points (bold X markers)
        sel = np.asarray(selected_indices)
        ax.scatter(emb[sel, 0], emb[sel, 1],
                   c="black", marker="X", s=40, linewidths=0.5,
                   edgecolors="white", zorder=5, label="Selected")

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE: Feature Space & Selected Samples")
        ax.legend(loc="best", markerscale=1.5, ncol=2, framealpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])

        _save(fig, save_path)



#  4. Training Curves (Loss + Accuracy)


def plot_training_curves(
    train_history: dict,
    save_path: str | Path = "plots/training_curves",
) -> None:
    # train_history: {"train_loss": [...], "train_acc": [...]}
    # train_acc values are in [0, 1].
    losses = np.asarray(train_history["train_loss"])
    accs   = np.asarray(train_history["train_acc"]) * 100
    epochs = np.arange(1, len(losses) + 1)

    with plt.rc_context(STYLE):
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=DOUBLE_COL)

        # Left: loss
        ax_loss.plot(epochs, losses, color="#D32F2F", marker="o", markersize=2)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Training Loss")
        ax_loss.set_title("Loss")
        ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Right: accuracy
        ax_acc.plot(epochs, accs, color="#1976D2", marker="o", markersize=2)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Training Accuracy (%)")
        ax_acc.set_title("Accuracy")
        ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.suptitle("Classifier Training Curves", fontsize=10, y=1.02)
        fig.tight_layout()

        _save(fig, save_path)
