"""
Utility functions: data loading, seeding, feature extraction, plotting, logging.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm

from .simclr import SimCLRModel


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cifar10(root: str = "./data") -> tuple:
    """
    Download and return raw CIFAR-10 train/test datasets (PIL images, no transform).

    Returns:
        (train_dataset, test_dataset) — torchvision CIFAR10 objects.
    """
    # No transform here; individual modules apply their own transforms.
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=None)
    return train, test


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    dataset,
    encoder: SimCLRModel,
    batch_size: int = 512,
    device: torch.device | None = None,
    num_workers: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract L2-normalised 512-dim backbone features for every sample in `dataset`.

    Delegates to `simclr.get_features`; kept here as a convenience import for
    notebooks that do `from src.utils import extract_features`.

    Returns:
        features: np.ndarray (N, 512), L2-normalised, float32.
        labels:   np.ndarray (N,), int64.
    """
    from .simclr import get_features as _get_features
    return _get_features(encoder, dataset,
                         batch_size=batch_size, device=device,
                         num_workers=num_workers)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_accuracy_curve(
    labelled_counts: list[int],
    accuracies: list[float],
    label: str = "TPC_RP",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot test accuracy vs. number of labelled samples.

    Args:
        labelled_counts: X-axis values (number of labelled samples per round).
        accuracies:      Y-axis values (test accuracy in [0, 1]).
        label:           Legend label for this curve.
        save_path:       If provided, save the figure to this path.
        ax:              Existing Axes to draw on. Creates a new figure if None.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(labelled_counts, [a * 100 for a in accuracies], marker="o", label=label)
    ax.set_xlabel("Number of labelled samples")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Active Learning: Test Accuracy vs. Label Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")

    return ax


def plot_comparison(
    results: dict[str, dict],
    save_path: str | Path | None = None,
) -> None:
    """
    Overlay multiple active learning curves on one plot.

    Args:
        results:   Mapping of {method_name: {'labelled_counts': [...], 'accuracies': [...]}}.
        save_path: Optional path to save the figure.
    """
    _, ax = plt.subplots(figsize=(9, 6))
    for name, hist in results.items():
        plot_accuracy_curve(hist["labelled_counts"], hist["accuracies"], label=name, ax=ax)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Comparison plot saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(history: dict, path: str | Path) -> None:
    """Save an active learning history dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Results saved to {path}")


def load_results(path: str | Path) -> dict:
    """Load an active learning history dict from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str, level: str = "INFO") -> None:
    """Simple timestamped console logger."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")
