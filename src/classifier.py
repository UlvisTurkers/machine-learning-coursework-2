"""
Supervised classifier training and evaluation on CIFAR-10.

Two training paths
------------------
CIFARClassifier     — full end-to-end ResNet-18 trained from scratch on the
                      labeled subset.  Follows the paper's supervised training
                      spec (Appendix F.2.1): SGD + Nesterov, lr=0.025, cosine
                      schedule, 200 epochs, random crop + h-flip augmentation.
                      Re-initialises weights on every call to ``train()``.

train_linear_probe  — lightweight linear head on top of frozen SimCLR features.
                      Used for fast evaluation inside the active learning loop.

Reference:
    Hacohen, G., Dekel, O., & Weinshall, D. (2022).
    Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.
    ICML 2022.  Appendix F.2.1 — supervised training protocol.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CIFAR-10 transforms
# ---------------------------------------------------------------------------

_CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR_STD  = [0.2023, 0.1994, 0.2010]

CIFAR10_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD),
])

CIFAR10_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD),
])


# ---------------------------------------------------------------------------
# Dataset wrapper (PIL → tensor with transform)
# ---------------------------------------------------------------------------

class _TransformDataset(torch.utils.data.Dataset):
    """Applies a torchvision transform to a dataset that returns PIL images."""

    def __init__(self, dataset, transform):
        self.dataset   = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img), label


# ---------------------------------------------------------------------------
# CIFARClassifier
# ---------------------------------------------------------------------------

class CIFARClassifier:
    """
    ResNet-18 trained from scratch on a labeled CIFAR-10 subset.

    Training protocol (Appendix F.2.1 of Hacohen et al., 2022):
    - SGD, momentum=0.9, Nesterov=True, weight_decay=5e-4
    - Initial LR 0.025, cosine annealing to 0 over ``epochs``
    - Augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip
    - Weights are **re-initialised** at the start of every ``train()`` call
      so each active learning round starts from a fresh model

    Args:
        num_classes: Output classes (default 10).
        device:      Torch device string or object.  Defaults to CUDA if
                     available.
        seed:        Base random seed.  Passed to ``train()``; individual
                     ``train()`` calls can override with their own seed.
        num_workers: DataLoader worker count.  Set to 0 on Windows if you
                     encounter multiprocessing issues.
    """

    def __init__(
        self,
        num_classes: int = 10,
        device: str | torch.device | None = None,
        seed: int = 42,
        num_workers: int = 2,
    ):
        self.num_classes = num_classes
        self.device = (
            torch.device(device) if isinstance(device, str)
            else (device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )
        self.seed        = seed
        self.num_workers = num_workers

        # Build the model skeleton once; weights are re-init in train().
        self.model: nn.Module = self._build_model()

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """
        ResNet-18 with CIFAR-10 stem modifications:
          - conv1: 3×3, stride 1, padding 1  (preserves 32×32 spatial size)
          - maxpool replaced with Identity   (prevents over-downsampling)
          - fc: Linear(512, num_classes)
        """
        model = tv_models.resnet18(weights=None)
        model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc      = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def _reset_weights(self, seed: int) -> None:
        """Re-initialise all parameters to break correlation between AL rounds."""
        torch.manual_seed(seed)
        # Re-build from scratch to guarantee clean state (avoids partial resets).
        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_indices: np.ndarray,
        train_dataset,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 0.025,
        weight_decay: float = 5e-4,
        seed: int | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Train on the labeled subset identified by ``train_indices``.

        Weights are re-initialised before training so that every call
        produces an independently trained model.

        Args:
            train_indices:  Integer indices into ``train_dataset`` that are
                            currently labeled.
            train_dataset:  Full CIFAR-10 training dataset returning
                            (PIL Image, label) pairs.
            epochs:         Training epochs (paper: 200).
            batch_size:     Mini-batch size (paper uses 64 for low budgets).
            lr:             Initial learning rate (paper: 0.025).
            weight_decay:   SGD weight decay (paper: 5e-4).
            seed:           Random seed for weight re-init.  Defaults to
                            ``self.seed``.
            verbose:        Print per-epoch loss / accuracy.

        Returns:
            history: dict with keys:
                ``'train_loss'``   — list[float], one value per epoch
                ``'train_acc'``    — list[float], train accuracy ∈ [0, 1]
        """
        _seed = seed if seed is not None else self.seed
        self._reset_weights(_seed)

        train_indices = np.asarray(train_indices, dtype=np.int64)

        subset = Subset(
            _TransformDataset(train_dataset, CIFAR10_TRAIN_TRANSFORM),
            train_indices,
        )
        loader = DataLoader(
            subset,
            batch_size=min(batch_size, len(subset)),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        self.model = self.model.to(self.device)

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0
        )
        criterion = nn.CrossEntropyLoss()

        history: dict[str, list] = {"train_loss": [], "train_acc": []}

        epoch_bar = tqdm(
            range(1, epochs + 1),
            desc=f"Train (n={len(train_indices)})",
            leave=False,
            unit="ep",
        )
        for epoch in epoch_bar:
            self.model.train()
            running_loss = correct = total = 0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss   = criterion(logits, yb)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(yb)
                correct      += (logits.argmax(1) == yb).sum().item()
                total        += len(yb)

            scheduler.step()

            epoch_loss = running_loss / total
            epoch_acc  = correct / total
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.3f}")
            if verbose and (epoch % 50 == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"  [Classifier] epoch {epoch:>3d}/{epochs}  "
                    f"loss={epoch_loss:.4f}  train_acc={epoch_acc:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.5f}"
                )

        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_dataset,
        batch_size: int = 256,
    ) -> dict:
        """
        Evaluate the current model on the full test set.

        Args:
            test_dataset: CIFAR-10 test dataset (PIL Image, label).
            batch_size:   Inference batch size.

        Returns:
            results: dict with keys:
                ``'accuracy'``        — float, overall top-1 accuracy (%)
                ``'per_class_acc'``   — np.ndarray of shape (num_classes,),
                                        per-class accuracy (%)
                ``'predictions'``     — np.ndarray of shape (N,), predicted labels
                ``'targets'``         — np.ndarray of shape (N,), true labels
        """
        loader = DataLoader(
            _TransformDataset(test_dataset, CIFAR10_TEST_TRANSFORM),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        self.model = self.model.to(self.device).eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for xb, yb in tqdm(loader, desc="Evaluating", leave=False):
                preds = self.model(xb.to(self.device)).argmax(dim=1).cpu()
                all_preds.append(preds.numpy())
                all_targets.append(yb.numpy())

        preds   = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        overall_acc = float((preds == targets).mean() * 100)

        per_class = np.zeros(self.num_classes, dtype=np.float64)
        for c in range(self.num_classes):
            mask = targets == c
            per_class[c] = float((preds[mask] == targets[mask]).mean() * 100) if mask.any() else 0.0

        return {
            "accuracy":      overall_acc,
            "per_class_acc": per_class,
            "predictions":   preds,
            "targets":       targets,
        }

    # ------------------------------------------------------------------
    # Softmax probabilities (for uncertainty-based selectors)
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        dataset,
        indices: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Return softmax class probabilities for the samples at ``indices``.

        Designed for use with ``UncertaintySelection`` and ``MarginSelection``
        in ``active_learning.py``::

            def make_proba_fn(classifier, dataset):
                def fn(indices):
                    return classifier.predict_proba(dataset, indices)
                return fn

        Args:
            dataset:     Full training dataset (PIL Image, label).
            indices:     1-D integer array of pool indices to score.
            batch_size:  Inference batch size.

        Returns:
            probs: np.ndarray of shape (len(indices), num_classes), float32,
                   rows sum to 1.
        """
        indices = np.asarray(indices, dtype=np.int64)
        subset  = Subset(
            _TransformDataset(dataset, CIFAR10_TEST_TRANSFORM),
            indices,
        )
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        self.model = self.model.to(self.device).eval()
        all_probs = []
        with torch.no_grad():
            for xb, _ in loader:
                probs = F.softmax(self.model(xb.to(self.device)), dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience: train + evaluate in one call (for the AL loop)
    # ------------------------------------------------------------------

    def fit_and_evaluate(
        self,
        train_indices: np.ndarray,
        train_dataset,
        test_dataset,
        epochs: int = 200,
        batch_size: int = 64,
        seed: int | None = None,
        verbose: bool = True,
    ) -> tuple[dict, dict]:
        """
        Train on ``train_indices`` then evaluate on the full test set.

        Returns:
            (train_history, eval_results)  — dicts from ``train`` and
            ``evaluate`` respectively.
        """
        train_history = self.train(
            train_indices, train_dataset,
            epochs=epochs, batch_size=batch_size,
            seed=seed, verbose=verbose,
        )
        eval_results = self.evaluate(test_dataset)
        if verbose:
            print(
                f"  → test acc={eval_results['accuracy']:.2f}%  "
                f"(n_labeled={len(train_indices)})"
            )
        return train_history, eval_results


# ---------------------------------------------------------------------------
# Multi-seed evaluation helper
# ---------------------------------------------------------------------------

def evaluate_multiple_seeds(
    train_indices: np.ndarray,
    train_dataset,
    test_dataset,
    seeds: list[int],
    epochs: int = 200,
    batch_size: int = 64,
    num_classes: int = 10,
    device: torch.device | None = None,
    num_workers: int = 2,
) -> dict:
    """
    Train CIFARClassifier with multiple random seeds and report mean ± std
    accuracy.

    Useful for reducing variance in AL curve comparisons.

    Args:
        train_indices:  Labeled pool indices.
        train_dataset:  Full training dataset.
        test_dataset:   Test dataset.
        seeds:          List of integer seeds, e.g. [0, 1, 2].
        epochs:         Training epochs per seed.
        batch_size:     Mini-batch size.
        num_classes:    Number of output classes.
        device:         Torch device.
        num_workers:    DataLoader workers.

    Returns:
        summary: dict with keys:
            ``'accuracies'``  — list[float], one per seed (%)
            ``'mean_acc'``    — float, mean accuracy (%)
            ``'std_acc'``     — float, std of accuracy (%)
            ``'per_seed'``    — list[dict], full eval_results per seed
    """
    clf = CIFARClassifier(
        num_classes=num_classes, device=device, num_workers=num_workers
    )
    accuracies = []
    per_seed   = []

    for seed in seeds:
        print(f"[Seed {seed}]")
        _, eval_res = clf.fit_and_evaluate(
            train_indices, train_dataset, test_dataset,
            epochs=epochs, batch_size=batch_size, seed=seed,
        )
        accuracies.append(eval_res["accuracy"])
        per_seed.append(eval_res)

    return {
        "accuracies": accuracies,
        "mean_acc":   float(np.mean(accuracies)),
        "std_acc":    float(np.std(accuracies)),
        "per_seed":   per_seed,
    }


# ---------------------------------------------------------------------------
# Feature-based linear probe (fast path for the AL loop)
# ---------------------------------------------------------------------------

def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int = 10,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
) -> nn.Linear:
    """
    Train a single linear layer on pre-extracted SimCLR features.

    This is the fast evaluation path used inside the active learning loop:
    SimCLR features are extracted once and cached; only the linear head is
    re-trained each round.

    Args:
        train_features: Shape (N, D) numpy array of L2-normalised features.
        train_labels:   Shape (N,) integer label array.
        num_classes:    Number of output classes.
        epochs:         Training epochs.
        batch_size:     Mini-batch size.
        lr:             AdamW learning rate.
        weight_decay:   L2 regularisation.
        device:         Torch device.

    Returns:
        Trained nn.Linear (moved to CPU).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X       = torch.tensor(train_features, dtype=torch.float32)
    y       = torch.tensor(train_labels,   dtype=torch.long)
    loader  = DataLoader(TensorDataset(X, y),
                         batch_size=batch_size, shuffle=True)

    head      = nn.Linear(train_features.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    head.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(head(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return head.cpu()


def evaluate_linear_probe(
    head: nn.Linear,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> float:
    """
    Evaluate a linear probe on pre-extracted test features.

    Returns:
        Top-1 accuracy in [0, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X      = torch.tensor(test_features, dtype=torch.float32)
    y      = torch.tensor(test_labels,   dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y),
                        batch_size=batch_size, shuffle=False)

    head = head.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (head(xb).argmax(1) == yb).sum().item()
            total   += len(yb)

    return correct / total
