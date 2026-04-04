"""
Supervised classifier training and evaluation on CIFAR-10.

Two training modes are supported:
  1. Linear probing  — backbone frozen, only the linear head is trained.
  2. Full fine-tuning — entire network is fine-tuned on the labelled set.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from .resnet import LinearClassifier
from .simclr import SimCLRModel


# ---------------------------------------------------------------------------
# Standard CIFAR-10 transforms for supervised training
# ---------------------------------------------------------------------------

CIFAR10_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

CIFAR10_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])


# ---------------------------------------------------------------------------
# Feature-based linear probe (fast)
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
    Train a single linear layer on pre-extracted features.

    This is the fast evaluation path: SimCLR features are extracted once
    and a logistic regression head is trained on the labelled subset.

    Args:
        train_features: Shape (N, D) numpy array of SimCLR features.
        train_labels:   Shape (N,) integer label array.
        num_classes:    Number of output classes.
        epochs:         Training epochs.
        batch_size:     Mini-batch size.
        lr:             Learning rate.
        weight_decay:   L2 regularisation.
        device:         Torch device.

    Returns:
        Trained nn.Linear layer (on CPU).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(train_features, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    D = train_features.shape[1]
    head = nn.Linear(D, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    head.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(head(xb), yb)
            optimizer.zero_grad()
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

    X = torch.tensor(test_features, dtype=torch.float32)
    y = torch.tensor(test_labels, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    head = head.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = head(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)

    return correct / total


# ---------------------------------------------------------------------------
# Full end-to-end fine-tuning
# ---------------------------------------------------------------------------

def train_classifier(
    train_dataset,
    labelled_indices: np.ndarray,
    encoder: SimCLRModel,
    num_classes: int = 10,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    num_workers: int = 2,
) -> LinearClassifier:
    """
    Train a LinearClassifier (frozen backbone + linear head) on the labelled subset.

    Args:
        train_dataset:    Full training dataset (returns PIL images).
        labelled_indices: Indices within train_dataset to use for training.
        encoder:          Pre-trained SimCLRModel (backbone frozen).
        num_classes:      Number of output classes.
        epochs:           Training epochs.
        batch_size:       Mini-batch size.
        lr:               Learning rate.
        weight_decay:     L2 regularisation.
        device:           Torch device.
        num_workers:      DataLoader workers.

    Returns:
        Trained LinearClassifier.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Attach supervised transform to the labelled subset.
    class TransformWrapper(torch.utils.data.Dataset):
        def __init__(self, ds, transform):
            self.ds = ds
            self.transform = transform

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img, label = self.ds[idx]
            return self.transform(img), label

    labelled_subset = Subset(
        TransformWrapper(train_dataset, CIFAR10_TRAIN_TRANSFORM),
        labelled_indices,
    )
    loader = DataLoader(
        labelled_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model = LinearClassifier(encoder, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    [Classifier] epoch {epoch:3d}/{epochs}  "
                  f"loss={total_loss / len(loader):.4f}  "
                  f"train_acc={correct / total:.4f}")

    return model


def evaluate_classifier(
    model: LinearClassifier,
    test_dataset,
    batch_size: int = 256,
    device: torch.device | None = None,
    num_workers: int = 2,
) -> float:
    """
    Evaluate a classifier on the full test set.

    Returns:
        Top-1 accuracy in [0, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TransformWrapper(torch.utils.data.Dataset):
        def __init__(self, ds, transform):
            self.ds = ds
            self.transform = transform

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img, label = self.ds[idx]
            return self.transform(img), label

    loader = DataLoader(
        TransformWrapper(test_dataset, CIFAR10_TEST_TRANSFORM),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)

    return correct / total
