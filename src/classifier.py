# Supervised classifier training and evaluation on CIFAR-10.
#
# CIFARClassifier - ResNet-18 trained from scratch on the labeled subset,
#   following Hacohen et al. (2022) Appendix F.2.1.
# train_linear_probe - lightweight linear head on frozen SimCLR features.

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


class _TransformDataset(torch.utils.data.Dataset):
    # applies a torchvision transform to a dataset that returns PIL images

    def __init__(self, dataset, transform):
        self.dataset   = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img), label


class CIFARClassifier:
    # ResNet-18 trained from scratch on a labeled CIFAR-10 subset.
    # Weights are re-initialised at the start of every train() call
    # so each active learning round starts fresh.

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
        self.model: nn.Module = self._build_model()

    def _build_model(self) -> nn.Module:
        # ResNet-18 with CIFAR-10 stem: 3x3 conv1, no maxpool
        model = tv_models.resnet18(weights=None)
        model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc      = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def _reset_weights(self, seed: int) -> None:
        # rebuild from scratch to guarantee clean state between AL rounds
        torch.manual_seed(seed)
        self.model = self._build_model()

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
        # train on the labeled subset, re-initialising weights first
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

    def evaluate(
        self,
        test_dataset,
        batch_size: int = 256,
    ) -> dict:
        # evaluate on the test set, returns accuracy and per-class breakdown
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

    def predict_proba(
        self,
        dataset,
        indices: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        # return softmax probabilities for samples at the given indices
        # used by UncertaintySelection and MarginSelection
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
        # convenience: train then evaluate in one call
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


class LinearClassifier:
    # linear head trained on frozen SimCLR features (Framework 2).
    # Unlike CIFARClassifier, this works directly on pre-extracted feature
    # vectors so no image augmentation or CNN training is involved.

    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 10,
        device: str | torch.device | None = None,
        seed: int = 42,
        num_workers: int = 0,
    ):
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.device = (
            torch.device(device) if isinstance(device, str)
            else (device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )
        self.seed        = seed
        self.num_workers = num_workers
        self.model: nn.Module = self._build_model()

    def _build_model(self) -> nn.Module:
        return nn.Linear(self.input_dim, self.num_classes).to(self.device)

    def _reset_weights(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.model = self._build_model()

    def train(
        self,
        labeled_indices: np.ndarray,
        all_features: np.ndarray,
        all_labels: np.ndarray,
        epochs: int = 100,
        lr: float = 0.1,
        batch_size: int = 256,
        weight_decay: float = 5e-4,
        seed: int | None = None,
    ) -> dict:
        # train on the subset of features selected by labeled_indices
        _seed = seed if seed is not None else self.seed
        self._reset_weights(_seed)

        labeled_indices = np.asarray(labeled_indices, dtype=np.int64)
        X = torch.tensor(all_features[labeled_indices], dtype=torch.float32)
        y = torch.tensor(all_labels[labeled_indices],   dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=min(batch_size, len(X)),
            shuffle=True,
            num_workers=self.num_workers,
        )

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0
        )
        criterion = nn.CrossEntropyLoss()

        history: dict[str, list] = {"train_loss": [], "train_acc": []}
        for epoch in range(epochs):
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
            history["train_loss"].append(running_loss / total)
            history["train_acc"].append(correct / total)

        return history

    def evaluate(
        self,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        batch_size: int = 512,
    ) -> dict:
        # evaluate on test features, returns accuracy and per-class breakdown
        X = torch.tensor(test_features, dtype=torch.float32)
        y = torch.tensor(test_labels,   dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.model = self.model.to(self.device).eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in loader:
                preds = self.model(xb.to(self.device)).argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
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
    # train with multiple seeds and report mean +/- std accuracy
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
    # train a single linear layer on pre-extracted SimCLR features
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
    # evaluate a linear probe, returns top-1 accuracy in [0, 1]
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
