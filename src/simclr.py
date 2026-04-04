"""
SimCLR self-supervised pre-training on CIFAR-10.

Implements the training procedure from:
    Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
    A Simple Framework for Contrastive Learning of Visual Representations.
    ICML 2020. https://arxiv.org/abs/2002.05709

Architecture
------------
- Encoder  : ResNet-18 with CIFAR-10 stem
              (conv1: 3×3, stride 1, padding 1; maxpool removed)
- Projector: Linear(512, 512) → ReLU → Linear(512, 128)

Training defaults (paper §B.9 / CIFAR settings)
-----------------------------------------------
- Optimizer : SGD, momentum 0.9, weight decay 1e-4
- LR        : 0.4, cosine annealing over num_epochs
- Batch size: 512
- Epochs    : 500 (pass num_epochs=100 for quick tests)
- NT-Xent temperature τ = 0.5

Usage
-----
    from src.simclr import SimCLRModel, train_simclr, get_features
    from torchvision.datasets import CIFAR10

    train_ds = CIFAR10(root='data', train=True, download=True)
    model = SimCLRModel()
    model = train_simclr(train_ds, model, num_epochs=100,
                         checkpoint_path='results/simclr.pt')
    # later: load checkpoint and extract features
    feats, labels = get_features(model, train_ds)   # (50000, 512), L2-normalised
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CIFAR-10 normalisation constants
# ---------------------------------------------------------------------------

_CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR_STD  = [0.2023, 0.1994, 0.2010]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimCLRModel(nn.Module):
    """
    ResNet-18 encoder (CIFAR-adapted) + 2-layer MLP projection head.

    The encoder outputs 512-dim backbone features (h).
    The projector maps h → 128-dim unit-sphere embeddings (z) used for
    NT-Xent loss during training.

    For downstream tasks, use h (encoder output) not z.
    """

    def __init__(self):
        super().__init__()

        # --- Backbone: ResNet-18 modified for 32×32 CIFAR images ---
        base = tv_models.resnet18(weights=None)
        # Replace 7×7 stride-2 conv with 3×3 stride-1 conv (keeps spatial resolution)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the maxpool that would halve 32→16 before the residual blocks
        base.maxpool = nn.Identity()

        # Drop the classification head; keep conv layers + global avgpool
        # children order: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # → (B, 512, 1, 1)
        self.feature_dim = 512  # ResNet-18 penultimate dimension

        # --- Projection head: Linear(512,512) → ReLU → Linear(512,128) ---
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, 3, 32, 32).

        Returns:
            h: Backbone features  (B, 512)  — use for downstream tasks.
            z: Projected embeddings (B, 128) — use for NT-Xent loss.
        """
        h = self.encoder(x).flatten(1)  # (B, 512)
        z = self.projector(h)           # (B, 128)
        return h, z


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def simclr_augment() -> transforms.Compose:
    """
    SimCLR augmentation pipeline for 32×32 CIFAR-10 images.

    Follows §A of Chen et al. (2020):
        1. RandomResizedCrop(32, scale=(0.2, 1.0))
        2. RandomHorizontalFlip(p=0.5)
        3. ColorJitter(b=0.4, c=0.4, s=0.4, h=0.1)  applied with p=0.8
        4. RandomGrayscale(p=0.2)
        5. Normalize with CIFAR-10 per-channel statistics
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD),
    ])


class _TwoViewDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset that returns (PIL Image, label) and produces two
    independently augmented views of each image.  Labels are discarded.
    """

    def __init__(self, dataset, transform: transforms.Compose):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.dataset[idx]  # img is a PIL Image
        return self.transform(img), self.transform(img)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------

def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Normalised Temperature-scaled Cross-Entropy loss (NT-Xent).

    For a mini-batch of N samples, there are 2N augmented views.
    Each view's positive pair is the other augmented view of the same image.
    All other 2(N-1) views in the batch are treated as negatives.

    Args:
        z1, z2:      Projected embeddings, shape (N, D).
                     Need NOT be pre-normalised — L2 normalisation is
                     applied inside this function.
        temperature: Softmax temperature τ (paper default: 0.5).

    Returns:
        Scalar loss averaged over all 2N views.
    """
    N = z1.size(0)
    # Stack and L2-normalise: (2N, D)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)

    # Pairwise cosine similarity matrix divided by τ: (2N, 2N)
    sim = torch.mm(z, z.T) / temperature

    # Mask self-similarity on the diagonal (log(exp(s_ii/τ) would be 1 after
    # normalisation, contributing zero loss — but we mask for numerical safety)
    self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float('-inf'))

    # Positive pair labels: view i pairs with view i+N, and vice versa
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(N,        device=z.device),
    ])

    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_simclr(
    dataset,
    model: SimCLRModel,
    num_epochs: int = 500,
    batch_size: int = 512,
    lr: float = 0.4,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 50,
    resume: bool = True,
    device: torch.device | None = None,
    num_workers: int = 2,
) -> SimCLRModel:
    """
    Train a SimCLRModel with SGD + cosine LR annealing.

    Args:
        dataset:          Dataset returning (PIL Image, label) pairs.
                          Labels are ignored — the full dataset is used as the
                          unlabelled pre-training corpus.
        model:            SimCLRModel instance (randomly initialised or from a
                          previous checkpoint).
        num_epochs:       Total training epochs (paper: 500; use 100 for quick
                          tests on Colab).
        batch_size:       Mini-batch size (paper: 512).
        lr:               Base learning rate (paper: 0.4 for batch 512).
        momentum:         SGD momentum (paper: 0.9).
        weight_decay:     SGD weight decay (paper: 1e-4).
        temperature:      NT-Xent temperature τ (paper: 0.5).
        checkpoint_path:  Path to save/load a `.pt` checkpoint file.
                          If None, no checkpointing is performed.
        checkpoint_every: Save a checkpoint every this many epochs.
        resume:           If True and `checkpoint_path` exists, resume from it.
        device:           Torch device.  Auto-detected if None.
        num_workers:      DataLoader worker processes.  Set to 0 on Windows if
                          you encounter multiprocessing errors.

    Returns:
        Trained SimCLRModel (on CPU after training completes).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SimCLR] device={device}  epochs={num_epochs}  batch={batch_size}  lr={lr}")

    # -----------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------
    two_view = _TwoViewDataset(dataset, simclr_augment())
    loader = DataLoader(
        two_view,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,  # NT-Xent assumes uniform batch size
    )

    # -----------------------------------------------------------------------
    # Optimiser + scheduler
    # -----------------------------------------------------------------------
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    # Cosine annealing decays LR from `lr` to 0 over `num_epochs` steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0.0
    )

    # -----------------------------------------------------------------------
    # Optional checkpoint resume
    # -----------------------------------------------------------------------
    start_epoch = 1
    if checkpoint_path is not None and resume and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[SimCLR] Resumed from checkpoint '{checkpoint_path}' "
              f"(epoch {ckpt['epoch']})")

    if start_epoch > num_epochs:
        print("[SimCLR] Already trained for the requested number of epochs.")
        return model.cpu()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model.train()
    epoch_bar = tqdm(
        range(start_epoch, num_epochs + 1),
        desc="SimCLR",
        unit="epoch",
    )
    for epoch in epoch_bar:
        running_loss = 0.0

        batch_bar = tqdm(loader, desc=f"  epoch {epoch:>3d}", leave=False, unit="batch")
        for v1, v2 in batch_bar:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)

            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = running_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.5f}")

        # -------------------------------------------------------------------
        # Checkpoint
        # -------------------------------------------------------------------
        if checkpoint_path is not None and epoch % checkpoint_every == 0:
            _save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)

    # Final checkpoint
    if checkpoint_path is not None:
        _save_checkpoint(model, optimizer, scheduler, num_epochs, checkpoint_path)
        print(f"[SimCLR] Final checkpoint saved to '{checkpoint_path}'")

    return model.cpu()


def _save_checkpoint(
    model: SimCLRModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str | Path,
) -> None:
    """Save a resumable training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_features(
    model: SimCLRModel,
    dataset,
    batch_size: int = 512,
    device: torch.device | None = None,
    num_workers: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract L2-normalised 512-dim backbone features for every sample in
    `dataset` using the trained SimCLR encoder (before the projection head).

    Args:
        model:       Trained SimCLRModel.
        dataset:     Dataset returning (PIL Image, label) pairs.
        batch_size:  Inference batch size.
        device:      Torch device.  Auto-detected if None.
        num_workers: DataLoader workers.

    Returns:
        features: np.ndarray of shape (N, 512), dtype float32, L2-normalised.
        labels:   np.ndarray of shape (N,),    dtype int64.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Deterministic single-crop transform (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD),
    ])

    class _EvalWrapper(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img, label = self.ds[idx]
            return eval_transform(img), label

    loader = DataLoader(
        _EvalWrapper(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = model.to(device).eval()
    all_features, all_labels = [], []

    for xb, yb in tqdm(loader, desc="Extracting features", unit="batch", leave=False):
        xb = xb.to(device, non_blocking=True)
        # Use encoder only (penultimate layer, before projection head)
        h = model.encoder(xb).flatten(1)                  # (B, 512)
        h = F.normalize(h, dim=1)                          # L2-normalise
        all_features.append(h.cpu().numpy())
        all_labels.append(yb.numpy())

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels   = np.concatenate(all_labels,   axis=0).astype(np.int64)
    return features, labels


def load_simclr_model(checkpoint_path: str | Path, device: torch.device | None = None) -> SimCLRModel:
    """
    Load a SimCLRModel from a saved checkpoint.

    Args:
        checkpoint_path: Path to a `.pt` file saved by `train_simclr`.
        device:          Device to load onto.

    Returns:
        SimCLRModel with weights restored, in eval mode on CPU.
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SimCLRModel()
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[SimCLR] Loaded model from '{checkpoint_path}' (epoch {ckpt['epoch']})")
    return model
