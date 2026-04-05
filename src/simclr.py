# SimCLR self-supervised pre-training on CIFAR-10.
# Based on Chen et al. (2020) - "A Simple Framework for Contrastive Learning"

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

_CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR_STD  = [0.2023, 0.1994, 0.2010]


class SimCLRModel(nn.Module):
    # ResNet-18 encoder (CIFAR-adapted) + 2-layer MLP projection head.
    # Encoder outputs 512-dim features (h), projector maps to 128-dim (z) for NT-Xent.

    def __init__(self):
        super().__init__()

        # ResNet-18 modified for 32x32 CIFAR images
        base = tv_models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        # everything except the final fc layer
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = 512

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)  # (B, 512) backbone features
        z = self.projector(h)           # (B, 128) projected embeddings
        return h, z


def simclr_augment() -> transforms.Compose:
    # SimCLR augmentation following the paper's appendix A
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
    # wraps a dataset to produce two independently augmented views per image

    def __init__(self, dataset, transform: transforms.Compose):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img, _ = self.dataset[idx]
        return self.transform(img), self.transform(img)


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    # NT-Xent (normalised temperature-scaled cross-entropy) loss.
    # each view's positive pair is the other augmentation of the same image.
    N = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)

    # cosine similarity matrix scaled by temperature
    sim = torch.mm(z, z.T) / temperature

    # mask out self-similarity on the diagonal
    self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float('-inf'))

    # positive pair labels: view i pairs with view i+N and vice versa
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(N,        device=z.device),
    ])

    return F.cross_entropy(sim, labels)


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
    # train SimCLR with SGD + cosine LR annealing
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SimCLR] device={device}  epochs={num_epochs}  batch={batch_size}  lr={lr}")

    two_view = _TwoViewDataset(dataset, simclr_augment())
    loader = DataLoader(
        two_view,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0.0
    )

    # resume from checkpoint if available
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

        if checkpoint_path is not None and epoch % checkpoint_every == 0:
            _save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)

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


@torch.no_grad()
def get_features(
    model: SimCLRModel,
    dataset,
    batch_size: int = 512,
    device: torch.device | None = None,
    num_workers: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    # extract L2-normalised 512-dim backbone features for every sample
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        h = model.encoder(xb).flatten(1)
        h = F.normalize(h, dim=1)
        all_features.append(h.cpu().numpy())
        all_labels.append(yb.numpy())

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels   = np.concatenate(all_labels,   axis=0).astype(np.int64)
    return features, labels


def load_simclr_model(checkpoint_path: str | Path, device: torch.device | None = None) -> SimCLRModel:
    # load a trained SimCLR model from a checkpoint file
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SimCLRModel()
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[SimCLR] Loaded model from '{checkpoint_path}' (epoch {ckpt['epoch']})")
    return model
