"""
SimCLR self-supervised pre-training on CIFAR-10.

Reference:
    Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
    A Simple Framework for Contrastive Learning of Visual Representations.
    ICML 2020. https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .resnet import SimCLREncoder


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def simclr_augment(image_size: int = 32) -> transforms.Compose:
    """Return the SimCLR augmentation pipeline for CIFAR-10 (32x32 images)."""
    s = 0.5  # colour jitter strength (reduced for small images)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])


class TwoViewDataset(torch.utils.data.Dataset):
    """Wraps an existing dataset and returns two augmented views per sample."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        # img is a PIL Image (raw dataset without transforms applied yet)
        return self.transform(img), self.transform(img)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Normalised temperature-scaled cross-entropy loss (NT-Xent).

    Args:
        z1, z2: Projected embeddings of shape (N, D), one per view.
        temperature: Softmax temperature tau.

    Returns:
        Scalar loss value.
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                       # (2N, D)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.T) / temperature                  # (2N, 2N)

    # Mask out self-similarities on the diagonal.
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_simclr(
    dataset,
    encoder: SimCLREncoder,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    temperature: float = 0.5,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    num_workers: int = 2,
) -> SimCLREncoder:
    """
    Train SimCLR on an unlabelled (or pseudo-unlabelled) dataset.

    Args:
        dataset: A dataset that returns (PIL Image, label) pairs. Labels are
                 ignored during SimCLR training.
        encoder: SimCLREncoder instance to train.
        epochs:  Number of training epochs.
        batch_size: Mini-batch size.
        lr:      Learning rate for AdamW.
        temperature: NT-Xent temperature.
        weight_decay: AdamW weight decay.
        device:  Torch device. Defaults to CUDA if available.
        num_workers: DataLoader worker count.

    Returns:
        Trained SimCLREncoder.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = simclr_augment()
    two_view = TwoViewDataset(dataset, transform)
    loader = DataLoader(
        two_view,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    encoder = encoder.to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    encoder.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for v1, v2 in tqdm(loader, desc=f"SimCLR epoch {epoch}/{epochs}", leave=False):
            v1, v2 = v1.to(device), v2.to(device)
            _, z1 = encoder(v1)
            _, z2 = encoder(v2)
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        print(f"  [SimCLR] epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    return encoder
