"""
ResNet-18 backbone for CIFAR-10.

Provides two variants:
  - SimCLREncoder: ResNet-18 with projection head for self-supervised pre-training.
  - Classifier:    ResNet-18 with linear head for supervised fine-tuning.

The architecture follows the CIFAR-10 adaptations used in the SimCLR paper
(He et al., 2016): the first 7x7 conv is replaced with a 3x3 conv, and the
initial max-pool is removed so that 32x32 inputs are not over-downsampled.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def _cifar_resnet18() -> nn.Module:
    """ResNet-18 with CIFAR-friendly stem (3x3 conv, no max-pool)."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class SimCLREncoder(nn.Module):
    """
    ResNet-18 backbone + 2-layer MLP projection head for SimCLR.

    Args:
        proj_dim: Output dimensionality of the projection head (default 128).
        hidden_dim: Hidden dimensionality of the projection MLP (default 512).
    """

    def __init__(self, proj_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        base = _cifar_resnet18()
        feature_dim = base.fc.in_features  # 512 for ResNet-18

        # Strip the classification head; keep the feature extractor.
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # -> (B, 512, 1, 1)

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)   # (B, 512)
        z = self.projector(h)            # (B, proj_dim)
        return h, z

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (before projection head)."""
        with torch.no_grad():
            h = self.encoder(x).flatten(1)
        return h


class LinearClassifier(nn.Module):
    """
    Frozen backbone + single linear layer for supervised evaluation.

    Args:
        encoder: Pretrained SimCLREncoder (backbone weights are frozen).
        num_classes: Number of output classes (default 10 for CIFAR-10).
    """

    def __init__(self, encoder: SimCLREncoder, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder.encoder(x).flatten(1)
        return self.fc(h)
