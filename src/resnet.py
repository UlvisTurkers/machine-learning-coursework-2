"""
ResNet-18 backbone for CIFAR-10.

The canonical SimCLR model (encoder + projection head) lives in simclr.py.
This module provides LinearClassifier for supervised evaluation on top of
a frozen SimCLRModel backbone.
"""

import torch
import torch.nn as nn

# SimCLRModel is defined in simclr.py to keep all SimCLR logic together.
# Import it here so the rest of the codebase can do `from src.resnet import ...`.
from .simclr import SimCLRModel  # noqa: F401  (re-exported for convenience)


class LinearClassifier(nn.Module):
    """
    Frozen SimCLR backbone + single linear head for supervised evaluation.

    Args:
        model:       Trained SimCLRModel whose encoder will be frozen.
        num_classes: Number of output classes (default 10 for CIFAR-10).
    """

    def __init__(self, model: SimCLRModel, num_classes: int = 10):
        super().__init__()
        self.encoder = model.encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(model.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder(x).flatten(1)  # (B, 512)
        return self.fc(h)
