# ResNet-18 backbone for CIFAR-10.
# SimCLRModel lives in simclr.py; this module provides LinearClassifier
# for supervised evaluation on top of a frozen SimCLR backbone.

import torch
import torch.nn as nn

from .simclr import SimCLRModel  # noqa: F401


class LinearClassifier(nn.Module):
    # frozen SimCLR backbone + single linear head for evaluation

    def __init__(self, model: SimCLRModel, num_classes: int = 10):
        super().__init__()
        self.encoder = model.encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(model.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.encoder(x).flatten(1)
        return self.fc(h)
