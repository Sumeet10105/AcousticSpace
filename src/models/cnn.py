"""CNN models for audio spoofing detection."""

import torch
import torch.nn as nn
from torchvision import models


class ResNetCNN(nn.Module):
    """ResNet-based CNN model for audio spoofing detection."""

    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize ResNet CNN model.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load pretrained ResNet18
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=None)

        # Replace final classification layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Forward pass."""
        return self.model(x)