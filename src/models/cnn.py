"""CNN model architectures for audio deepfake spoofing detection."""

import logging
import torch
import torch.nn as nn
import torchvision.models as models
from src.models.base_model import BaseModel
from src.models.classifier_head import ClassifierHead

logger = logging.getLogger(__name__)


class ResNetCNN(BaseModel):
    """ResNet-based CNN model for audio spoofing detection.

    Adapts a standard ResNet18 backbone to take 1-channel spectrogram features
    and maps them to class logits.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """Initialize ResNet CNN model.

        Args:
            num_classes: Number of target output classes (default: 2).
            pretrained: Whether to use ImageNet pretrained weights (default: True).
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Load torchvision resnet18
        if pretrained:
            # Handle newer and older torchvision versions for loading weights
            try:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except AttributeError:
                resnet = models.resnet18(pretrained=True)
            logger.info("Loaded pretrained ResNet18 weights.")
        else:
            resnet = models.resnet18(weights=None)
            logger.info("Initialized ResNet18 with random weights.")
            
        # Modify the first conv layer to accept 1 channel instead of 3
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # Adapt pretrained weights for the new 1-channel conv layer
        if pretrained:
            with torch.no_grad():
                # Average the weights across the channel dimension
                resnet.conv1.weight.copy_(torch.mean(original_conv.weight, dim=1, keepdim=True))
                
        # Remove the fully connected layer, we will use our classifier head
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier Head
        self.classifier = ClassifierHead(
            in_dim=512,  # ResNet18 output feature dim
            num_classes=num_classes,
            hidden_dims=[128],
            dropout=0.3
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input spectrogram tensor of shape [batch, channels, freq_bins, time_steps].
               Normally channels=1 for raw spectrograms.

        Returns:
            Logits tensor of shape [batch, num_classes].
        """
        # Ensure input has channel dimension [batch, 1, freq_bins, time_steps]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        features = self.backbone(x)
        # flatten features
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits
