"""Audio Spectrogram Transformer (AST) model architecture."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel
from src.models.classifier_head import ClassifierHead

logger = logging.getLogger(__name__)


class AudioSpectrogramTransformer(BaseModel):
    """Audio Spectrogram Transformer (AST) for spoofing detection.

    Wraps Hugging Face AST model. Resizes arbitrary input spectrograms
    to 128x1024 to match AST's pretraining dimensions, enabling transfer learning
    from AudioSet.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """Initialize AST model.

        Args:
            num_classes: Number of target output classes (default: 2).
            pretrained: Whether to use pretrained weights from AudioSet (default: True).
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Lazy import of transformers to prevent heavy load on startup
        from transformers import ASTModel, ASTConfig
        
        # Load backbone
        if pretrained:
            try:
                self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset")
                logger.info("Loaded pretrained Audio Spectrogram Transformer (AST) from Hugging Face.")
            except Exception as e:
                logger.warning(
                    f"Could not download pretrained AST weights ({str(e)}). "
                    "Falling back to randomly initialized AST configuration."
                )
                config = ASTConfig(num_mel_bins=128, max_length=1024)
                self.ast = ASTModel(config)
        else:
            config = ASTConfig(num_mel_bins=128, max_length=1024)
            self.ast = ASTModel(config)
            logger.info("Initialized AST model with random weights.")
            
        # The AST backbone outputs 768-dim embeddings
        self.classifier = ClassifierHead(
            in_dim=768,
            num_classes=num_classes,
            hidden_dims=[256],
            dropout=0.3
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input spectrogram tensor of shape [batch, channels, freq_bins, time_steps]
               or [batch, freq_bins, time_steps].

        Returns:
            Logits tensor of shape [batch, num_classes].
        """
        # Remove channel dimension if present [batch, 1, freq_bins, time_steps] -> [batch, freq_bins, time_steps]
        if len(x.shape) == 4:
            x = x.squeeze(1)
            
        batch_size, freq_bins, time_steps = x.shape
        
        # AST expects input size [batch, 1024, 128] where 1024 is time length, 128 is mel bins.
        # We resize/interpolate the input spectrogram to [128, 1024]
        # F.interpolate takes [batch, channel, height, width]
        x_reshaped = x.unsqueeze(1)  # shape [batch, 1, freq_bins, time_steps]
        x_resized = F.interpolate(
            x_reshaped,
            size=(128, 1024),
            mode='bilinear',
            align_corners=False
        )  # shape [batch, 1, 128, 1024]
        
        # Remove channel dimension and transpose to [batch, 1024, 128] to match AST input
        x_ast = x_resized.squeeze(1).transpose(1, 2)  # shape [batch, 1024, 128]
        
        # Forward through Hugging Face AST
        # outputs.last_hidden_state: shape [batch, 1214, 768] (sequence length including CLS token)
        # outputs.pooler_output: shape [batch, 768] (CLS token output state representation)
        outputs = self.ast(x_ast)
        
        # Use pooler output (CLS token representation) for classification
        cls_representation = outputs.pooler_output
        
        # Classify
        logits = self.classifier(cls_representation)
        return logits
