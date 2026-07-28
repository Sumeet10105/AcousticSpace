"""Convolutional Recurrent Neural Network (CRNN) for audio deepfake detection."""

import logging
import torch
import torch.nn as nn
from src.models.base_model import BaseModel
from src.models.classifier_head import ClassifierHead

logger = logging.getLogger(__name__)


class CRNN(BaseModel):
    """Convolutional Recurrent Neural Network (CRNN).

    Combines 2D CNN layers for local spatial-frequency feature extraction
    with a bidirectional GRU for temporal sequence modeling.
    """
    
    def __init__(
        self, 
        num_classes: int = 2, 
        n_mels: int = 64, 
        rnn_hidden_dim: int = 128,
        rnn_layers: int = 2
    ):
        """Initialize CRNN model.

        Args:
            num_classes: Number of target output classes (default: 2).
            n_mels: Number of frequency bins in the input spectrogram (default: 64).
            rnn_hidden_dim: Hidden dimension of the GRU (default: 128).
            rnn_layers: Number of GRU layers (default: 2).
        """
        super().__init__()
        self.num_classes = num_classes
        
        # 1. 2D CNN Block
        # Input shape: [batch, 1, freq_bins, time_steps]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # reduces freq_bins and time_steps by 2
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # reduces freq_bins and time_steps by 2
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1))   # reduces only freq_bins, keeps time resolution
        )
        
        # Calculate the feature size after CNN reduction
        # For n_mels=64, after pools of (2,2), (2,2), (2,1):
        # freq size = 64 // 2 // 2 // 2 = 8
        cnn_freq_out = n_mels // 8
        self.cnn_features_dim = 128 * cnn_freq_out
        
        # 2. Recurrent Block (Bidirectional GRU)
        self.gru = nn.GRU(
            input_size=self.cnn_features_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if rnn_layers > 1 else 0.0
        )
        
        # 3. Attention Layer (Self-Attention over GRU hidden states)
        self.attention = nn.Sequential(
            nn.Linear(rnn_hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 4. Classifier Head
        self.classifier = ClassifierHead(
            in_dim=rnn_hidden_dim * 2,
            num_classes=num_classes,
            hidden_dims=[64],
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
        # Ensure shape has channel dimension [batch, 1, freq_bins, time_steps]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        batch_size, _, _, time_steps = x.shape
        
        # CNN Forward: shape [batch, 128, freq_bins_reduced, time_steps_reduced]
        cnn_out = self.conv(x)
        
        # Reshape to sequence along time dimension
        # shape [batch, channels, features, sequence_len] -> permute to [batch, sequence_len, channels, features]
        # then flatten channels & features: [batch, sequence_len, channels * features]
        b, c, f, t = cnn_out.shape
        # Permute so time is index 1: [batch, time, channels, features]
        features = cnn_out.permute(0, 3, 1, 2).contiguous()
        # Flatten channels and features: [batch, time, channels * features]
        features = features.view(b, t, c * f)
        
        # GRU Forward: shape [batch, time, hidden_dim * 2]
        gru_out, _ = self.gru(features)
        
        # Self-Attention pooling over time steps
        # weights shape: [batch, time, 1]
        att_weights = torch.softmax(self.attention(gru_out), dim=1)
        # pooled features shape: [batch, hidden_dim * 2]
        pooled = torch.sum(gru_out * att_weights, dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        return logits
