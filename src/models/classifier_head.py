"""Unified classification heads for embedding-to-class mapping."""

import torch
import torch.nn as nn
from typing import List, Optional


class ClassifierHead(nn.Module):
    """Universal classification head supporting pooling and Multi-Layer Perceptrons."""
    
    def __init__(
        self, 
        in_dim: int, 
        num_classes: int = 2, 
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        use_batchnorm: bool = True
    ):
        """Initialize ClassifierHead.

        Args:
            in_dim: Input embedding feature dimension.
            num_classes: Number of target output classes.
            hidden_dims: Optional list of hidden layer sizes for MLP.
            dropout: Dropout probability.
            use_batchnorm: Whether to apply batch normalization in hidden layers.
        """
        super().__init__()
        
        layers = []
        current_dim = in_dim
        
        # Build MLP hidden layers if specified
        if hidden_dims is not None:
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                current_dim = h_dim
                
        # Output mapping layer
        layers.append(nn.Linear(current_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature representation of shape [batch, in_dim] or [batch, in_dim, time].

        Returns:
            Logits tensor of shape [batch, num_classes].
        """
        # If input has temporal dimension [batch, features, time], perform global average pooling
        if len(x.shape) == 3:
            x = torch.mean(x, dim=2)
            
        return self.mlp(x)
