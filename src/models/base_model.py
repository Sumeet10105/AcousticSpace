"""Base model class with generic utilities for AcousticSpace models."""

import logging
import os
import torch
import torch.nn as nn
from typing import Dict

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """Abstract Base Model for Deepfake Audio Detection models.
    
    Provides serialization, logging, and inspection utilities.
    """
    
    def __init__(self):
        super().__init__()
        
    def count_parameters(self) -> int:
        """Returns the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def save_model(self, filepath: str) -> None:
        """Saves model weights to a file.

        Args:
            filepath: Path to save the model file.
        """
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)
        logger.info(f"Model saved successfully to {filepath}. Trainable params: {self.count_parameters()}")
        
    def load_model(self, filepath: str, device: str = "cpu") -> None:
        """Loads model weights from a file. Handles raw state dicts and full checkpoints.

        Args:
            filepath: Path to the saved weights file.
            device: Device to map model to.
        """
        if not os.path.exists(filepath):
            logger.error(f"Weights file not found: {filepath}")
            raise FileNotFoundError(f"Weights file not found: {filepath}")
            
        checkpoint = torch.load(filepath, map_location=device)
        
        # Check if the file is a full checkpoint state dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        self.load_state_dict(state_dict)
        logger.info(f"Model weights loaded successfully from {filepath}.")

