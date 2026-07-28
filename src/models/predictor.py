"""Model predictor class for running inference on feature tensors."""

import logging
import torch
import torch.nn as nn
from typing import Dict, Union

logger = logging.getLogger(__name__)


class Predictor:
    """Predictor class for running model inference and returning detailed scores."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', threshold: float = 0.5):
        """Initialize predictor.

        Args:
            model: The PyTorch model backbone.
            device: Device to run inference on (default: 'cuda').
            threshold: Decision threshold above which sample is FAKE (default: 0.5).
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.model.eval()
        
    def predict(self, x: torch.Tensor) -> Dict[str, Union[str, float, torch.Tensor]]:
        """Makes predictions on a batch or single feature tensor.

        Args:
            x: Input feature tensor of shape [batch, channels, freq, time]
               or [freq, time] (will add batch/channel dimensions).

        Returns:
            Dictionary containing:
            - 'prediction': 'REAL' or 'FAKE'
            - 'confidence': confidence score (probability of the winning class)
            - 'probability': raw probability of class FAKE (1)
            - 'probabilities': full probabilities list/tensor
        """
        self.model.eval()
        
        # Ensure correct input dimensions: [batch, channels, freq, time]
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            
        x = x.to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            
        # Class 0: REAL, Class 1: FAKE
        # prob_fake is the probability of class 1 (FAKE)
        prob_fake = probabilities[0, 1].item()
        
        if prob_fake >= self.threshold:
            prediction = "FAKE"
            confidence = prob_fake
        else:
            prediction = "REAL"
            confidence = 1.0 - prob_fake
            
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probability": float(prob_fake),
            "probabilities": probabilities.cpu()
        }
