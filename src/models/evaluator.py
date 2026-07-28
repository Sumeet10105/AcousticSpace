"""Model evaluator class for scoring performance on a test dataset."""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any
from torch.utils.data import DataLoader

from src.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_confusion_matrix,
    accuracy,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class for scoring and evaluating model performance on a test set."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """Initialize evaluator.

        Args:
            model: PyTorch model to evaluate.
            device: Device to use (default: 'cuda').
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluates the model on the test dataset.

        Calculates EER, Accuracy, ROC-AUC, and Confusion Matrix.

        Args:
            test_loader: DataLoader containing test dataset samples.

        Returns:
            Dictionary containing computed evaluation metric scores.
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                
                _, preds = torch.max(logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of FAKE (class 1)
                all_labels.extend(y.numpy())
                
        all_preds = np.asarray(all_preds) if 'np' in globals() else list(all_preds)
        all_probs = np.asarray(all_probs) if 'np' in globals() else list(all_probs)
        all_labels = np.asarray(all_labels) if 'np' in globals() else list(all_labels)
        
        # Import numpy if not imported
        import numpy as np
        all_preds = np.asarray(all_preds)
        all_probs = np.asarray(all_probs)
        all_labels = np.asarray(all_labels)
        
        # Calculate metrics
        acc = accuracy(all_labels, all_preds)
        eer, eer_threshold = compute_eer(all_labels, all_probs)
        fpr, tpr, auc_score = compute_roc_auc(all_labels, all_probs)
        cm = compute_confusion_matrix(all_labels, all_preds)
        
        logger.info(f"Evaluation results: Accuracy={acc:.4f}, EER={eer:.4f}, AUC={auc_score:.4f}")
        
        return {
            "accuracy": float(acc),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "auc": float(auc_score),
            "confusion_matrix": cm,
            "y_true": all_labels,
            "y_pred": all_preds,
            "y_prob": all_probs
        }
