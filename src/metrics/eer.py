"""Equal Error Rate (EER) computation utility."""

import logging
import numpy as np
from sklearn.metrics import roc_curve
from typing import Tuple, Union

logger = logging.getLogger(__name__)


def compute_eer(
    y_true: Union[np.ndarray, list], 
    y_pred_scores: Union[np.ndarray, list]
) -> Tuple[float, float]:
    """Computes the Equal Error Rate (EER) and the crossing decision threshold.

    EER is the rate where False Positive Rate (FPR/FAR) equals False Negative Rate (FNR/FRR).

    Args:
        y_true: Ground truth binary labels (0 for real, 1 for spoof).
        y_pred_scores: Predicted probabilities or scores for class 1 (spoof).

    Returns:
        A tuple of (eer_value, threshold).
    """
    y_true = np.asarray(y_true)
    y_pred_scores = np.asarray(y_pred_scores)
    
    if len(y_true) == 0:
        logger.error("Empty labels array provided for EER computation.")
        raise ValueError("Inputs cannot be empty.")
        
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # Find the index where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fpr - fnr))
    
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    
    logger.info(f"Computed EER: {eer:.6f} at decision threshold: {threshold:.6f}")
    return float(eer), float(threshold)
