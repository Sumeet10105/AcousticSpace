"""ROC, Precision-Recall, and DET curve computation utilities."""

import logging
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from typing import Tuple, Union

logger = logging.getLogger(__name__)


def compute_roc_auc(
    y_true: Union[np.ndarray, list], 
    y_pred_scores: Union[np.ndarray, list]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes False Positive Rate, True Positive Rate, and Area Under ROC Curve.

    Args:
        y_true: Ground truth binary labels.
        y_pred_scores: Predicted probabilities/scores.

    Returns:
        Tuple of (fpr, tpr, auc_score).
    """
    y_true = np.asarray(y_true)
    y_pred_scores = np.asarray(y_pred_scores)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_scores, pos_label=1)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, float(auc_score)


def compute_precision_recall(
    y_true: Union[np.ndarray, list], 
    y_pred_scores: Union[np.ndarray, list]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes Precision, Recall, and Average Precision.

    Args:
        y_true: Ground truth binary labels.
        y_pred_scores: Predicted probabilities/scores.

    Returns:
        Tuple of (precision, recall, average_precision).
    """
    y_true = np.asarray(y_true)
    y_pred_scores = np.asarray(y_pred_scores)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_scores)
    ap = average_precision_score(y_true, y_pred_scores)
    return precision, recall, float(ap)


def compute_det_curve(
    y_true: Union[np.ndarray, list], 
    y_pred_scores: Union[np.ndarray, list]
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes False Positive Rate and False Negative Rate for Detection Error Tradeoff (DET).

    Args:
        y_true: Ground truth binary labels.
        y_pred_scores: Predicted probabilities/scores.

    Returns:
        Tuple of (fpr, fnr).
    """
    y_true = np.asarray(y_true)
    y_pred_scores = np.asarray(y_pred_scores)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_scores, pos_label=1)
    fnr = 1.0 - tpr
    return fpr, fnr
