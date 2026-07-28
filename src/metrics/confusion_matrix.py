"""Confusion matrix calculation utility."""

import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Union

logger = logging.getLogger(__name__)


def compute_confusion_matrix(
    y_true: Union[np.ndarray, list], 
    y_pred: Union[np.ndarray, list]
) -> np.ndarray:
    """Computes the confusion matrix for binary classification.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        A 2x2 numpy array representing the confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm
