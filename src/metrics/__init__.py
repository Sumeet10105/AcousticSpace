"""Evaluation metrics module for AcousticSpace."""

from src.metrics.eer import compute_eer
from src.metrics.roc import compute_roc_auc, compute_precision_recall, compute_det_curve
from src.metrics.confusion_matrix import compute_confusion_matrix
from src.metrics.accuracy import accuracy, attack_wise_accuracy, speaker_wise_accuracy

__all__ = [
    "compute_eer",
    "compute_roc_auc",
    "compute_precision_recall",
    "compute_det_curve",
    "compute_confusion_matrix",
    "accuracy",
    "attack_wise_accuracy",
    "speaker_wise_accuracy",
]
