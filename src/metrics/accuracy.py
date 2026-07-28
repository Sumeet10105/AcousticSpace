"""Accuracy calculation utilities: global, attack-wise, and speaker-wise."""

import logging
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Dict, Union

logger = logging.getLogger(__name__)


def accuracy(
    y_true: Union[np.ndarray, list], 
    y_pred: Union[np.ndarray, list]
) -> float:
    """Computes standard classification accuracy.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Accuracy score as a float.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    score = accuracy_score(y_true, y_pred)
    return float(score)


def attack_wise_accuracy(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    attack_types: Union[np.ndarray, list]
) -> Dict[str, float]:
    """Computes accuracy scores sliced by different attack types (A01-A19 or '-').

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        attack_types: List/array identifying the attack types corresponding to each sample.

    Returns:
        Dictionary mapping each attack type key to its classification accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    attack_types = np.asarray(attack_types)
    
    unique_attacks = np.unique(attack_types)
    attack_accs = {}
    
    for attack in unique_attacks:
        mask = (attack_types == attack)
        if np.sum(mask) > 0:
            score = accuracy_score(y_true[mask], y_pred[mask])
            attack_accs[str(attack)] = float(score)
            
    return attack_accs


def speaker_wise_accuracy(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    speaker_ids: Union[np.ndarray, list]
) -> Dict[str, float]:
    """Computes accuracy scores sliced by speaker ID.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        speaker_ids: List/array identifying the speaker ID corresponding to each sample.

    Returns:
        Dictionary mapping speaker ID key to its classification accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    speaker_ids = np.asarray(speaker_ids)
    
    unique_speakers = np.unique(speaker_ids)
    speaker_accs = {}
    
    for speaker in unique_speakers:
        mask = (speaker_ids == speaker)
        if np.sum(mask) > 0:
            score = accuracy_score(y_true[mask], y_pred[mask])
            speaker_accs[str(speaker)] = float(score)
            
    return speaker_accs
