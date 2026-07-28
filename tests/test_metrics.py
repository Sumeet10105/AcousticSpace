"""Tests for the evaluation metrics module."""

import numpy as np
import pytest

from src.metrics import (
    compute_eer,
    compute_roc_auc,
    compute_precision_recall,
    compute_det_curve,
    compute_confusion_matrix,
    accuracy,
    attack_wise_accuracy,
    speaker_wise_accuracy,
)


class TestMetrics:
    """Evaluation metrics tests."""
    
    def test_eer_computation(self):
        """Test Equal Error Rate (EER) computation."""
        # Clean labels and simple scores
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0.1, 0.2, 0.4, 0.3, 0.8, 0.9]
        
        eer, threshold = compute_eer(y_true, y_pred)
        assert 0.0 <= eer <= 1.0
        assert 0.0 <= threshold <= 1.0
        
    def test_roc_auc_and_curves(self):
        """Test ROC-AUC, PR, and DET calculations."""
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.35, 0.8]
        
        # ROC AUC
        fpr, tpr, auc_score = compute_roc_auc(y_true, y_pred)
        assert len(fpr) > 0
        assert len(tpr) > 0
        assert 0.0 <= auc_score <= 1.0
        
        # Precision-Recall
        precision, recall, ap = compute_precision_recall(y_true, y_pred)
        assert len(precision) > 0
        assert len(recall) > 0
        assert 0.0 <= ap <= 1.0
        
        # DET
        det_fpr, det_fnr = compute_det_curve(y_true, y_pred)
        assert len(det_fpr) == len(det_fnr)
        
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 1, 0, 1, 1]
        
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        # Check actual values
        # TN=1, FP=1, FN=1, TP=2
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 2  # TP
        
    def test_accuracies_slicing(self):
        """Test global, attack-wise, and speaker-wise accuracy computations."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]  # 3/4 correct -> 0.75 accuracy
        
        # Global
        global_acc = accuracy(y_true, y_pred)
        assert abs(global_acc - 0.75) < 1e-5
        
        # Attack wise
        attacks = ["-", "-", "A01", "A02"]
        att_acc = attack_wise_accuracy(y_true, y_pred, attacks)
        # "-" (real): y_true=[0,0], y_pred=[0,1] -> 1/2 correct -> 0.5
        # "A01" (fake): y_true=[1], y_pred=[1] -> 1/1 correct -> 1.0
        # "A02" (fake): y_true=[1], y_pred=[1] -> 1/1 correct -> 1.0
        assert att_acc["-"] == 0.5
        assert att_acc["A01"] == 1.0
        assert att_acc["A02"] == 1.0
        
        # Speaker wise
        speakers = ["SP1", "SP1", "SP2", "SP2"]
        sp_acc = speaker_wise_accuracy(y_true, y_pred, speakers)
        # SP1: y_true=[0,0], y_pred=[0,1] -> 0.5
        # SP2: y_true=[1,1], y_pred=[1,1] -> 1.0
        assert sp_acc["SP1"] == 0.5
        assert sp_acc["SP2"] == 1.0
