"""Tests for the visualization module."""

import os
import tempfile
import matplotlib
# Use non-interactive backend for testing
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_mel_spectrogram,
    plot_mfcc,
    plot_attention_overlay,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
)


@pytest.fixture
def temp_dir():
    """Fixture for temporary save path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestVisualization:
    """Visualization tests."""
    
    def test_waveform_plotting(self, temp_dir):
        """Test waveform plotting and automatic saving."""
        waveform = torch.randn(1, 16000)
        save_path = os.path.join(temp_dir, "waveform.png")
        
        fig = plot_waveform(waveform, sr=16000, title="Test Waveform", save_path=save_path)
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(save_path)
        plt.close(fig)
        
    def test_spectrogram_plotting(self, temp_dir):
        """Test spectrogram plotting options."""
        spec = torch.randn(1, 64, 100)
        save_path_mel = os.path.join(temp_dir, "mel.png")
        save_path_mfcc = os.path.join(temp_dir, "mfcc.png")
        
        fig_mel = plot_mel_spectrogram(spec, title="Test Mel Spec", save_path=save_path_mel)
        fig_mfcc = plot_mfcc(spec, title="Test MFCC", save_path=save_path_mfcc)
        
        assert isinstance(fig_mel, plt.Figure)
        assert isinstance(fig_mfcc, plt.Figure)
        assert os.path.exists(save_path_mel)
        assert os.path.exists(save_path_mfcc)
        
        plt.close(fig_mel)
        plt.close(fig_mfcc)
        
    def test_attention_overlay(self, temp_dir):
        """Test overlaying attention heatmaps on spectrograms."""
        spec = torch.randn(1, 64, 100)
        # 1D attention per frame
        att_map = torch.rand(100)
        save_path = os.path.join(temp_dir, "attention.png")
        
        fig = plot_attention_overlay(spec, att_map, title="XAI Overlay", save_path=save_path)
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(save_path)
        plt.close(fig)
        
    def test_general_plots(self, temp_dir):
        """Test confusion matrix, ROC curve, and training history plots."""
        # 1. Confusion Matrix
        cm = np.array([[10, 2], [1, 15]])
        save_cm = os.path.join(temp_dir, "cm.png")
        fig_cm = plot_confusion_matrix(cm, save_path=save_cm)
        assert isinstance(fig_cm, plt.Figure)
        assert os.path.exists(save_cm)
        plt.close(fig_cm)
        
        # 2. ROC Curve
        fpr = np.array([0.0, 0.1, 0.2, 1.0])
        tpr = np.array([0.0, 0.8, 0.9, 1.0])
        save_roc = os.path.join(temp_dir, "roc.png")
        fig_roc = plot_roc_curve(fpr, tpr, auc_score=0.92, save_path=save_roc)
        assert isinstance(fig_roc, plt.Figure)
        assert os.path.exists(save_roc)
        plt.close(fig_roc)
        
        # 3. Training History
        history = {
            "train_loss": [0.5, 0.3, 0.2],
            "val_loss": [0.6, 0.4, 0.3],
            "train_acc": [0.7, 0.8, 0.9],
            "val_acc": [0.65, 0.75, 0.82],
        }
        save_history = os.path.join(temp_dir, "history.png")
        fig_hist = plot_training_history(history, save_path=save_history)
        assert isinstance(fig_hist, plt.Figure)
        assert os.path.exists(save_history)
        plt.close(fig_hist)
