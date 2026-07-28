"""Tests for the Explainable AI (XAI) modules."""

import pytest
import torch
import numpy as np

from src.models import CRNN
from src.xai import GradCAM, compute_integrated_gradients, compute_attention_rollout


class TestXAI:
    """Explainable AI (XAI) tests."""
    
    def test_grad_cam(self):
        """Test Grad-CAM heatmap generation on CRNN model."""
        model = CRNN(num_classes=2, n_mels=64)
        # Target conv1 layer in the conv block
        target_layer = model.conv[0]
        
        gradcam = GradCAM(model, target_layer)
        
        # Dummy batch: [batch_size, channels, freq, time]
        x = torch.randn(1, 1, 64, 100)
        
        # Generate heatmap for class FAKE (1)
        heatmap = gradcam.generate_heatmap(x, class_idx=1)
        
        assert isinstance(heatmap, np.ndarray)
        assert len(heatmap.shape) == 2  # 2D heatmap
        # Check normalization boundaries [0, 1]
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        
        # Clean up hooks
        gradcam.remove_hooks()
        
    def test_integrated_gradients(self):
        """Test Integrated Gradients attribution calculation."""
        model = CRNN(num_classes=2, n_mels=64)
        x = torch.randn(1, 1, 64, 100)
        
        attributions = compute_integrated_gradients(
            model=model,
            x=x,
            target_class=1,
            steps=5,
            device="cpu"
        )
        
        # Attribution must match shape of x (excluding batch dim)
        assert isinstance(attributions, np.ndarray)
        assert attributions.shape == (1, 64, 100)
        
    def test_attention_rollout(self):
        """Test ViT/AST attention rollout algorithm."""
        # 2 layers of attention weights
        # Each shape: [batch, heads, seq_len, seq_len]
        attn1 = torch.rand(1, 4, 10, 10)
        attn2 = torch.rand(1, 4, 10, 10)
        
        rollout = compute_attention_rollout(
            attention_matrices=[attn1, attn2],
            discard_ratio=0.1,
            head_fusion="mean"
        )
        
        assert isinstance(rollout, np.ndarray)
        # Should return attention weights for all tokens except the CLS token (seq_len - 1 = 9)
        assert rollout.shape[0] == 9
        assert rollout.min() >= 0.0
        assert rollout.max() <= 1.0
