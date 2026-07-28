"""Tests for the model architectures."""

import os
import tempfile
import pytest
import torch

from src.models import (
    ResNetCNN,
    CRNN,
    AudioSpectrogramTransformer,
)


@pytest.fixture
def dummy_batch():
    """Generates a dummy batch of spectrograms: [batch_size, freq_bins, time_steps]."""
    # 2 samples, 64 mel bins, 100 time frames
    return torch.randn(2, 64, 100)


class TestModels:
    """Model architectures and base model tests."""
    
    def test_cnn_forward_and_shapes(self, dummy_batch):
        """Test ResNetCNN forward pass, parameters count and output shape."""
        model = ResNetCNN(num_classes=2, pretrained=False)
        logits = model(dummy_batch)
        
        # Test shape
        assert logits.shape[0] == 2
        assert logits.shape[1] == 2
        
        # Test parameter count is an integer and > 0
        params_count = model.count_parameters()
        assert isinstance(params_count, int)
        assert params_count > 0
        
    def test_crnn_forward_and_shapes(self, dummy_batch):
        """Test CRNN forward pass, parameters count and output shape."""
        model = CRNN(num_classes=2, n_mels=64)
        logits = model(dummy_batch)
        
        assert logits.shape[0] == 2
        assert logits.shape[1] == 2
        
        params_count = model.count_parameters()
        assert params_count > 0

    def test_ast_forward_and_shapes(self, dummy_batch):
        """Test AST forward pass, parameters count and output shape."""
        # Test with pretrained=False to avoid downloading large weights during tests
        model = AudioSpectrogramTransformer(num_classes=2, pretrained=False)
        logits = model(dummy_batch)
        
        assert logits.shape[0] == 2
        assert logits.shape[1] == 2
        
        params_count = model.count_parameters()
        assert params_count > 0

    def test_base_model_save_load(self, dummy_batch):
        """Test BaseModel weights serialization and deserialization."""
        model = CRNN(num_classes=2, n_mels=64)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "crnn_weights.pt")
            
            # Save weights
            model.save_model(save_path)
            assert os.path.exists(save_path)
            
            # Create a new blank model and load weights
            new_model = CRNN(num_classes=2, n_mels=64)
            new_model.load_model(save_path)
            
            # Put both models in evaluation mode to turn off dropout/batchnorm
            model.eval()
            new_model.eval()
            
            # Verify outputs are identical for the same input
            with torch.no_grad():
                out_orig = model(dummy_batch)
                out_loaded = new_model(dummy_batch)
            assert torch.allclose(out_orig, out_loaded, atol=1e-4)
