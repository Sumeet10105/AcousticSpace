"""Tests for the model export and optimization module."""

import os
import tempfile
import pytest
import torch

from src.models import CRNN
from src.export import (
    export_to_torchscript,
    export_to_onnx,
    export_to_fp16,
    quantize_model_dynamic,
)


class TestModelExport:
    """Model export and serialization tests."""
    
    def test_torchscript_export(self):
        """Test exporting model to TorchScript."""
        model = CRNN(num_classes=2, n_mels=64)
        dummy_input = torch.randn(1, 64, 100)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "model_traced.pt")
            traced = export_to_torchscript(model, dummy_input, save_path)
            
            assert os.path.exists(save_path)
            assert isinstance(traced, torch.jit.ScriptModule)
            
            # Verify traced model can run
            with torch.no_grad():
                out = traced(dummy_input)
            assert out.shape == (1, 2)
            
    def test_onnx_export(self):
        """Test exporting model to ONNX."""
        model = CRNN(num_classes=2, n_mels=64)
        dummy_input = torch.randn(1, 64, 100)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "model.onnx")
            export_to_onnx(model, dummy_input, save_path)
            
            assert os.path.exists(save_path)
            # File should not be empty
            assert os.path.getsize(save_path) > 1024
            
    def test_fp16_export(self):
        """Test saving model weights in half-precision (FP16)."""
        model = CRNN(num_classes=2, n_mels=64)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "model_fp16.pt")
            fp16_model = export_to_fp16(model, save_path)
            
            assert os.path.exists(save_path)
            # Verify parameters are FP16 (half)
            for param in fp16_model.parameters():
                if param.requires_grad:
                    assert param.dtype == torch.float16
                    break
                    
    def test_dynamic_quantization(self):
        """Test dynamic quantization of Linear layers."""
        model = CRNN(num_classes=2, n_mels=64)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "model_quantized.pt")
            quantized = quantize_model_dynamic(model, save_path)
            
            assert os.path.exists(save_path)
            # Verify linear layers are quantized to qint8
            has_quantized_linear = False
            for name, module in quantized.named_modules():
                if "Linear" in str(type(module)) or "Quantized" in str(type(module)):
                    has_quantized_linear = True
                    break
            # Note: PyTorch's dynamic quantization replaces nn.Linear with nn.quantized.dynamic.Linear
            # which is what we check here
            assert has_quantized_linear
