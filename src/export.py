"""Model export utilities (TorchScript, ONNX, FP16, and Quantization) for AcousticSpace."""

import logging
import os
import torch
import torch.nn as nn
from typing import Union

logger = logging.getLogger(__name__)


def export_to_torchscript(
    model: nn.Module, 
    dummy_input: torch.Tensor, 
    save_path: str
) -> torch.jit.ScriptModule:
    """Exports a model to TorchScript format using tracing.

    Args:
        model: Loaded PyTorch model instance.
        dummy_input: Sample input tensor matching expected forward shape.
        save_path: Destination path for the .pt/.pts TorchScript file.

    Returns:
        The traced TorchScript model.
    """
    model.eval()
    try:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(save_path)
        logger.info(f"Model successfully exported to TorchScript at {save_path}.")
        return traced_model
    except Exception as e:
        logger.error(f"TorchScript export failed: {str(e)}")
        raise e


def export_to_onnx(
    model: nn.Module, 
    dummy_input: torch.Tensor, 
    save_path: str,
    opset_version: int = 14
) -> None:
    """Exports a model to ONNX format.

    Args:
        model: Loaded PyTorch model instance.
        dummy_input: Sample input tensor.
        save_path: Destination path for the .onnx file.
        opset_version: ONNX opset version (default: 14).
    """
    model.eval()
    try:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"Model successfully exported to ONNX at {save_path}.")
    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}")
        raise e


def export_to_fp16(
    model: nn.Module, 
    save_path: str
) -> nn.Module:
    """Saves a half-precision (FP16) version of the model.

    Args:
        model: Loaded PyTorch model.
        save_path: Destination checkpoint path.

    Returns:
        The half-precision model.
    """
    model.eval()
    try:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        fp16_model = model.half()
        torch.save(fp16_model.state_dict(), save_path)
        logger.info(f"FP16 model weights successfully saved to {save_path}.")
        return fp16_model
    except Exception as e:
        logger.error(f"FP16 conversion failed: {str(e)}")
        raise e


def quantize_model_dynamic(
    model: nn.Module, 
    save_path: str
) -> nn.Module:
    """Applies post-training dynamic quantization (quantizing nn.Linear layers to int8).

    Reduces model size and latency on CPU.

    Args:
        model: Loaded PyTorch model.
        save_path: Destination path for quantized weights.

    Returns:
        The quantized model.
    """
    model.eval()
    try:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        # Quantize Linear layers dynamically
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        torch.save(quantized_model.state_dict(), save_path)
        logger.info(f"Dynamically quantized model successfully saved to {save_path}.")
        return quantized_model
    except Exception as e:
        logger.error(f"Dynamic quantization failed: {str(e)}")
        raise e
