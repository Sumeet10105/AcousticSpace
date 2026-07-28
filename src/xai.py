"""Explainable AI (XAI) module for audio deepfake classification models."""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN/CRNN models."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """Initialize Grad-CAM.

        Args:
            model: PyTorch model instance.
            target_layer: The convolutional layer to inspect.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        self.activations = output
        
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, x: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """Generates the Grad-CAM heatmap for class_idx.

        Args:
            x: Input feature tensor [batch, channel, freq, time] or [batch, freq, time].
            class_idx: Class index (default: 1 for FAKE).

        Returns:
            2D numpy array heatmap of the same resolution as the conv layer activations.
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(x)
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Check if hooks captured tensors
        if self.gradients is None or self.activations is None:
            logger.warning("Grad-CAM hooks failed to capture gradients or activations. Returning dummy map.")
            return np.ones((x.shape[-2], x.shape[-1]))
            
        # Global average pooling of gradients
        # shape of gradients: [batch, channels, height, width]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to keep only positive contributions
        cam = torch.clamp(cam, min=0.0)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-9)
        
        # Convert to numpy and remove batch/channel dimensions
        heatmap = cam.squeeze().detach().cpu().numpy()
        return heatmap
        
    def remove_hooks(self):
        """Removes the forward and backward hooks from the target layer."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def compute_integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int = 1,
    steps: int = 20,
    device: str = "cpu"
) -> np.ndarray:
    """Computes Integrated Gradients attribution for the input features.

    Args:
        model: PyTorch model.
        x: Input feature tensor [batch, channels, freq, time].
        target_class: Target class for attribution.
        steps: Number of linear interpolation steps (default: 20).
        device: Device to run on.

    Returns:
        Numpy array of attributions of the same shape as x.
    """
    model.eval()
    x = x.to(device)
    
    # Baseline is zero features
    baseline = torch.zeros_like(x).to(device)
    
    # Generate scaled inputs along the path
    scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(steps + 1)]
    
    grads = []
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        logits = model(scaled_input)
        
        model.zero_grad()
        loss = logits[0, target_class]
        loss.backward()
        
        grads.append(scaled_input.grad.data.cpu().numpy())
        
    # Average the gradients
    avg_grads = np.mean(np.concatenate(grads, axis=0), axis=0, keepdims=True)
    
    # Attribution = (input - baseline) * average_gradients
    delta = (x - baseline).cpu().numpy()
    attributions = delta * avg_grads
    
    return attributions[0]


def compute_attention_rollout(
    attention_matrices: List[torch.Tensor],
    discard_ratio: float = 0.2,
    head_fusion: str = "max"
) -> np.ndarray:
    """Computes attention rollout map for ViT / AST models.

    Args:
        attention_matrices: List of self-attention weight tensors from each layer.
                            Each has shape [batch, heads, sequence_len, sequence_len].
        discard_ratio: Ratio of lowest attention weights to discard for noise reduction.
        head_fusion: How to fuse attention heads ('max', 'mean', 'min').

    Returns:
        1D or 2D attention weights over the sequence tokens.
    """
    # Rollout calculation:
    # A_rollout = I
    # For each layer L:
    # A = fuse_heads(attention_matrix)
    # A = (1 - factor) * I + factor * A
    # A_rollout = A * A_rollout
    
    num_layers = len(attention_matrices)
    if num_layers == 0:
        return np.ones((1, 1))
        
    batch_size, num_heads, seq_len, _ = attention_matrices[0].shape
    rollout = torch.eye(seq_len)
    
    for attn_weights in attention_matrices:
        # Fuse heads
        if head_fusion == "max":
            attn_fused, _ = torch.max(attn_weights, dim=1)
        elif head_fusion == "mean":
            attn_fused = torch.mean(attn_weights, dim=1)
        else:
            attn_fused = torch.mean(attn_weights, dim=1)
            
        attn_fused = attn_fused.squeeze(0)  # Remove batch dim
        
        # Discard lowest attention weights to reduce noise
        if discard_ratio > 0.0:
            flat_attn = attn_fused.flatten()
            val, _ = torch.topk(flat_attn, int(len(flat_attn) * discard_ratio), largest=False)
            if len(val) > 0:
                threshold = val[-1]
                attn_fused[attn_fused < threshold] = 0.0
                
        # Self-attention identity connection (residual)
        identity = torch.eye(seq_len, device=attn_fused.device)
        attn_layer = 0.5 * attn_fused + 0.5 * identity
        
        # Normalize columns
        attn_layer = attn_layer / (torch.sum(attn_layer, dim=-1, keepdim=True) + 1e-9)
        
        # Rollout matrix multiplication
        rollout = torch.matmul(attn_layer, rollout.to(attn_layer.device))
        
    # Take attention of the CLS token (token 0) to all other tokens
    cls_attention = rollout[0, 1:]  # Exclude CLS token attention to itself
    
    # Normalize between 0 and 1
    cls_attention = cls_attention - torch.min(cls_attention)
    cls_attention = cls_attention / (torch.max(cls_attention) + 1e-9)
    
    return cls_attention.detach().cpu().numpy()
