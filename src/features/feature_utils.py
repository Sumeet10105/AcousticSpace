"""Feature extraction utilities including alignment and standardization."""

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def align_temporal_dimension(
    feature_tensor: torch.Tensor, 
    target_len: int
) -> torch.Tensor:
    """Interpolates/resamples feature tensor to match target time length.

    Args:
        feature_tensor: Feature tensor of shape [channels, feature_dim, time_steps].
        target_len: Desired number of time steps.

    Returns:
        Resampled feature tensor of shape [channels, feature_dim, target_len].
    """
    current_len = feature_tensor.shape[2]
    if current_len == target_len:
        return feature_tensor
        
    # F.interpolate requires 3D/4D input: [batch, channels, length]
    # We treat channels as batch dim: [channels, feature_dim, current_len]
    resampled = F.interpolate(
        feature_tensor, 
        size=target_len, 
        mode="linear", 
        align_corners=False
    )
    
    return resampled


def standardize_features(
    feature_tensor: torch.Tensor, 
    epsilon: float = 1e-9
) -> torch.Tensor:
    """Standardizes features to zero mean and unit variance.

    Args:
        feature_tensor: Feature tensor [channels, feature_dim, time_steps].
        epsilon: Small epsilon to avoid division by zero.

    Returns:
        Standardized feature tensor.
    """
    mean = torch.mean(feature_tensor, dim=2, keepdim=True)
    std = torch.std(feature_tensor, dim=2, keepdim=True) + epsilon
    return (feature_tensor - mean) / std
