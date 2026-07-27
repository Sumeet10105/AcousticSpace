"""Feature fusion module for combining multi-modal acoustic descriptors."""

import logging
import torch
from typing import List, Optional

from src.features.feature_utils import align_temporal_dimension

logger = logging.getLogger(__name__)


def fuse_features(
    features_list: List[torch.Tensor],
    target_time_steps: Optional[int] = None
) -> torch.Tensor:
    """Fuses multiple acoustic features by aligning their time steps and concatenating them.

    Args:
        features_list: List of feature tensors, each of shape [channels, feature_dim, time_steps].
        target_time_steps: The target time length for interpolation. If None,
                           aligns to the maximum time length present in features_list.

    Returns:
        Concatenated/fused feature tensor of shape [channels, sum(feature_dims), target_time_steps].
    """
    if not features_list:
        logger.error("Features list is empty.")
        raise ValueError("Cannot fuse empty list of features.")
        
    if len(features_list) == 1:
        return features_list[0]
        
    # Determine target time length if not provided
    if target_time_steps is None:
        target_time_steps = max(feat.shape[2] for feat in features_list)
        
    aligned_features = []
    for feat in features_list:
        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat)
            
        aligned = align_temporal_dimension(feat, target_time_steps)
        aligned_features.append(aligned)
        
    # Concatenate along the feature dimension (dim=1)
    fused_tensor = torch.cat(aligned_features, dim=1)
    return fused_tensor
