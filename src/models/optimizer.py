"""Optimizer configuration utility for PyTorch models."""

import logging
import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def get_optimizer(
    model: torch.nn.Module, 
    opt_name: str = "adam", 
    lr: float = 1e-4, 
    weight_decay: float = 1e-5
) -> Optimizer:
    """Configures and returns the requested PyTorch optimizer.

    Args:
        model: PyTorch model whose parameters will be optimized.
        opt_name: Name of the optimizer ('adam', 'adamw', 'sgd').
        lr: Learning rate.
        weight_decay: Weight decay coefficient.

    Returns:
        The configured PyTorch Optimizer instance.
    """
    opt_name = opt_name.lower().strip()
    
    # Filter out parameters that do not require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            trainable_params, 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params, 
            lr=lr, 
            momentum=0.9, 
            weight_decay=weight_decay
        )
    else:
        logger.warning(f"Unrecognized optimizer '{opt_name}'. Defaulting to Adam.")
        optimizer = torch.optim.Adam(
            trainable_params, 
            lr=lr, 
            weight_decay=weight_decay
        )
        
    logger.info(f"Initialized '{opt_name}' optimizer with lr={lr}, weight_decay={weight_decay}.")
    return optimizer
