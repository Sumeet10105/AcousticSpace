"""Learning rate scheduler configuration utility for PyTorch optimizers."""

import logging
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6
) -> Optional[Any]:
    """Configures and returns the requested learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_name: Name of the scheduler ('cosine', 'plateau', 'step', 'none').
        epochs: Total number of training epochs (used for CosineAnnealing).
        warmup_epochs: Number of warmup epochs.
        min_lr: Minimum learning rate (used for CosineAnnealing).

    Returns:
        The configured scheduler instance, or None if 'none' is specified.
    """
    scheduler_name = scheduler_name.lower().strip()
    
    if scheduler_name == "cosine":
        # Using CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs - warmup_epochs, 
            eta_min=min_lr
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=min_lr
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10, 
            gamma=0.5
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        logger.warning(f"Unrecognized scheduler '{scheduler_name}'. Defaulting to None.")
        scheduler = None
        
    if scheduler:
        logger.info(f"Initialized '{scheduler_name}' learning rate scheduler.")
    return scheduler
