"""Checkpoint utilities for saving and resuming model training states."""

import logging
import os
import torch
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any], 
    filepath: str, 
    is_best: bool = False, 
    best_filepath: Optional[str] = None
) -> None:
    """Saves the training checkpoint state (epoch, model state, optimizer state, metrics, etc.).

    Args:
        state: State dictionary to serialize.
        filepath: Target checkpoint file path.
        is_best: If True, also duplicates/saves to best_filepath.
        best_filepath: Optional path to save the best model weights.
    """
    try:
        # Create directory if it doesn't exist
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        torch.save(state, filepath)
        logger.info(f"Checkpoint saved to {filepath} at epoch {state.get('epoch', 0)}.")
        
        if is_best and best_filepath:
            if os.path.dirname(best_filepath):
                os.makedirs(os.path.dirname(best_filepath), exist_ok=True)
            torch.save(state, best_filepath)
            logger.info(f"Best checkpoint copied to {best_filepath}.")
            
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {str(e)}")
        raise e


def load_checkpoint(
    filepath: str, 
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    scheduler: Optional[Any] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Loads a training checkpoint state and restores model/optimizer/scheduler parameters.

    Args:
        filepath: Path to the checkpoint file.
        model: PyTorch model instance to restore.
        optimizer: Optional PyTorch optimizer to restore.
        scheduler: Optional learning rate scheduler to restore.
        device: Device to map tensors to.

    Returns:
        The loaded state dictionary containing epoch, best_metric, loss history, etc.
    """
    if not os.path.exists(filepath):
        logger.error(f"Checkpoint file not found: {filepath}")
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
    try:
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model weights
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback to direct state load
            model.load_state_dict(checkpoint)
            
        # Load optimizer state
        if optimizer is not None:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            elif 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                
        # Load scheduler state
        if scheduler is not None:
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                
        logger.info(f"Loaded checkpoint from {filepath} (resuming from epoch {checkpoint.get('epoch', 0)}).")
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filepath}: {str(e)}")
        raise e
