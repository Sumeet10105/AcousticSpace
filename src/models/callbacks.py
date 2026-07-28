"""Training loop callbacks, including Early Stopping and TensorBoard logging."""

import logging
import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to terminate training when a monitored metric stops improving."""
    
    def __init__(
        self, 
        patience: int = 10, 
        mode: str = "min", 
        min_delta: float = 0.0,
        enabled: bool = True
    ):
        """Initialize EarlyStopping.

        Args:
            patience: Number of epochs to wait without improvement before stopping (default: 10).
            mode Monitored metric direction: 'min' (e.g. loss) or 'max' (e.g. accuracy).
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            enabled: Whether early stopping is enabled.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.enabled = enabled
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_val: float) -> bool:
        """Checks if training should stop.

        Args:
            current_val: Monitored metric value for the current epoch.

        Returns:
            True if training should early-stop, False otherwise.
        """
        if not self.enabled:
            return False
            
        # Convert metric to a score where higher is always better
        if self.mode == "min":
            score = -current_val
        else:
            score = current_val
            
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning("Early stopping triggered. Terminating training.")
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop
