"""Master trainer module for model training and validation loops."""

import logging
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Dict, Optional
from torch.utils.tensorboard import SummaryWriter

from src.models.callbacks import EarlyStopping
from src.models.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class supporting mixed precision training, gradient clipping, early stopping, and tensorboard."""
    
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module, 
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 10,
        tensorboard_dir: Optional[str] = None
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model instance.
            optimizer: PyTorch optimizer instance.
            criterion: Loss function module.
            scheduler: Learning rate scheduler.
            device: Device to train on ('cuda' or 'cpu').
            mixed_precision: Whether to use automatic mixed precision (AMP) (default: True).
            gradient_clip: Maximum norm for gradient clipping (default: 1.0).
            early_stopping_patience: Patience for early stopping (default: 10).
            tensorboard_dir: Directory to save TensorBoard logs.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision and (device == 'cuda' or device.startswith('cuda:'))
        self.gradient_clip = gradient_clip
        
        # Initialize AMP gradient scaler if mixed precision is enabled
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")
        
        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=tensorboard_dir) if tensorboard_dir else None
        
        logger.info(
            f"Trainer initialized on device='{device}'. "
            f"Mixed precision={self.mixed_precision}, gradient_clip={gradient_clip}."
        )
        
    def train_epoch(self, train_loader) -> float:
        """Trains the model for one epoch.

        Args:
            train_loader: Dataloader containing training samples.

        Returns:
            The average training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc="Training Batch", leave=False)):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                self.optimizer.step()
                
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        return avg_loss
        
    def validate(self, val_loader) -> Dict[str, float]:
        """Validates the model.

        Args:
            val_loader: Dataloader containing validation samples.

        Returns:
            Dictionary containing evaluation metrics (e.g., 'loss', 'accuracy').
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validation Batch", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(x)
                        loss = self.criterion(outputs, y)
                else:
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                    
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
    def fit(
        self, 
        train_loader, 
        val_loader, 
        epochs: int = 10,
        checkpoint_dir: str = "./checkpoints",
        resume_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs the complete training and validation cycle.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Total epochs to train.
            checkpoint_dir: Path to directory for saving model checkpoints.
            resume_path: Optional file path to resume training from.

        Returns:
            Dictionary containing metrics history.
        """
        start_epoch = 1
        best_val_loss = float('inf')
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'lr': []
        }
        
        # Resume training from checkpoint if provided
        if resume_path:
            checkpoint = load_checkpoint(
                resume_path, 
                self.model, 
                self.optimizer, 
                self.scheduler, 
                self.device
            )
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            history = checkpoint.get('history', history)
            logger.info(f"Resuming training from epoch {start_epoch}.")
            
        for epoch in range(start_epoch, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            
            # Train one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            
            # Step scheduler if applicable
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Record metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['lr'].append(current_lr)
            
            # Log to console and TensorBoard
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}")
            if self.tb_writer:
                self.tb_writer.add_scalar('Loss/Train', train_loss, epoch)
                self.tb_writer.add_scalar('Loss/Val', val_loss, epoch)
                self.tb_writer.add_scalar('Accuracy/Val', val_acc, epoch)
                self.tb_writer.add_scalar('LR', current_lr, epoch)
                
            # Checkpoint saving logic
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                
            checkpoint_state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_loss': best_val_loss,
                'history': history
            }
            
            latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_checkpoint(checkpoint_state, latest_path, is_best=is_best, best_filepath=best_path)
            
            # Check early stopping
            if self.early_stopping(val_loss):
                logger.warning(f"Early stopping triggered at epoch {epoch}.")
                break
                
        if self.tb_writer:
            self.tb_writer.close()
            
        return history
