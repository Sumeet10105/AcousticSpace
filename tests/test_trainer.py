"""Tests for the training pipeline module."""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models import (
    CRNN,
    Trainer,
    get_optimizer,
    get_scheduler,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
)


@pytest.fixture
def dummy_data():
    """Generates dummy features and binary labels for training tests."""
    torch.manual_seed(42)
    # 10 samples, 64 features, 100 time frames
    x = torch.randn(10, 64, 100)
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return loader


class TestTrainingPipeline:
    """Trainer, optimizer, scheduler, and callbacks tests."""
    
    def test_optimizer_retrieval(self):
        """Test optimizer configuration utility."""
        model = CRNN(num_classes=2)
        opt_adam = get_optimizer(model, "adam", lr=1e-3)
        assert isinstance(opt_adam, torch.optim.Adam)
        
        opt_adamw = get_optimizer(model, "adamw", lr=1e-3)
        assert isinstance(opt_adamw, torch.optim.AdamW)
        
        opt_sgd = get_optimizer(model, "sgd", lr=1e-3)
        assert isinstance(opt_sgd, torch.optim.SGD)
        
    def test_scheduler_retrieval(self):
        """Test scheduler configuration utility."""
        model = CRNN(num_classes=2)
        optimizer = get_optimizer(model, "adam")
        
        sched_cosine = get_scheduler(optimizer, "cosine", epochs=10)
        assert isinstance(sched_cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        sched_plateau = get_scheduler(optimizer, "plateau")
        assert isinstance(sched_plateau, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        sched_step = get_scheduler(optimizer, "step")
        assert isinstance(sched_step, torch.optim.lr_scheduler.StepLR)
        
    def test_early_stopping(self):
        """Test EarlyStopping callback functionality."""
        early_stopping = EarlyStopping(patience=3, mode="min")
        
        # Test no early stop with improving values
        assert early_stopping(1.0) is False
        assert early_stopping(0.9) is False
        
        # Test patience counting and triggering
        assert early_stopping(0.95) is False  # Counter = 1
        assert early_stopping(0.96) is False  # Counter = 2
        assert early_stopping(0.97) is True   # Counter = 3 -> Trigger!
        
    def test_trainer_fit_and_checkpointing(self, dummy_data):
        """Test Trainer class fit loop and checkpoint saving/resuming."""
        model = CRNN(num_classes=2, n_mels=64)
        optimizer = get_optimizer(model, "adam", lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = get_scheduler(optimizer, "step")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                device="cpu",
                mixed_precision=False,
                gradient_clip=1.0,
                early_stopping_patience=3,
                tensorboard_dir=os.path.join(temp_dir, "tb")
            )
            
            # Train for 2 epochs
            history = trainer.fit(
                train_loader=dummy_data,
                val_loader=dummy_data,
                epochs=2,
                checkpoint_dir=temp_dir
            )
            
            assert len(history['train_loss']) == 2
            assert os.path.exists(os.path.join(temp_dir, "latest_checkpoint.pt"))
            assert os.path.exists(os.path.join(temp_dir, "best_model.pt"))
            
            # Test resuming from checkpoint
            resume_path = os.path.join(temp_dir, "latest_checkpoint.pt")
            new_model = CRNN(num_classes=2, n_mels=64)
            new_optimizer = get_optimizer(new_model, "adam", lr=1e-3)
            new_scheduler = get_scheduler(new_optimizer, "step")
            
            new_trainer = Trainer(
                model=new_model,
                optimizer=new_optimizer,
                criterion=criterion,
                scheduler=new_scheduler,
                device="cpu",
                mixed_precision=False
            )
            
            # This should load weights, optimizer state, epoch history, and start at epoch 3
            history_resume = new_trainer.fit(
                train_loader=dummy_data,
                val_loader=dummy_data,
                epochs=3,  # Train 1 more epoch
                checkpoint_dir=temp_dir,
                resume_path=resume_path
            )
            
            # Total epoch length should be 3 in history
            assert len(history_resume['train_loss']) == 3
