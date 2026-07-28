"""Main training script executing the end-to-end training pipeline."""

import argparse
import os
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

# Setup Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config
from src.core.logger import get_logger
from src.utils.seed import set_seed
from src.data.dataset import AudioSpoofingDataset
from src.data.dataloader import create_dataloader
from src.models import (
    ResNetCNN,
    CRNN,
    AudioSpectrogramTransformer,
    Trainer,
    get_optimizer,
    get_scheduler,
)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train audio spoofing detection model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to main config file")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml",
                        help="Path to training config file")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml",
                        help="Path to model architecture config file")
    
    args = parser.parse_args()
    
    # Load configurations
    config = Config(args.config)
    train_config = Config(args.train_config)
    model_config = Config(args.model_config)
    
    # Setup logger
    log_file = config.get("logging.log_file", "./outputs/logs/training.log")
    logger = get_logger(__name__, log_file=log_file)
    
    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)
    
    logger.info("Initializing AcousticSpace Training Pipeline...")
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Loaded training config: {args.train_config}")
    logger.info(f"Loaded model config: {args.model_config}")
    
    # Resolve device
    device = train_config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU training.")
        device = "cpu"
        
    # Get paths
    dataset_root = config.get("paths.dataset_root", "./datasets")
    metadata_path = os.path.join(config.get("paths.metadata", "./datasets/metadata"), "manifest.csv")
    checkpoint_dir = config.get("paths.checkpoints", "./saved_models/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check if metadata manifest exists
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata manifest not found at {metadata_path}. Please generate it first.")
        sys.exit(1)
        
    logger.info(f"Loading datasets using manifest at {metadata_path}...")
    
    # Create datasets
    try:
        train_dataset = AudioSpoofingDataset(
            manifest_path=metadata_path,
            split="train",
            target_sr=config.get("dataset.sample_rate", 16000),
            duration_seconds=config.get("dataset.duration", 4.0),
            augment=train_config.get("augmentation.enabled", True)
        )
        
        val_dataset = AudioSpoofingDataset(
            manifest_path=metadata_path,
            split="val",
            target_sr=config.get("dataset.sample_rate", 16000),
            duration_seconds=config.get("dataset.duration", 4.0),
            augment=False
        )
        
        # Create DataLoader helper instances
        batch_size = train_config.get("training.batch_size", 32)
        num_workers = train_config.get("num_workers", 4)
        pin_memory = train_config.get("pin_memory", True)
        
        train_loader = create_dataloader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        
        val_loader = create_dataloader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        
        logger.info(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
        
    except Exception as e:
        logger.error(f"Failed to load dataloaders: {str(e)}")
        sys.exit(1)
        
    # Instantiate Model
    model_name = model_config.get("model.name", "AST").upper().strip()
    num_classes = model_config.get("model.num_classes", 2)
    pretrained = model_config.get("model.pretrained", True)
    
    logger.info(f"Initializing {model_name} architecture...")
    if model_name == "AST":
        model = AudioSpectrogramTransformer(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "CRNN":
        model = CRNN(
            num_classes=num_classes, 
            n_mels=config.get("audio.n_mels", 64),
            rnn_hidden_dim=model_config.get("model.embedding_dim", 128)
        )
    elif model_name == "CNN":
        model = ResNetCNN(num_classes=num_classes, pretrained=pretrained)
    else:
        logger.warning(f"Unknown model name '{model_name}'. Defaulting to ResNetCNN.")
        model = ResNetCNN(num_classes=num_classes, pretrained=pretrained)
        
    # Configure optimizer and scheduler
    optimizer = get_optimizer(
        model, 
        opt_name=train_config.get("training.optimizer", "adam"),
        lr=float(train_config.get("training.learning_rate", 1e-4)),
        weight_decay=float(train_config.get("training.weight_decay", 1e-5))
    )
    
    epochs = train_config.get("training.epochs", 10)
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=train_config.get("training.scheduler", "cosine"),
        epochs=epochs,
        warmup_epochs=train_config.get("training.warmup_epochs", 5)
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        mixed_precision=train_config.get("mixed_precision", True),
        gradient_clip=float(train_config.get("training.gradient_clip", 1.0)),
        early_stopping_patience=train_config.get("validation.early_stopping.patience", 10),
        tensorboard_dir=os.path.join(config.get("paths.outputs", "./outputs"), "tensorboard")
    )
    
    # Run fit cycle
    logger.info("Starting training loop...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir
    )
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
