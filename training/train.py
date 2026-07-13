"""Main training script."""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Setup Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_config
from src.core.logger import get_logger
from src.utils.seed import set_seed


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train audio spoofing detection model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    logger = get_logger(__name__, log_file=config.get("logging.log_file"))
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    # TODO: Implement training pipeline
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
