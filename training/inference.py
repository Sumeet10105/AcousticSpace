"""Inference script."""

import argparse
from pathlib import Path

import torch

# Setup Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logger import get_logger


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference on audio files")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--audio_file", type=str, required=True,
                        help="Path to audio file")
    
    args = parser.parse_args()
    logger = get_logger(__name__)
    
    logger.info(f"Running inference on: {args.audio_file}")
    
    # TODO: Implement inference pipeline
    
    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
