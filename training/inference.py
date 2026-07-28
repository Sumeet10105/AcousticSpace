"""Main inference script running deepfake audio prediction on files or directories."""

import argparse
import json
import os
from pathlib import Path
import sys
import torch

# Setup Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config
from src.core.logger import get_logger
from src.models import (
    ResNetCNN,
    CRNN,
    AudioSpectrogramTransformer,
)
from src.predict import predict_single_file, predict_directory

def main():
    """Main inference script function."""
    parser = argparse.ArgumentParser(description="Run inference on audio files")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--audio_file", type=str, default=None,
                        help="Path to single audio file")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Path to directory containing audio files")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml",
                        help="Path to model config file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold")
    
    args = parser.parse_args()
    logger = get_logger(__name__)
    
    if args.audio_file is None and args.audio_dir is None:
        logger.error("You must specify either --audio_file or --audio_dir.")
        sys.exit(1)
        
    # Load configurations
    config = Config(args.config)
    model_config = Config(args.model_config)
    
    # Resolve device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize Model architecture
    model_name = model_config.get("model.name", "AST").upper().strip()
    num_classes = model_config.get("model.num_classes", 2)
    
    logger.info(f"Loading {model_name} model architecture...")
    if model_name == "AST":
        model = AudioSpectrogramTransformer(num_classes=num_classes, pretrained=False)
    elif model_name == "CRNN":
        model = CRNN(
            num_classes=num_classes, 
            n_mels=config.get("audio.n_mels", 64),
            rnn_hidden_dim=model_config.get("model.embedding_dim", 128)
        )
    elif model_name == "CNN":
        model = ResNetCNN(num_classes=num_classes, pretrained=False)
    else:
        model = ResNetCNN(num_classes=num_classes, pretrained=False)
        
    # Load model checkpoint
    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint not found at: {args.model_path}")
        sys.exit(1)
        
    logger.info(f"Loading weights from {args.model_path}...")
    checkpoint_data = torch.load(args.model_path, map_location=device)
    if "state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["state_dict"])
    elif "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)
        
    model.eval()
    
    duration = config.get("dataset.duration", 4.0)
    
    # Run prediction
    if args.audio_file:
        logger.info(f"Running inference on file: {args.audio_file}")
        try:
            result = predict_single_file(
                audio_path=args.audio_file,
                model=model,
                device=device,
                threshold=args.threshold,
                duration_seconds=duration
            )
            # Remove raw features and numpy arrays for clean JSON print
            print_result = {
                "filepath": result["filepath"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "probability": result["probability"],
                "latency_ms": result["latency_ms"],
                "memory_used_mb": result["memory_used_mb"]
            }
            print(json.dumps(print_result, indent=2))
        except Exception as e:
            logger.error(f"Inference failed for {args.audio_file}: {str(e)}")
            sys.exit(1)
            
    elif args.audio_dir:
        logger.info(f"Running inference on directory: {args.audio_dir}")
        try:
            summary = predict_directory(
                directory_path=args.audio_dir,
                model=model,
                device=device,
                threshold=args.threshold,
                duration_seconds=duration
            )
            # Clean up predicted list for printout
            cleaned_predictions = []
            for pred in summary["predictions"]:
                cleaned_predictions.append({
                    "filepath": pred["filepath"],
                    "prediction": pred["prediction"],
                    "confidence": pred["confidence"],
                    "probability": pred["probability"],
                    "latency_ms": pred["latency_ms"],
                    "memory_used_mb": pred["memory_used_mb"]
                })
            summary["predictions"] = cleaned_predictions
            print(json.dumps(summary, indent=2))
        except Exception as e:
            logger.error(f"Inference directory scan failed for {args.audio_dir}: {str(e)}")
            sys.exit(1)
            
    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    main()
