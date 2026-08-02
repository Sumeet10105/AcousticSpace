"""Main model export script to package AcousticSpace deepfake audio models for production deployment."""

import argparse
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
from src.export import (
    export_to_torchscript,
    export_to_fp16,
    quantize_model_dynamic,
)

def main():
    """Main function to export models."""
    parser = argparse.ArgumentParser(description="Export model for deployment")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--export_format", type=str, default="torchscript",
                        choices=["torchscript", "fp16", "quantized"],
                        help="Export format")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Output path for the exported model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to main config file")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml",
                        help="Path to model config file")
    
    args = parser.parse_args()
    logger = get_logger(__name__)
    
    # Load configs
    config = Config(args.config)
    model_config = Config(args.model_config)
    
    # Initialize Model architecture
    model_name = model_config.get("model.name", "AST").upper().strip()
    num_classes = model_config.get("model.num_classes", 2)
    
    logger.info(f"Loading checkpoint weights from: {args.model_path}")
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
        
    # Load model weights
    if not os.path.exists(args.model_path):
        logger.error(f"Checkpoint weights file not found: {args.model_path}")
        sys.exit(1)
        
    checkpoint_data = torch.load(args.model_path, map_location="cpu")
    if "state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["state_dict"])
    elif "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)
        
    model.eval()
    
    # Create output directory
    outputs_dir = os.path.join(config.get("paths.outputs", "./outputs"), "exported")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Determine default save path if not provided
    save_path = args.save_path
    if not save_path:
        filename = f"{model_name.lower()}_exported"
        if args.export_format == "torchscript":
            save_path = os.path.join(outputs_dir, f"{filename}_traced.pt")
        elif args.export_format == "fp16":
            save_path = os.path.join(outputs_dir, f"{filename}_fp16.pt")
        elif args.export_format == "quantized":
            save_path = os.path.join(outputs_dir, f"{filename}_quantized.pt")
            
    # Setup dummy input for ONNX / TorchScript tracing
    # (batch_size=1, n_mels=64, time_frames=100)
    dummy_input = torch.randn(1, config.get("audio.n_mels", 64), 100)
    
    logger.info(f"Exporting model to {args.export_format} format...")
    try:
        if args.export_format == "torchscript":
            export_to_torchscript(model, dummy_input, save_path)
        elif args.export_format == "fp16":
            export_to_fp16(model, save_path)
        elif args.export_format == "quantized":
            quantize_model_dynamic(model, save_path)
            
        logger.info(f"Export completed! Exported file saved to: {save_path}")
    except Exception as e:
        logger.error(f"Model export failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
