"""Main evaluation script evaluating model on the test dataset split and generating visualization reports."""

import argparse
import os
from pathlib import Path
import sys
import torch

# Setup Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config
from src.core.logger import get_logger
from src.data.dataset import AudioSpoofingDataset
from src.data.dataloader import create_dataloader
from src.models import (
    ResNetCNN,
    CRNN,
    AudioSpectrogramTransformer,
    Evaluator,
)
from src.visualization.plots import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate audio spoofing detection model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to main config file")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml",
                        help="Path to model architecture config file")
    parser.add_argument("--checkpoint", type=str, default="./saved_models/checkpoints/best_model.pt",
                        help="Path to checkpoint weights file")
    
    args = parser.parse_args()
    
    # Load configurations
    config = Config(args.config)
    model_config = Config(args.model_config)
    
    # Setup logger
    log_file = config.get("logging.log_file", "./outputs/logs/evaluation.log")
    logger = get_logger(__name__, log_file=log_file)
    
    logger.info("Initializing AcousticSpace Evaluation Pipeline...")
    
    # Resolve device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Instantiate Model structure
    model_name = model_config.get("model.name", "AST").upper().strip()
    num_classes = model_config.get("model.num_classes", 2)
    
    logger.info(f"Initializing model architecture: {model_name}")
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
        
    # Load model weights from checkpoint
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint weights not found at: {args.checkpoint}")
        sys.exit(1)
        
    logger.info(f"Loading checkpoint weights from: {args.checkpoint}")
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["state_dict"])
    elif "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)
        
    # Get paths
    metadata_path = os.path.join(config.get("paths.metadata", "./datasets/metadata"), "manifest.csv")
    outputs_dir = config.get("paths.outputs", "./outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata manifest not found at: {metadata_path}")
        sys.exit(1)
        
    # Create test dataset
    try:
        test_dataset = AudioSpoofingDataset(
            manifest_path=metadata_path,
            split="test",
            target_sr=config.get("dataset.sample_rate", 16000),
            duration_seconds=config.get("dataset.duration", 4.0),
            augment=False
        )
        
        test_loader = create_dataloader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        
        logger.info(f"Loaded test dataset: {len(test_dataset)} samples.")
        
    except Exception as e:
        logger.error(f"Failed to load test dataset: {str(e)}")
        sys.exit(1)
        
    # Evaluate model
    evaluator = Evaluator(model=model, device=device)
    logger.info("Evaluating model on test dataset split...")
    results = evaluator.evaluate(test_loader)
    
    # Save visualizations
    logger.info("Generating evaluation curves...")
    
    # Plot and save ROC curve
    from src.metrics.roc import compute_roc_auc
    fpr, tpr, auc_score = compute_roc_auc(results["y_true"], results["y_prob"])
    roc_plot_path = os.path.join(outputs_dir, "roc_curve.png")
    plot_roc_curve(fpr, tpr, auc_score, save_path=roc_plot_path)
    logger.info(f"Saved ROC curve plot to: {roc_plot_path}")
    
    # Plot and save PR curve
    from src.metrics.roc import compute_precision_recall
    precision, recall, ap = compute_precision_recall(results["y_true"], results["y_prob"])
    pr_plot_path = os.path.join(outputs_dir, "precision_recall_curve.png")
    plot_precision_recall_curve(recall, precision, ap, save_path=pr_plot_path)
    logger.info(f"Saved Precision-Recall curve plot to: {pr_plot_path}")
    
    # Plot and save Confusion Matrix
    cm_plot_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], classes=["REAL", "FAKE"], save_path=cm_plot_path)
    logger.info(f"Saved Confusion Matrix plot to: {cm_plot_path}")
    
    logger.info(
        f"Evaluation finished! Final Scores: "
        f"Accuracy={results['accuracy']:.4f}, EER={results['eer']:.4f} "
        f"at threshold {results['eer_threshold']:.4f}, AUC={results['auc']:.4f}."
    )

if __name__ == "__main__":
    main()
