"""Evaluation metrics and training history plotting utilities."""

import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray, 
    classes: List[str] = ["REAL", "FAKE"],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots a confusion matrix using a beautiful Seaborn heatmap.

    Args:
        cm: Confusion matrix array [2, 2].
        classes: List of class labels (default: ["REAL", "FAKE"]).
        title: Title of the plot.
        save_path: Optional file path to automatically save the figure.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Custom color palette (cool warm or blues)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=classes, 
        yticklabels=classes,
        ax=ax,
        cbar=True,
        annot_kws={"size": 12, "weight": "bold"}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("True Label", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        try:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}.")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix: {str(e)}")
            
    return fig


def plot_roc_curve(
    fpr: np.ndarray, 
    tpr: np.ndarray, 
    auc_score: float,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots the ROC curve with AUC metric.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc_score: Area Under Curve.
        title: Title of the plot.
        save_path: Optional file path to automatically save the figure.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.plot(fpr, tpr, color='#00a8cc', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='#e43f5a', lw=1.5, linestyle='--', label='Random Guess')
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        try:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}.")
        except Exception as e:
            logger.error(f"Failed to save ROC curve: {str(e)}")
            
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots training vs validation history curves (loss and accuracy).

    Args:
        history: Dictionary containing lists for 'train_loss', 'val_loss',
                 'train_acc', 'val_acc' (or similar metric keys).
        save_path: Optional file path to automatically save the figure.

    Returns:
        The matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # 1. Loss Curves
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#00a8cc', lw=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#e43f5a', lw=2)
    ax1.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Metric (e.g. Accuracy or EER) Curves
    metric_key = 'train_acc' if 'train_acc' in history else ('train_eer' if 'train_eer' in history else None)
    val_metric_key = 'val_acc' if 'val_acc' in history else ('val_eer' if 'val_eer' in history else None)
    
    metric_name = 'Accuracy' if 'acc' in (metric_key or '') else 'EER'
    
    if metric_key and metric_key in history:
        ax2.plot(epochs, history[metric_key], label=f'Train {metric_name}', color='#00a8cc', lw=2)
    if val_metric_key and val_metric_key in history:
        ax2.plot(epochs, history[val_metric_key], label=f'Val {metric_name}', color='#e43f5a', lw=2)
        
    ax2.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax2.set_ylabel(metric_name, fontsize=11, fontweight='bold')
    ax2.set_title(f'Training & Validation {metric_name}', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        try:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}.")
        except Exception as e:
            logger.error(f"Failed to save training history: {str(e)}")
            
    return fig
