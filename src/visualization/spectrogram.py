"""Spectrogram and feature map visualization utilities with overlay capabilities."""

import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Union

logger = logging.getLogger(__name__)


def plot_spectrogram(
    spec: Union[torch.Tensor, np.ndarray], 
    sr: int = 16000,
    title: str = "Spectrogram",
    ylabel: str = "Frequency (Hz)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots a linear or mel spectrogram with a rich colormap.

    Args:
        spec: Spectrogram tensor or array of shape [channels, freq_bins, time_steps] or [freq_bins, time_steps].
        sr: Sample rate in Hz (default: 16000).
        title: Title of the plot.
        ylabel: Label for the Y-axis (default: "Frequency (Hz)").
        save_path: Optional file path to automatically save the figure.

    Returns:
        The matplotlib Figure object.
    """
    if isinstance(spec, torch.Tensor):
        spec_np = spec.detach().cpu().numpy()
    else:
        spec_np = np.asarray(spec)
        
    if len(spec_np.shape) == 2:
        spec_np = np.expand_dims(spec_np, axis=0)
        
    channels, freq_bins, time_steps = spec_np.shape
    
    fig, axes = plt.subplots(channels, 1, figsize=(10, 4 * channels), squeeze=False)
    
    for c in range(channels):
        ax = axes[c, 0]
        # Display spectrogram with 'magma' colormap
        im = ax.imshow(
            spec_np[c], 
            aspect='auto', 
            origin='lower', 
            cmap='magma', 
            interpolation='nearest'
        )
        ax.set_title(f"{title} - Channel {c}" if channels > 1 else title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, format="%+2.0f dB" if "dB" in title or "Mel" in title else None)
        
    axes[-1, 0].set_xlabel("Time Frames", fontsize=10, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        try:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spectrogram plot saved to {save_path}.")
        except Exception as e:
            logger.error(f"Failed to save spectrogram plot to {save_path}: {str(e)}")
            
    return fig


def plot_mel_spectrogram(
    mel_spec: Union[torch.Tensor, np.ndarray],
    sr: int = 16000,
    title: str = "Mel Spectrogram",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots a Mel spectrogram."""
    return plot_spectrogram(mel_spec, sr=sr, title=title, ylabel="Mel Bin", save_path=save_path)


def plot_mfcc(
    mfcc: Union[torch.Tensor, np.ndarray],
    title: str = "MFCC Coefficients",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots MFCC coefficients."""
    return plot_spectrogram(mfcc, sr=16000, title=title, ylabel="Coefficients", save_path=save_path)


def plot_attention_overlay(
    spec: Union[torch.Tensor, np.ndarray],
    attention_map: Union[torch.Tensor, np.ndarray],
    title: str = "Explainable AI - Attention Overlay",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Overlays an attention map or Grad-CAM heatmap over the spectrogram.

    Args:
        spec: Base spectrogram tensor/array of shape [freq_bins, time_steps] or [1, freq_bins, time_steps].
        attention_map: Attention heatmap tensor/array of shape [freq_bins, time_steps] or [time_steps] (which will be tiled).
        title: Title of the plot.
        save_path: Optional file path to automatically save the figure.

    Returns:
        The matplotlib Figure object.
    """
    if isinstance(spec, torch.Tensor):
        spec_np = spec.detach().cpu().numpy()
    else:
        spec_np = np.asarray(spec)
        
    if isinstance(attention_map, torch.Tensor):
        att_np = attention_map.detach().cpu().numpy()
    else:
        att_np = np.asarray(attention_map)
        
    # Clean shapes
    if len(spec_np.shape) == 3:
        spec_np = spec_np[0]  # Take first channel
        
    freq_bins, time_steps = spec_np.shape
    
    # If attention map is 1D (e.g. attention per frame), tile it to 2D
    if len(att_np.shape) == 1:
        att_np = np.tile(att_np, (freq_bins, 1))
        
    # Resize attention map to match spectrogram shape if needed
    if att_np.shape != spec_np.shape:
        import scipy.ndimage
        zoom_factors = (spec_np.shape[0] / att_np.shape[0], spec_np.shape[1] / att_np.shape[1])
        att_np = scipy.ndimage.zoom(att_np, zoom_factors, order=1)
        
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot background spectrogram
    ax.imshow(spec_np, aspect='auto', origin='lower', cmap='gray', alpha=0.9)
    
    # Plot semi-transparent overlay
    # Normalize attention map for overlay colors
    att_norm = (att_np - att_np.min()) / (att_np.max() - att_np.min() + 1e-9)
    
    im = ax.imshow(att_norm, aspect='auto', origin='lower', cmap='jet', alpha=0.45)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Frequency / Mel Bins", fontsize=11, fontweight='bold')
    ax.set_xlabel("Time Frames", fontsize=11, fontweight='bold')
    
    fig.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    
    if save_path:
        try:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention overlay saved to {save_path}.")
        except Exception as e:
            logger.error(f"Failed to save attention overlay plot: {str(e)}")
            
    return fig
