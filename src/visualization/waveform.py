"""Waveform visualization utilities using a modern aesthetic design system."""

import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Union

logger = logging.getLogger(__name__)


def plot_waveform(
    audio: Union[torch.Tensor, np.ndarray], 
    sr: int = 16000,
    title: str = "Audio Waveform",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots an audio waveform with a professional, clean theme.

    Args:
        audio: Audio waveform tensor or array [channels, time] or [time].
        sr: Sample rate in Hz (default: 16000).
        title: Title of the plot (default: "Audio Waveform").
        save_path: Optional file path to automatically save the figure.

    Returns:
        The matplotlib Figure object.
    """
    # Standardize input to numpy array [channels, time]
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = np.asarray(audio)
        
    if len(audio_np.shape) == 1:
        audio_np = np.expand_dims(audio_np, axis=0)
        
    channels, num_samples = audio_np.shape
    time = np.arange(0, num_samples) / sr
    
    # Modern dark mode theme setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(channels, 1, figsize=(10, 3 * channels), sharex=True, squeeze=False)
    
    # Beautiful color palette (Sleek tech blue/cyan)
    wave_color = '#00a8cc'
    
    for c in range(channels):
        ax = axes[c, 0]
        ax.plot(time, audio_np[c], color=wave_color, linewidth=1.0, alpha=0.85)
        ax.set_ylabel("Amplitude", fontsize=10, fontweight='bold', color='#333333')
        ax.set_title(f"{title} - Channel {c}" if channels > 1 else title, fontsize=12, fontweight='bold', color='#111111')
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    axes[-1, 0].set_xlabel("Time (seconds)", fontsize=10, fontweight='bold', color='#333333')
    plt.tight_layout()
    
    if save_path:
        try:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waveform plot saved successfully to {save_path}.")
        except Exception as e:
            logger.error(f"Failed to save waveform plot to {save_path}: {str(e)}")
            
    return fig
