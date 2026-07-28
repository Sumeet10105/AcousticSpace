"""Audio amplitude normalization utilities."""

import logging
import torch

logger = logging.getLogger(__name__)


def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """Applies peak amplitude normalization to a waveform tensor.

    Scales the waveform so that its maximum absolute amplitude is equal to target_peak.

    Args:
        waveform: The input waveform tensor [channels, time].
        target_peak: The desired maximum absolute peak amplitude (default: 0.95).

    Returns:
        The peak-normalized waveform tensor.
    """
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        normalized = (waveform / max_val) * target_peak
        return normalized
    return waveform


def rms_normalize(waveform: torch.Tensor, target_db: float = -20.0) -> torch.Tensor:
    """Applies RMS (Root Mean Square) normalization to a waveform tensor.

    Scales the waveform so that its power level matches target_db decibels.

    Args:
        waveform: The input waveform tensor [channels, time].
        target_db: The desired RMS target level in dB (default: -20.0).

    Returns:
        The RMS-normalized waveform tensor.
    """
    # Calculate current RMS
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-9)
    # Calculate target RMS from target_db
    target_rms = 10.0 ** (target_db / 20.0)
    
    # Scale waveform
    normalized = waveform * (target_rms / rms)
    
    # Clip peaks to avoid digital clipping (keep in range [-0.99, 0.99])
    max_peak = torch.max(torch.abs(normalized))
    if max_peak > 0.99:
        normalized = normalized / max_peak * 0.99
        logger.debug(f"RMS normalized waveform clipped peaks from {max_peak:.4f} to 0.99.")
        
    return normalized
