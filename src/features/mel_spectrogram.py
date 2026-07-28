"""Mel spectrogram feature extraction."""

import logging
import torch
import torchaudio
from typing import Optional

logger = logging.getLogger(__name__)


def compute_mel_spectrogram(
    audio: torch.Tensor, 
    sample_rate: int = 16000,
    n_fft: int = 512, 
    hop_length: int = 160,
    n_mels: int = 64,
    f_min: float = 0.0,
    f_max: Optional[float] = None
) -> torch.Tensor:
    """Computes log-mel spectrogram from an audio waveform tensor.

    Args:
        audio: Audio waveform tensor of shape [channels, time].
        sample_rate: Audio sampling rate in Hz (default: 16000).
        n_fft: FFT size (default: 512).
        hop_length: Hop length (default: 160).
        n_mels: Number of Mel frequency bands (default: 64).
        f_min: Minimum frequency (default: 0.0).
        f_max: Maximum frequency (default: None).

    Returns:
        Log-mel spectrogram tensor of shape [channels, n_mels, time_steps].
    """
    try:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        ).to(audio.device)
        
        # Compute mel spectrogram
        mel_spec = mel_transform(audio)
        
        # Convert to power decibels (log-scale)
        log_mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec, 
            multiplier=10.0, 
            amin=1e-10, 
            db_multiplier=0.0
        )
        return log_mel_spec
        
    except Exception as e:
        logger.error(f"Mel spectrogram computation failed: {str(e)}")
        raise ValueError(f"Mel spectrogram computation failed: {str(e)}")
