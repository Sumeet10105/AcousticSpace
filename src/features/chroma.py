"""Chroma features extraction."""

import logging
import torch
import librosa
import numpy as np

logger = logging.getLogger(__name__)


def extract_chroma(
    audio: torch.Tensor, 
    sample_rate: int = 16000, 
    n_fft: int = 512, 
    hop_length: int = 160,
    n_chroma: int = 12
) -> torch.Tensor:
    """Extracts Chroma STFT features from an audio waveform tensor.

    Args:
        audio: Audio waveform tensor of shape [channels, time].
        sample_rate: Audio sampling rate in Hz.
        n_fft: FFT size.
        hop_length: Hop length.
        n_chroma: Number of chroma bins (default: 12).

    Returns:
        Chroma features tensor of shape [channels, n_chroma, time_steps].
    """
    channels = audio.shape[0]
    device = audio.device
    audio_np = audio.cpu().numpy()
    
    channels_chroma = []
    
    for c in range(channels):
        y = audio_np[c]
        chroma = librosa.feature.chroma_stft(
            y=y, 
            sr=sample_rate, 
            n_fft=n_fft, 
            hop_length=hop_length,
            n_chroma=n_chroma
        )
        channels_chroma.append(torch.from_numpy(chroma).float())
        
    return torch.stack(channels_chroma, dim=0).to(device)
