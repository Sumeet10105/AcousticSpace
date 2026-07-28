"""Spectral features extraction: centroid, bandwidth, rolloff, and contrast."""

import logging
import torch
import librosa
import numpy as np

logger = logging.getLogger(__name__)


def extract_spectral_features(
    audio: torch.Tensor, 
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160
) -> torch.Tensor:
    """Extracts multiple spectral features and returns them as a concatenated tensor.

    Features extracted:
    - Spectral Centroid (1 dimension)
    - Spectral Bandwidth (1 dimension)
    - Spectral Rolloff (1 dimension)
    - Spectral Contrast (7 dimensions by default in librosa)

    Args:
        audio: Audio waveform tensor [channels, time].
        sample_rate: Sample rate in Hz.
        n_fft: FFT size.
        hop_length: Hop length.

    Returns:
        Concatenated spectral features tensor [channels, feature_dim, time_steps].
    """
    channels = audio.shape[0]
    device = audio.device
    
    # Convert audio to numpy for librosa processing
    audio_np = audio.cpu().numpy()
    
    channels_features = []
    
    for c in range(channels):
        y = audio_np[c]
        
        # Calculate features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        
        # Concatenate features along feature dimension
        # shapes: centroid: [1, frames], bandwidth: [1, frames], rolloff: [1, frames], contrast: [7, frames]
        combined = np.concatenate([centroid, bandwidth, rolloff, contrast], axis=0) # shape [10, frames]
        channels_features.append(torch.from_numpy(combined).float())
        
    # Stack channels back
    spectral_tensor = torch.stack(channels_features, dim=0).to(device)
    return spectral_tensor
