"""Breathing pattern and cadence feature extraction."""

import logging
import torch
import numpy as np
import librosa

logger = logging.getLogger(__name__)


def extract_breathing_features(
    audio: torch.Tensor, 
    sample_rate: int = 16000,
    frame_length: int = 512,
    hop_length: int = 160
) -> torch.Tensor:
    """Extracts features indicating breathing patterns (inhalation, pauses, cadence).

    Features calculated:
    - Spectral Flatness (higher for breath noise/friction)
    - High-frequency energy ratio (above 3.5 kHz vs total energy)
    - Zero Crossing Rate
    - Spectral Slope (spectral roll-off proxy)

    Args:
        audio: Audio waveform tensor of shape [channels, time].
        sample_rate: Audio sampling rate in Hz.
        frame_length: Frame size.
        hop_length: Hop length.

    Returns:
        Breathing features tensor of shape [channels, feature_dim, time_steps].
    """
    channels = audio.shape[0]
    device = audio.device
    audio_np = audio.cpu().numpy()
    
    # Calculate dimensions
    n_samples = audio.shape[1]
    num_frames = max(1, (n_samples - frame_length) // hop_length + 1)
    feature_dim = 4
    
    features_all_channels = []
    
    for c in range(channels):
        y = audio_np[c]
        channel_features = np.zeros((feature_dim, num_frames), dtype=np.float32)
        
        # 1. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)
        
        # 2. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop_length)
        
        # 3. High-frequency energy ratio and slope
        # Compute spectrogram
        spec = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
        
        # High freq index (above 3500 Hz)
        high_freq_mask = freqs > 3500
        
        # Calculate ratio and roll-off for each frame
        for f in range(num_frames):
            frame_spec = spec[:, min(f, spec.shape[1]-1)]
            total_energy = np.sum(frame_spec) + 1e-9
            high_energy = np.sum(frame_spec[high_freq_mask])
            
            high_ratio = high_energy / total_energy
            
            # Simple spectral slope proxy (energy weighted average frequency)
            centroid_freq = np.sum(freqs * frame_spec) / total_energy
            
            channel_features[0, f] = zcr[0, min(f, zcr.shape[1]-1)]
            channel_features[1, f] = flatness[0, min(f, flatness.shape[1]-1)]
            channel_features[2, f] = high_ratio
            channel_features[3, f] = centroid_freq / (sample_rate / 2.0)  # Normalize
            
        features_all_channels.append(torch.from_numpy(channel_features).float())
        
    return torch.stack(features_all_channels, dim=0).to(device)
