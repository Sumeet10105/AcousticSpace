"""Audio noise reduction utilities."""

import logging
import torch
import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)


def highpass_filter(
    waveform: torch.Tensor, 
    sample_rate: int = 16000, 
    cutoff: float = 80.0, 
    order: int = 5
) -> torch.Tensor:
    """Applies a Butterworth high-pass filter to remove low-frequency rumble.

    Args:
        waveform: The input waveform tensor [channels, time].
        sample_rate: Sample rate in Hz.
        cutoff: Cutoff frequency in Hz (default: 80.0).
        order: Filter order (default: 5).

    Returns:
        The filtered waveform tensor.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    
    waveform_np = waveform.numpy()
    filtered_np = scipy.signal.filtfilt(b, a, waveform_np, axis=1)
    
    return torch.from_numpy(filtered_np.copy()).float()


def spectral_subtraction(
    waveform: torch.Tensor, 
    sample_rate: int = 16000, 
    noise_estimation_floor: float = 0.02,
    noise_profile_fraction: float = 0.1
) -> torch.Tensor:
    """Performs basic spectral subtraction for noise reduction.

    Estimates the noise floor from the quietest part (first fraction of audio)
    and subtracts it from the STFT magnitude spectrum.

    Args:
        waveform: The input waveform tensor [channels, time].
        sample_rate: Sample rate in Hz.
        noise_estimation_floor: Minimum scaling floor to avoid empty spectral bins (default: 0.02).
        noise_profile_fraction: Fraction of the audio at the start used to profile the noise (default: 0.1).

    Returns:
        The noise-reduced waveform tensor.
    """
    channels = waveform.shape[0]
    reduced_channels = []
    
    # Define STFT parameters
    n_fft = 512
    hop_length = 160
    win_length = 512
    window = torch.hann_window(win_length)
    
    for c in range(channels):
        channel_wave = waveform[c]
        
        # Compute STFT
        stft = torch.stft(
            channel_wave, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            window=window, 
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Estimate noise profile from the beginning frames of the magnitude
        num_frames = magnitude.shape[1]
        noise_frames = max(1, int(num_frames * noise_profile_fraction))
        noise_profile = torch.mean(magnitude[:, :noise_frames], dim=1, keepdim=True)
        
        # Perform spectral subtraction
        subtracted = magnitude - noise_profile
        
        # Keep a floor to prevent non-negative magnitudes and artifacts
        subtracted = torch.maximum(subtracted, magnitude * noise_estimation_floor)
        
        # Reconstruct complex STFT
        reconstructed_stft = subtracted * torch.exp(1j * phase)
        
        # Inverse STFT
        reduced_wave = torch.istft(
            reconstructed_stft, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            window=window,
            length=channel_wave.shape[0]
        )
        reduced_channels.append(reduced_wave)
        
    return torch.stack(reduced_channels, dim=0)
