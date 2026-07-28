"""Linear spectrogram feature extraction."""

import logging
import torch
import torchaudio

logger = logging.getLogger(__name__)


def compute_spectrogram(
    audio: torch.Tensor, 
    n_fft: int = 512, 
    hop_length: int = 160
) -> torch.Tensor:
    """Computes a linear power spectrogram from an audio waveform tensor.

    Args:
        audio: Audio waveform tensor of shape [channels, time].
        n_fft: FFT size (default: 512).
        hop_length: Hop length in samples (default: 160).

    Returns:
        Linear spectrogram tensor of shape [channels, freq_bins, time_steps].
    """
    try:
        # Define window function
        win_length = n_fft
        window = torch.hann_window(win_length, device=audio.device)
        
        # Compute STFT
        stft = torch.stft(
            audio, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            window=window, 
            return_complex=True
        )
        
        # Compute power spectrum
        spectrogram = torch.abs(stft) ** 2
        return spectrogram
        
    except Exception as e:
        logger.error(f"Spectrogram computation failed: {str(e)}")
        raise ValueError(f"Spectrogram computation failed: {str(e)}")
