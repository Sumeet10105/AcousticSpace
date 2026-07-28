"""Audio data augmentation utilities at waveform and spectrogram levels."""

import logging
import random
import torch
import torchaudio

logger = logging.getLogger(__name__)


def add_gaussian_noise(
    waveform: torch.Tensor, 
    snr_db_range: tuple = (10, 40)
) -> torch.Tensor:
    """Adds white Gaussian noise to the waveform with a random SNR within the given range.

    Args:
        waveform: The input waveform tensor [channels, time].
        snr_db_range: A tuple of (min_snr_db, max_snr_db) (default: (10, 40)).

    Returns:
        The noisy waveform tensor.
    """
    snr_db = random.uniform(snr_db_range[0], snr_db_range[1])
    
    # Calculate power of the signal
    sig_power = torch.mean(waveform ** 2)
    
    # SNR = 10 * log10(sig_power / noise_power)
    # noise_power = sig_power / (10^(SNR/10))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0) + 1e-9)
    
    # Generate Gaussian noise
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    
    return waveform + noise


def random_gain(
    waveform: torch.Tensor, 
    gain_range: tuple = (0.5, 1.5)
) -> torch.Tensor:
    """Scales the amplitude of the waveform by a random factor.

    Args:
        waveform: The input waveform tensor [channels, time].
        gain_range: A tuple of (min_gain, max_gain) (default: (0.5, 1.5)).

    Returns:
        The augmented waveform tensor.
    """
    factor = random.uniform(gain_range[0], gain_range[1])
    return waveform * factor


def time_mask_spectrogram(
    spectrogram: torch.Tensor, 
    max_mask_percentage: float = 0.1, 
    num_masks: int = 1
) -> torch.Tensor:
    """Applies time masking to a spectrogram tensor (SpecAugment).

    Masks (sets to zero) vertical bands of the spectrogram.

    Args:
        spectrogram: The input spectrogram tensor [channels, freq, time].
        max_mask_percentage: Maximum percentage of time frames to mask.
        num_masks: Number of masks to apply.

    Returns:
        The time-masked spectrogram tensor.
    """
    augmented = spectrogram.clone()
    _, _, num_time_steps = augmented.shape
    max_mask_width = int(num_time_steps * max_mask_percentage)
    
    for _ in range(num_masks):
        if max_mask_width <= 1:
            continue
        mask_width = random.randint(1, max_mask_width)
        mask_start = random.randint(0, num_time_steps - mask_width)
        augmented[:, :, mask_start:mask_start + mask_width] = 0.0
        
    return augmented


def frequency_mask_spectrogram(
    spectrogram: torch.Tensor, 
    max_mask_percentage: float = 0.15, 
    num_masks: int = 1
) -> torch.Tensor:
    """Applies frequency masking to a spectrogram tensor (SpecAugment).

    Masks (sets to zero) horizontal bands of the spectrogram.

    Args:
        spectrogram: The input spectrogram tensor [channels, freq, time].
        max_mask_percentage: Maximum percentage of frequency bins to mask.
        num_masks: Number of masks to apply.

    Returns:
        The frequency-masked spectrogram tensor.
    """
    augmented = spectrogram.clone()
    _, num_freq_bins, _ = augmented.shape
    max_mask_width = int(num_freq_bins * max_mask_percentage)
    
    for _ in range(num_masks):
        if max_mask_width <= 1:
            continue
        mask_width = random.randint(1, max_mask_width)
        mask_start = random.randint(0, num_freq_bins - mask_width)
        augmented[:, mask_start:mask_start + mask_width, :] = 0.0
        
    return augmented


def apply_mixup(
    wave1: torch.Tensor, 
    wave2: torch.Tensor, 
    alpha: float = 0.2
) -> tuple:
    """Performs Mixup augmentation on two waveforms.

    Args:
        wave1: First waveform tensor.
        wave2: Second waveform tensor.
        alpha: Mixup interpolation coefficient parameter (beta distribution).

    Returns:
        A tuple of (mixed_waveform, lam), where lam is the mix ratio.
    """
    if alpha <= 0.0:
        return wave1, 1.0
        
    # Sample lambda from beta distribution
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    
    # Adjust lengths if they differ
    l1, l2 = wave1.shape[1], wave2.shape[1]
    if l1 != l2:
        min_len = min(l1, l2)
        wave1 = wave1[:, :min_len]
        wave2 = wave2[:, :min_len]
        
    mixed = lam * wave1 + (1.0 - lam) * wave2
    return mixed, lam
