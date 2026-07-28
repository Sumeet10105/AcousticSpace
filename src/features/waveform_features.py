"""Raw waveform statistical and time-domain features."""

import logging
import torch

logger = logging.getLogger(__name__)


def extract_waveform_features(
    audio: torch.Tensor, 
    frame_length: int = 512, 
    hop_length: int = 160
) -> torch.Tensor:
    """Extracts time-domain statistical descriptors from raw audio waveform.

    Features extracted per frame:
    - Zero Crossing Rate (ZCR)
    - Mean
    - Standard Deviation
    - Skewness
    - Kurtosis

    Args:
        audio: Audio waveform tensor of shape [channels, time].
        frame_length: Frame size.
        hop_length: Hop length.

    Returns:
        Time-domain features tensor of shape [channels, feature_dim, time_steps].
    """
    channels, current_samples = audio.shape
    
    if current_samples < frame_length:
        import torch.nn.functional as F
        pad_len = frame_length - current_samples
        audio = F.pad(audio, (0, pad_len))
        current_samples = frame_length
        
    # Unfold to shape [channels, num_frames, frame_length]
    unfolded = audio.unfold(dimension=1, size=frame_length, step=hop_length)
    
    # Calculate statistics
    mean = torch.mean(unfolded, dim=2)  # [channels, num_frames]
    std = torch.std(unfolded, dim=2) + 1e-9  # [channels, num_frames]
    
    # Skewness: E[(X-mu)^3] / std^3
    diff = unfolded - mean.unsqueeze(2)
    skewness = torch.mean(diff ** 3, dim=2) / (std ** 3)
    
    # Kurtosis: E[(X-mu)^4] / std^4 - 3 (excess kurtosis)
    kurtosis = torch.mean(diff ** 4, dim=2) / (std ** 4) - 3.0
    
    # Zero Crossing Rate (ZCR)
    # ZCR(t) = 1/(2*N) * sum(|sign(x_n) - sign(x_{n-1})|)
    signs = torch.sign(unfolded)
    # diff of adjacent signs
    sign_diffs = torch.abs(signs[:, :, 1:] - signs[:, :, :-1])
    zcr = torch.mean(sign_diffs, dim=2) * 0.5
    
    # Concatenate features
    # features shape: [channels, 5, num_frames]
    features = torch.stack([zcr, mean, std, skewness, kurtosis], dim=1)
    
    return features
