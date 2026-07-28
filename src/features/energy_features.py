"""RMS Energy features extraction."""

import logging
import torch

logger = logging.getLogger(__name__)


def extract_rms_energy(
    audio: torch.Tensor, 
    frame_length: int = 512, 
    hop_length: int = 160
) -> torch.Tensor:
    """Computes Root Mean Square (RMS) energy per frame directly in PyTorch.

    Args:
        audio: Audio waveform tensor of shape [channels, time].
        frame_length: Size of the frame in samples (default: 512).
        hop_length: Hop length between frames in samples (default: 160).

    Returns:
        RMS energy tensor of shape [channels, 1, time_steps].
    """
    channels, current_samples = audio.shape
    
    if current_samples < frame_length:
        import torch.nn.functional as F
        pad_len = frame_length - current_samples
        audio = F.pad(audio, (0, pad_len))
        current_samples = frame_length
        
    # Unfold waveform to extract overlapping frames
    # [channels, num_frames, frame_length]
    unfolded = audio.unfold(dimension=1, size=frame_length, step=hop_length)
    
    # Calculate RMS: sqrt(mean(x^2))
    # [channels, num_frames]
    rms = torch.sqrt(torch.mean(unfolded ** 2, dim=2) + 1e-9)
    
    # Add a dimension to align with [channels, feature_dim, time_steps]
    return rms.unsqueeze(1)
