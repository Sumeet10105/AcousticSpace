"""Audio segmentation utilities to split waveforms into sub-segments."""

import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def segment_waveform(
    waveform: torch.Tensor, 
    segment_samples: int, 
    hop_samples: int
) -> torch.Tensor:
    """Segments a waveform tensor into overlapping or non-overlapping windows.

    Args:
        waveform: The input waveform tensor [channels, time].
        segment_samples: Length of each segment in samples.
        hop_samples: Hop size between segments in samples.

    Returns:
        A tensor of shape [num_segments, channels, segment_samples].
    """
    channels, current_samples = waveform.shape
    
    if current_samples < segment_samples:
        # Pad to at least one segment
        import torch.nn.functional as F
        pad_len = segment_samples - current_samples
        waveform = F.pad(waveform, (0, pad_len))
        current_samples = segment_samples
        
    # Standard unfolding
    # unfolded shape: [channels, num_segments, segment_samples]
    unfolded = waveform.unfold(dimension=1, size=segment_samples, step=hop_samples)
    
    # Permute to [num_segments, channels, segment_samples]
    segmented = unfolded.permute(1, 0, 2)
    return segmented


def segment_waveform_by_time(
    waveform: torch.Tensor,
    sample_rate: int,
    segment_seconds: float,
    hop_seconds: float
) -> torch.Tensor:
    """Segments a waveform tensor using durations in seconds.

    Args:
        waveform: The input waveform tensor [channels, time].
        sample_rate: Sample rate in Hz.
        segment_seconds: Length of each segment in seconds.
        hop_seconds: Hop size between segments in seconds.

    Returns:
        A tensor of shape [num_segments, channels, segment_samples].
    """
    segment_samples = int(sample_rate * segment_seconds)
    hop_samples = int(sample_rate * hop_seconds)
    return segment_waveform(waveform, segment_samples, hop_samples)
