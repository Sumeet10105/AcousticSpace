"""Audio silence removal utilities."""

import logging
import torch

logger = logging.getLogger(__name__)


def trim_silence(
    waveform: torch.Tensor, 
    threshold_db: float = -50.0
) -> torch.Tensor:
    """Trims leading and trailing silence from the waveform.

    Args:
        waveform: The input waveform tensor [channels, time].
        threshold_db: The threshold in dB below which audio is considered silent (default: -50.0).

    Returns:
        The trimmed waveform tensor.
    """
    # Use average energy across channels
    mono_waveform = torch.mean(waveform, dim=0)
    
    # Convert amplitude to dB
    ref_val = 1.0  # reference peak value
    db = 20 * torch.log10(torch.abs(mono_waveform) + 1e-9)
    
    # Find indices where energy exceeds the threshold
    active_indices = torch.where(db > threshold_db)[0]
    
    if len(active_indices) == 0:
        logger.warning(f"Whole audio is below threshold of {threshold_db} dB. Returning original waveform.")
        return waveform
        
    start_idx = active_indices[0].item()
    end_idx = active_indices[-1].item() + 1
    
    trimmed = waveform[:, start_idx:end_idx]
    return trimmed


def remove_silence(
    waveform: torch.Tensor, 
    sample_rate: int = 16000,
    frame_length_ms: float = 20.0, 
    hop_length_ms: float = 10.0, 
    threshold_db: float = -45.0
) -> torch.Tensor:
    """Removes both interior and exterior silent segments using a frame-based energy threshold.

    Args:
        waveform: The input waveform tensor [channels, time].
        sample_rate: The sample rate of the audio (default: 16000).
        frame_length_ms: The length of each frame in milliseconds.
        hop_length_ms: The hop size between frames in milliseconds.
        threshold_db: Threshold in dB below which frames are deleted.

    Returns:
        A concatenated waveform tensor of active frames.
    """
    channels = waveform.shape[0]
    frame_size = int(sample_rate * (frame_length_ms / 1000.0))
    hop_size = int(sample_rate * (hop_length_ms / 1000.0))
    
    if waveform.shape[1] < frame_size:
        return waveform
        
    # Standard unfolder / sliding window to extract frames
    # shape of waveform: [channels, time] -> unfold to [channels, num_frames, frame_size]
    unfolded = waveform.unfold(dimension=1, size=frame_size, step=hop_size)
    num_frames = unfolded.shape[1]
    
    # Calculate energy per frame across channels
    # [channels, num_frames, frame_size]
    # average energy for each frame:
    frame_rms = torch.sqrt(torch.mean(unfolded ** 2, dim=2))  # shape [channels, num_frames]
    mean_rms = torch.mean(frame_rms, dim=0)  # shape [num_frames]
    
    # Convert RMS to dB
    frame_db = 20 * torch.log10(mean_rms + 1e-9)
    
    # Select active frame indices
    active_frame_mask = frame_db > threshold_db
    
    if not torch.any(active_frame_mask):
        logger.warning(f"All frames are silent (below {threshold_db} dB). Returning original waveform.")
        return waveform
        
    active_indices = torch.where(active_frame_mask)[0]
    
    # Reconstruct the active segments
    # Since hop_size < frame_size, we overlap. To avoid complex window reconstruction, 
    # we copy the hop parts of active frames to a new tensor
    active_segments = []
    for idx in active_indices:
        start = idx.item() * hop_size
        end = start + hop_size
        active_segments.append(waveform[:, start:end])
        
    if len(active_segments) == 0:
        return waveform
        
    return torch.cat(active_segments, dim=1)
