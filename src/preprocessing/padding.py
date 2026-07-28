"""Audio padding and truncation utilities to standardize duration."""

import logging
import torch

logger = logging.getLogger(__name__)


def pad_or_truncate(
    waveform: torch.Tensor, 
    target_samples: int, 
    mode: str = "zero"
) -> torch.Tensor:
    """Pads or truncates a waveform to a exact number of target samples.

    Args:
        waveform: The input waveform tensor [channels, time].
        target_samples: The target length in samples.
        mode: Padding mode: "zero", "wrap", or "reflect" (default: "zero").

    Returns:
        The padded/truncated waveform tensor [channels, target_samples].
    """
    channels, current_samples = waveform.shape
    
    if current_samples == target_samples:
        return waveform
        
    if current_samples > target_samples:
        # Truncate: take the middle or start segment. Let's take the middle segment.
        start = (current_samples - target_samples) // 2
        return waveform[:, start:start + target_samples]
        
    # Pad
    pad_len = target_samples - current_samples
    
    if mode == "zero":
        # Pad at the end
        pad_tensor = torch.zeros(channels, pad_len, dtype=waveform.dtype, device=waveform.device)
        return torch.cat([waveform, pad_tensor], dim=1)
        
    elif mode == "wrap":
        # Repeat the waveform until it fits
        repeats = (target_samples // current_samples) + 1
        repeated = waveform.repeat(1, repeats)
        return repeated[:, :target_samples]
        
    elif mode == "reflect":
        # PyTorch F.pad requires 4D or 3D tensors for reflection padding, so we do it manually or via repeat/reversal
        # For simplicity, we can do zero-padding or wrap-padding if the length is too small, or pad using torch.nn.functional.pad
        import torch.nn.functional as F
        # Reflection pad requires the pad to be less than the input length
        if pad_len < current_samples:
            # F.pad expects padding for dimensions in reverse order: (left, right) for 1D signal
            # waveform has shape [channels, time] -> add batch dim to make it 3D: [1, channels, time]
            padded = F.pad(waveform.unsqueeze(0), (0, pad_len), mode="reflect")
            return padded.squeeze(0)
        else:
            # Fallback to wrap if pad is larger than current waveform length
            repeats = (target_samples // current_samples) + 1
            repeated = waveform.repeat(1, repeats)
            return repeated[:, :target_samples]
            
    else:
        logger.warning(f"Unrecognized padding mode '{mode}'. Defaulting to zero padding.")
        pad_tensor = torch.zeros(channels, pad_len, dtype=waveform.dtype, device=waveform.device)
        return torch.cat([waveform, pad_tensor], dim=1)


def pad_or_truncate_to_duration(
    waveform: torch.Tensor,
    sample_rate: int,
    duration_seconds: float,
    mode: str = "zero"
) -> torch.Tensor:
    """Pads or truncates a waveform to a target duration in seconds.

    Args:
        waveform: The input waveform tensor [channels, time].
        sample_rate: The sample rate of the waveform in Hz.
        duration_seconds: Target duration in seconds.
        mode: Padding mode: "zero", "wrap", or "reflect" (default: "zero").

    Returns:
        The padded/truncated waveform tensor.
    """
    target_samples = int(sample_rate * duration_seconds)
    return pad_or_truncate(waveform, target_samples, mode=mode)
