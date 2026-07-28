"""Audio resampling utilities."""

import logging
import torch
import torchaudio

logger = logging.getLogger(__name__)


def resample_waveform(
    waveform: torch.Tensor, 
    orig_sr: int, 
    target_sr: int
) -> torch.Tensor:
    """Resamples a waveform tensor from orig_sr to target_sr.

    Args:
        waveform: The input waveform tensor [channels, time].
        orig_sr: The original sample rate in Hz.
        target_sr: The target sample rate in Hz.

    Returns:
        The resampled waveform tensor.
    """
    if orig_sr == target_sr:
        return waveform
        
    try:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        resampled_waveform = resampler(waveform)
        return resampled_waveform
    except Exception as e:
        logger.error(f"Resampling failed from {orig_sr} to {target_sr}: {str(e)}")
        raise ValueError(f"Resampling failed: {str(e)}")
