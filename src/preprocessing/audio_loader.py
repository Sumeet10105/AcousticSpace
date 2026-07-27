"""Audio loading and standardization module."""

import logging
import os
import torch
import torchaudio
from typing import Tuple

logger = logging.getLogger(__name__)


def load_audio(
    filepath: str, 
    target_sr: int = 16000, 
    mono: bool = True
) -> Tuple[torch.Tensor, int]:
    """Loads an audio file and standardizes it to target sample rate and mono channel.

    Args:
        filepath: Path to the audio file.
        target_sr: The target sample rate in Hz (default: 16000).
        mono: If True, downmixes multi-channel audio to mono (default: True).

    Returns:
        A tuple of (waveform, sample_rate), where waveform is a 2D tensor of shape [channels, time].

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio file is corrupted or cannot be read.
    """
    if not os.path.exists(filepath):
        logger.error(f"Audio file not found: {filepath}")
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception as e:
        logger.error(f"Failed to load audio file {filepath}: {str(e)}")
        raise ValueError(f"Corrupted or invalid audio file: {filepath}. Error: {str(e)}")
    
    # Convert to mono if required
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample if sample rate does not match target
    if sr != target_sr:
        try:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        except Exception as e:
            logger.error(f"Resampling failed for {filepath} from {sr} to {target_sr}: {str(e)}")
            raise ValueError(f"Resampling failed: {str(e)}")
            
    return waveform, sr
