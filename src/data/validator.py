"""Audio data validation module."""

import logging
import os
import torchaudio

logger = logging.getLogger(__name__)

import os
from typing import Optional, Tuple

<<<<<<< HEAD
import torch


def validate_audio_data(
    audio_data: torch.Tensor,
    sample_rate: Optional[int] = None,
    min_duration_seconds: float = 0.1,
    max_duration_seconds: float = 600.0,
) -> bool:
    """Validate waveform tensor shape, finiteness, and optional duration bounds."""
    if not isinstance(audio_data, torch.Tensor):
        return False
    if audio_data.ndim != 2 or audio_data.shape[0] < 1:
        return False
    if not torch.isfinite(audio_data).all():
        return False
    if sample_rate is not None:
        duration = audio_data.shape[1] / float(sample_rate)
        if duration < min_duration_seconds or duration > max_duration_seconds:
            return False
    return True


def validate_uploaded_file(
    filename: str,
    content: bytes,
    max_size_mb: float = 50.0,
) -> Tuple[bool, str]:
    """Validate uploaded audio file name, size, and extension."""
    if not filename:
        return False, "Missing filename"
    if len(content) == 0:
        return False, "Empty file"
    if len(content) > max_size_mb * 1024 * 1024:
        return False, f"File exceeds {max_size_mb} MB limit"

    allowed = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    ext = os.path.splitext(filename.lower())[1]
    if ext not in allowed:
        return False, f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}"
    return True, ""
=======
def validate_audio_file(filepath: str) -> bool:
    """Validates if an audio file exists, has a valid format, and is not corrupted.

    Args:
        filepath: Path to the audio file to validate.

    Returns:
        True if the file is valid and readable, False otherwise.
    """
    if not isinstance(filepath, str) or not filepath:
        logger.warning("Invalid filepath type or empty string.")
        return False

    if not os.path.exists(filepath):
        logger.debug(f"Audio file does not exist: {filepath}")
        return False

    # Check if the file is empty (0 bytes)
    if os.path.getsize(filepath) == 0:
        logger.warning(f"Audio file is empty (0 bytes): {filepath}")
        return False

    # Attempt to read metadata info to ensure it is not corrupted
    try:
        info = torchaudio.info(filepath)
        if info.num_frames == 0 or info.sample_rate == 0:
            logger.warning(f"Audio file has no frames or invalid sample rate: {filepath}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Audio file validation failed (corrupted or unsupported format) for {filepath}: {str(e)}")
        return False


def validate_audio_data(audio_data) -> bool:
    """Validate audio data array/tensor shape and contents.

    Args:
        audio_data: The audio waveform tensor or array.

    Returns:
        True if the audio data is valid (non-empty, contains valid float numbers), False otherwise.
    """
    import torch
    import numpy as np
    
    if audio_data is None:
        logger.warning("Audio data is None.")
        return False
        
    if isinstance(audio_data, torch.Tensor):
        if audio_data.numel() == 0:
            logger.warning("Audio tensor is empty.")
            return False
        if torch.isnan(audio_data).any() or torch.isinf(audio_data).any():
            logger.warning("Audio tensor contains NaNs or Infs.")
            return False
        return True
        
    elif isinstance(audio_data, np.ndarray):
        if audio_data.size == 0:
            logger.warning("Audio numpy array is empty.")
            return False
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            logger.warning("Audio numpy array contains NaNs or Infs.")
            return False
        return True
        
    logger.warning(f"Unsupported audio data type: {type(audio_data)}")
    return False
>>>>>>> 01d9eff5e902e9e68f29504ca915f1ccf7624734
