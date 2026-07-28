"""Audio data validation module."""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def validate_audio_file(filepath: str) -> bool:
    """Validate that an audio file exists, is non-empty, and is readable."""
    if not isinstance(filepath, str) or not filepath:
        logger.warning("Invalid filepath type or empty string.")
        return False

    if not os.path.exists(filepath):
        logger.debug("Audio file does not exist: %s", filepath)
        return False

    if os.path.getsize(filepath) == 0:
        logger.warning("Audio file is empty (0 bytes): %s", filepath)
        return False

    try:
        info = torchaudio.info(filepath)
        if info.num_frames == 0 or info.sample_rate == 0:
            logger.warning("Audio file has no frames or invalid sample rate: %s", filepath)
            return False
        return True
    except Exception as exc:
        logger.warning("Audio file validation failed for %s: %s", filepath, exc)
        return False


def validate_audio_data(
    audio_data: torch.Tensor,
    sample_rate: Optional[int] = None,
    min_duration_seconds: float = 0.1,
    max_duration_seconds: float = 600.0,
) -> bool:
    """Validate waveform tensor shape, finiteness, and optional duration bounds."""
    if audio_data is None:
        logger.warning("Audio data is None.")
        return False

    if isinstance(audio_data, torch.Tensor):
        if audio_data.ndim != 2 or audio_data.shape[0] < 1:
            return False
        if not torch.isfinite(audio_data).all():
            return False
        if sample_rate is not None:
            duration = audio_data.shape[1] / float(sample_rate)
            if duration < min_duration_seconds or duration > max_duration_seconds:
                return False
        return True

    if isinstance(audio_data, np.ndarray):
        if audio_data.size == 0:
            logger.warning("Audio numpy array is empty.")
            return False
        if np.isnan(audio_data).any() or np.isinf(audio_data).any():
            logger.warning("Audio numpy array contains NaNs or Infs.")
            return False
        return True

    logger.warning("Unsupported audio data type: %s", type(audio_data))
    return False


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
