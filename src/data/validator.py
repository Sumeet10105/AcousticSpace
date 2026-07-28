"""Data validation utilities."""

import os
from typing import Optional, Tuple

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
