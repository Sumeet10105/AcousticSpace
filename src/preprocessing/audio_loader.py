"""Audio loading and standardization module."""

import logging
import os
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_audio(
    filepath: str,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and standardize it.

    Features:
    - Supports FLAC, WAV and other formats supported by torchaudio/soundfile.
    - Falls back to soundfile if torchaudio fails.
    - Converts multi-channel audio to mono.
    - Resamples audio to the target sample rate.

    Args:
        filepath: Path to the audio file.
        target_sr: Target sample rate.
        mono: Convert multi-channel audio to mono.

    Returns:
        Tuple containing:
            waveform (Tensor): Shape [channels, samples]
            sample_rate (int)

    Raises:
        FileNotFoundError:
            If the audio file does not exist.

        ValueError:
            If the file cannot be loaded or resampled.
    """

    if not os.path.exists(filepath):
        logger.error("Audio file not found: %s", filepath)
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # ---------------------------------------------------------
    # Try loading with torchaudio
    # ---------------------------------------------------------
    try:
        waveform, sr = torchaudio.load(filepath)

    except Exception as e:
        logger.warning(
            "torchaudio failed for %s. Falling back to soundfile. (%s)",
            filepath,
            str(e),
        )

        try:
            audio, sr = sf.read(filepath)

            audio = np.asarray(audio, dtype=np.float32)

            if audio.ndim == 1:
                waveform = torch.from_numpy(audio).unsqueeze(0)

            else:
                waveform = torch.from_numpy(audio).transpose(0, 1)

        except Exception as sf_error:
            logger.error(
                "Failed to load audio using both torchaudio and soundfile: %s",
                filepath,
            )

            raise ValueError(
                f"Unable to load audio file '{filepath}'. "
                f"torchaudio error: {e}; soundfile error: {sf_error}"
            )

    # ---------------------------------------------------------
    # Convert to mono
    # ---------------------------------------------------------
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # ---------------------------------------------------------
    # Resample
    # ---------------------------------------------------------
    if sr != target_sr:
        try:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=target_sr,
            )

            waveform = resampler(waveform)
            sr = target_sr

        except Exception as e:
            logger.error(
                "Failed to resample %s from %d Hz to %d Hz",
                filepath,
                sr,
                target_sr,
            )

            raise ValueError(
                f"Resampling failed for '{filepath}': {e}"
            )

    return waveform, sr