"""Preprocessing utilities for AcousticSpace."""

from src.preprocessing.audio_loader import load_audio
from src.preprocessing.normalize import peak_normalize, rms_normalize
from src.preprocessing.resample import resample_waveform
from src.preprocessing.silence_removal import trim_silence, remove_silence
from src.preprocessing.noise_reduction import (
    highpass_filter,
    spectral_subtraction,
)
from src.preprocessing.padding import (
    pad_or_truncate,
    pad_or_truncate_to_duration,
)
from src.preprocessing.segmentation import (
    segment_waveform,
    segment_waveform_by_time,
)
from src.preprocessing.augmentation import (
    add_gaussian_noise,
    random_gain,
    time_mask_spectrogram,
    frequency_mask_spectrogram,
    apply_mixup,
)

__all__ = [
    "load_audio",
    "peak_normalize",
    "rms_normalize",
    "resample_waveform",
    "trim_silence",
    "remove_silence",
    "highpass_filter",
    "spectral_subtraction",
    "pad_or_truncate",
    "pad_or_truncate_to_duration",
    "segment_waveform",
    "segment_waveform_by_time",
    "add_gaussian_noise",
    "random_gain",
    "time_mask_spectrogram",
    "frequency_mask_spectrogram",
    "apply_mixup",
]