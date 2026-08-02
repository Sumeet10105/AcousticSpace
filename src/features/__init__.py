"""Feature engineering module for AcousticSpace."""

from src.features.spectrogram import compute_spectrogram
from src.features.mel_spectrogram import compute_mel_spectrogram
from src.features.mfcc import compute_mfcc
from src.features.spectral_features import extract_spectral_features
from src.features.chroma import extract_chroma
from src.features.energy_features import extract_rms_energy
from src.features.rir import extract_rir_features
from src.features.breathing import extract_breathing_features
from src.features.waveform_features import extract_waveform_features
from src.features.feature_utils import align_temporal_dimension, standardize_features
from src.features.feature_fusion import fuse_features
from src.features.explanation import generate_spoof_explanation

__all__ = [
    "compute_spectrogram",
    "compute_mel_spectrogram",
    "compute_mfcc",
    "extract_spectral_features",
    "extract_chroma",
    "extract_rms_energy",
    "extract_rir_features",
    "extract_breathing_features",
    "extract_waveform_features",
    "align_temporal_dimension",
    "standardize_features",
    "fuse_features",
    "generate_spoof_explanation",
]
