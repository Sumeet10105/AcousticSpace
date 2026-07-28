"""MFCC feature extraction module."""

import logging
import torch
import torchaudio

logger = logging.getLogger(__name__)


def compute_mfcc(
    audio: torch.Tensor, 
    sample_rate: int = 16000, 
    n_mfcc: int = 13, 
    n_fft: int = 512, 
    hop_length: int = 160, 
    n_mels: int = 64
) -> torch.Tensor:
    """Computes MFCC features from an audio waveform tensor.

    Args:
        audio: Audio waveform tensor of shape [channels, time].
        sample_rate: Audio sampling rate in Hz (default: 16000).
        n_mfcc: Number of MFCC coefficients to retain (default: 13).
        n_fft: FFT size (default: 512).
        hop_length: Hop length (default: 160).
        n_mels: Number of Mel filter bands (default: 64).

    Returns:
        MFCC tensor of shape [channels, n_mfcc, time_steps].
    """
    try:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "mel_scale": "htk",
            }
        ).to(audio.device)
        
        mfccs = mfcc_transform(audio)
        return mfccs
        
    except Exception as e:
        logger.error(f"MFCC computation failed: {str(e)}")
        raise ValueError(f"MFCC computation failed: {str(e)}")
