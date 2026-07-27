"""Tests for the audio preprocessing module."""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch

from src.preprocessing import (
    load_audio,
    peak_normalize,
    rms_normalize,
    resample_waveform,
    trim_silence,
    remove_silence,
    highpass_filter,
    spectral_subtraction,
    pad_or_truncate,
    pad_or_truncate_to_duration,
    segment_waveform,
    segment_waveform_by_time,
    add_gaussian_noise,
    random_gain,
    time_mask_spectrogram,
    frequency_mask_spectrogram,
    apply_mixup,
)


@pytest.fixture
def temp_audio():
    """Fixture to generate a temporary multi-channel and resampled audio file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a stereo signal at 22050 Hz
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wave_left = np.sin(2 * np.pi * 440 * t)
        wave_right = np.sin(2 * np.pi * 880 * t)
        stereo_signal = np.stack([wave_left, wave_right], axis=1)  # shape [samples, channels]
        
        stereo_file = os.path.join(temp_dir, "stereo.flac")
        sf.write(stereo_file, stereo_signal, sr, format="FLAC")
        
        yield temp_dir, stereo_file, sr


class TestPreprocessing:
    """Preprocessing tests."""
    
    def test_audio_loading(self, temp_audio):
        """Test audio loading downmixes to mono and resamples correctly."""
        temp_dir, stereo_file, orig_sr = temp_audio
        
        # Load and resample to 16000 Hz and mono
        target_sr = 16000
        waveform, sr = load_audio(stereo_file, target_sr=target_sr, mono=True)
        
        assert sr == target_sr
        assert waveform.shape[0] == 1  # mono channel
        assert abs(waveform.shape[1] - target_sr) < 5  # Allow tiny rounding diffs
        assert isinstance(waveform, torch.Tensor)
        
    def test_normalization(self, temp_audio):
        """Test peak and RMS normalization."""
        temp_dir, stereo_file, orig_sr = temp_audio
        waveform, sr = load_audio(stereo_file, target_sr=16000, mono=True)
        
        # Test peak normalize
        peak_normed = peak_normalize(waveform, target_peak=0.9)
        assert abs(torch.max(torch.abs(peak_normed)).item() - 0.9) < 1e-5
        
        # Test RMS normalize
        rms_normed = rms_normalize(waveform, target_db=-15.0)
        rms = torch.sqrt(torch.mean(rms_normed ** 2))
        expected_rms = 10.0 ** (-15.0 / 20.0)
        assert abs(rms.item() - expected_rms) < 1e-2  # Clipped peak fallback may affect it slightly, check bounds
        
    def test_resample(self):
        """Test resampling utility directly."""
        waveform = torch.randn(1, 8000)  # 1 sec of 8kHz
        resampled = resample_waveform(waveform, orig_sr=8000, target_sr=16000)
        assert resampled.shape[1] == 16000
        
    def test_silence_removal(self):
        """Test silence removal and trimming."""
        # Waveform containing silence and a burst of signal
        waveform = torch.zeros(1, 16000)
        waveform[0, 4000:12000] = 0.5  # signal in middle
        
        trimmed = trim_silence(waveform, threshold_db=-40.0)
        # Expected to trim start and end silence
        assert trimmed.shape[1] <= 9000
        assert trimmed.shape[1] >= 7900
        
        removed = remove_silence(waveform, sample_rate=16000, threshold_db=-40.0)
        assert removed.shape[1] <= 9000
        
    def test_noise_reduction(self, temp_audio):
        """Test Butterworth filter and spectral subtraction."""
        temp_dir, stereo_file, orig_sr = temp_audio
        waveform, sr = load_audio(stereo_file, target_sr=16000, mono=True)
        
        hp_filtered = highpass_filter(waveform, sample_rate=sr, cutoff=100.0)
        assert hp_filtered.shape == waveform.shape
        
        denoised = spectral_subtraction(waveform, sample_rate=sr)
        assert denoised.shape == waveform.shape
        
    def test_padding(self):
        """Test padding and truncation."""
        waveform = torch.ones(1, 1000)
        
        # Test truncation
        truncated = pad_or_truncate(waveform, target_samples=500)
        assert truncated.shape[1] == 500
        
        # Test zero padding
        padded_zero = pad_or_truncate(waveform, target_samples=2000, mode="zero")
        assert padded_zero.shape[1] == 2000
        assert torch.all(padded_zero[0, 1000:] == 0.0)
        
        # Test wrap padding
        padded_wrap = pad_or_truncate(waveform, target_samples=2500, mode="wrap")
        assert padded_wrap.shape[1] == 2500
        assert torch.all(padded_wrap[0, 1000:2000] == 1.0)
        
    def test_segmentation(self):
        """Test wave segmentation."""
        waveform = torch.randn(1, 10000)
        segmented = segment_waveform(waveform, segment_samples=2000, hop_samples=1000)
        # 10000 length with 2000 width and 1000 hop:
        # starts at: 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000. Total 9 segments.
        assert list(segmented.shape) == [9, 1, 2000]
        
    def test_augmentations(self):
        """Test augmentations (noise, gain, masks, mixup)."""
        waveform = torch.randn(1, 16000)
        
        noisy = add_gaussian_noise(waveform, snr_db_range=(20, 30))
        assert noisy.shape == waveform.shape
        
        gain_adjusted = random_gain(waveform, gain_range=(0.8, 1.2))
        assert gain_adjusted.shape == waveform.shape
        
        spec = torch.randn(1, 64, 100)
        masked_time = time_mask_spectrogram(spec, max_mask_percentage=0.1, num_masks=2)
        assert masked_time.shape == spec.shape
        
        masked_freq = frequency_mask_spectrogram(spec, max_mask_percentage=0.1, num_masks=2)
        assert masked_freq.shape == spec.shape
        
        wave2 = torch.randn(1, 16000)
        mixed, lam = apply_mixup(waveform, wave2, alpha=0.2)
        assert mixed.shape == waveform.shape
        assert 0.0 <= lam <= 1.0
