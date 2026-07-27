"""Tests for preprocessing module."""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch

from src.preprocessing.audio_loader import load_audio


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
    
    def test_audio_loading_stereo_to_mono_and_resample(self, temp_audio):
        """Test audio loading downmixes to mono and resamples correctly."""
        temp_dir, stereo_file, orig_sr = temp_audio
        
        # Load and resample to 16000 Hz and mono
        target_sr = 16000
        waveform, sr = load_audio(stereo_file, target_sr=target_sr, mono=True)
        
        assert sr == target_sr
        assert waveform.shape[0] == 1  # mono channel
        # 1.0 second * 16000 Hz = 16000 samples
        assert abs(waveform.shape[1] - target_sr) < 5  # Allow tiny rounding diffs if any
        assert isinstance(waveform, torch.Tensor)
        
    def test_audio_loading_file_not_found(self):
        """Test audio loader raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_audio("nonexistent_file_path.wav")
            
    def test_audio_loading_corrupt_file(self):
        """Test audio loader raises ValueError for invalid files."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"non audio garbage data")
            corrupt_file = f.name
            
        try:
            with pytest.raises(ValueError):
                load_audio(corrupt_file)
        finally:
            if os.path.exists(corrupt_file):
                os.remove(corrupt_file)
