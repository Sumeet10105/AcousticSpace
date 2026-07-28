"""Tests for the prediction and inference pipeline modules."""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch

from src.models import CRNN
from src.predict import predict_single_file, predict_batch, predict_directory


@pytest.fixture
def temp_inference_data():
    """Generates synthetic audio files and a test directory for inference testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid audio file
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        waveform = np.sin(2 * np.pi * 440 * t)
        
        valid_path = os.path.join(temp_dir, "test_valid.flac")
        sf.write(valid_path, waveform, sr, format="FLAC")
        
        # Create a corrupted file
        corrupt_path = os.path.join(temp_dir, "test_corrupt.flac")
        with open(corrupt_path, "w") as f:
            f.write("corrupted file")
            
        yield temp_dir, valid_path, corrupt_path


class TestInferencePipeline:
    """Inference pipelines tests."""
    
    def test_single_file_prediction(self, temp_inference_data):
        """Test single file prediction pipeline returns correct keys and performance stats."""
        temp_dir, valid_path, corrupt_path = temp_inference_data
        
        # Initialize a small mock model
        model = CRNN(num_classes=2, n_mels=64)
        
        res = predict_single_file(
            audio_path=valid_path,
            model=model,
            device="cpu",
            threshold=0.5,
            duration_seconds=1.0
        )
        
        # Verify result dictionary keys
        assert res["filepath"] == valid_path
        assert res["prediction"] in ["REAL", "FAKE"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert 0.0 <= res["probability"] <= 1.0
        assert res["latency_ms"] > 0.0
        assert res["memory_used_mb"] >= 0.0
        assert isinstance(res["feature_importance"], np.ndarray)
        assert res["feature_importance"].shape[0] == 64  # average power per Mel bin
        
    def test_batch_prediction(self, temp_inference_data):
        """Test batch prediction handles missing and corrupted files gracefully."""
        temp_dir, valid_path, corrupt_path = temp_inference_data
        model = CRNN(num_classes=2, n_mels=64)
        
        # Input paths: one valid, one corrupt, one missing
        paths = [valid_path, corrupt_path, "nonexistent.wav"]
        
        batch_res = predict_batch(
            audio_paths=paths,
            model=model,
            device="cpu",
            threshold=0.5,
            duration_seconds=1.0
        )
        
        assert batch_res["total_files"] == 3
        # Should skip corrupt and nonexistent, only 1 succeeds
        assert batch_res["successful_predictions"] == 1
        assert len(batch_res["predictions"]) == 1
        assert batch_res["predictions"][0]["filepath"] == valid_path
        
    def test_directory_prediction(self, temp_inference_data):
        """Test directory scanning and prediction."""
        temp_dir, valid_path, corrupt_path = temp_inference_data
        model = CRNN(num_classes=2, n_mels=64)
        
        # Scan temporary directory
        dir_res = predict_directory(
            directory_path=temp_dir,
            model=model,
            device="cpu",
            threshold=0.5,
            duration_seconds=1.0,
            supported_extensions=(".flac",)
        )
        
        # Valid path is .flac, corrupt is .flac -> total 2 discovered
        assert dir_res["total_files"] == 2
        # Only valid succeeds
        assert dir_res["successful_predictions"] == 1
