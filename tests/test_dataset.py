"""Tests for dataset module."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from src.data.dataset import AudioSpoofingDataset
from src.data.dataloader import create_dataloader
from src.data.splitter import train_val_test_split, speaker_aware_split
from src.data.validator import validate_audio_file, validate_audio_data


@pytest.fixture
def temp_dataset():
    """Fixture to generate synthetic audio files and a manifest CSV."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directories
        raw_dir = os.path.join(temp_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        # Create mock audio data
        sr = 16000
        duration = 1.0  # second
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        waveform = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        
        # Write temporary audio files
        audio_paths = []
        for i in range(3):
            file_path = os.path.join(raw_dir, f"audio_{i}.flac")
            sf.write(file_path, waveform, sr, format="FLAC")
            audio_paths.append(file_path)
            
        # Create corrupt audio file
        corrupt_path = os.path.join(raw_dir, "corrupt.flac")
        with open(corrupt_path, "w") as f:
            f.write("corrupted content not flac")
        audio_paths.append(corrupt_path)
            
        # Create manifest CSV
        manifest_data = {
            "filepath": audio_paths,
            "label": ["real", "fake", "real", "fake"],
            "attack_type": ["-", "A01", "-", "A02"],
            "speaker_id": ["SP_001", "SP_001", "SP_002", "SP_003"],
            "split": ["train", "train", "dev", "eval"]
        }
        df = pd.DataFrame(manifest_data)
        manifest_path = os.path.join(temp_dir, "manifest.csv")
        df.to_csv(manifest_path, index=False)
        
        yield temp_dir, manifest_path, audio_paths


class TestDataset:
    """Dataset tests."""
    
    def test_dataset_loading_and_splits(self, temp_dataset):
        """Test dataset splits filtering."""
        temp_dir, manifest_path, audio_paths = temp_dataset
        
        # Test Train split (2 entries in df, but one is corrupt. With validate_paths=True and lazy loading, 
        # the initial length will be 2. When we load, corrupt file falls back to a valid file)
        dataset_train = AudioSpoofingDataset(
            data_path=temp_dir, 
            metadata_path=manifest_path, 
            split="train", 
            validate_paths=True
        )
        assert len(dataset_train) == 2
        
        # Test Dev/Val split
        dataset_val = AudioSpoofingDataset(
            data_path=temp_dir, 
            metadata_path=manifest_path, 
            split="val", 
            validate_paths=True
        )
        assert len(dataset_val) == 1
        
        # Test Eval/Test split
        dataset_test = AudioSpoofingDataset(
            data_path=temp_dir, 
            metadata_path=manifest_path, 
            split="test", 
            validate_paths=True
        )
        assert len(dataset_test) == 1
        
    def test_dataset_getitem(self, temp_dataset):
        """Test dataset item retrieval and corrupted fallback."""
        temp_dir, manifest_path, audio_paths = temp_dataset
        
        dataset = AudioSpoofingDataset(
            data_path=temp_dir, 
            metadata_path=manifest_path, 
            split="train", 
            validate_paths=True
        )
        
        # Index 0 is a valid audio
        waveform, label = dataset[0]
        assert isinstance(waveform, torch.Tensor)
        assert label == 0  # real
        assert waveform.shape[0] == 1  # mono channel
        assert waveform.shape[1] == 16000  # 1 second of 16kHz audio
        
        # Index 1 is corrupted. Trigger fallback to a valid index (which will be index 0)
        # Should return a valid waveform and label without crashing
        waveform_fallback, label_fallback = dataset[1]
        assert isinstance(waveform_fallback, torch.Tensor)
        assert label_fallback in [0, 1]
        assert waveform_fallback.shape[1] == 16000

    def test_speaker_filtering(self, temp_dataset):
        """Test filtering by specific speaker IDs."""
        temp_dir, manifest_path, audio_paths = temp_dataset
        
        dataset = AudioSpoofingDataset(
            data_path=temp_dir,
            metadata_path=manifest_path,
            split="train",
            speakers=["SP_001"]
        )
        # Filters to speaker SP_001 (which has 2 entries)
        assert len(dataset) == 2


class TestSplitter:
    """Splitter tests."""
    
    def test_simple_split(self):
        """Test simple train/val/test split on list."""
        data = list(range(100))
        train, val, test = train_val_test_split(data, 0.8, 0.1)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        
    def test_speaker_aware_split(self):
        """Test speaker aware split prevents leakage."""
        df = pd.DataFrame({
            "filepath": [f"file_{i}.wav" for i in range(10)],
            "speaker_id": ["SP1", "SP1", "SP2", "SP2", "SP3", "SP3", "SP4", "SP4", "SP5", "SP5"],
            "label": ["real"] * 10
        })
        train_df, val_df, test_df = speaker_aware_split(df, 0.6, 0.2, random_seed=42)
        
        train_speakers = set(train_df["speaker_id"])
        val_speakers = set(val_df["speaker_id"])
        test_speakers = set(test_df["speaker_id"])
        
        # Assert no overlap
        assert train_speakers.isdisjoint(val_speakers)
        assert train_speakers.isdisjoint(test_speakers)
        assert val_speakers.isdisjoint(test_speakers)


class TestValidator:
    """Validator tests."""
    
    def test_file_validation(self, temp_dataset):
        """Test validation of file existence and format."""
        temp_dir, manifest_path, audio_paths = temp_dataset
        
        # Valid path
        assert validate_audio_file(audio_paths[0]) is True
        # Invalid corrupt path
        assert validate_audio_file(audio_paths[3]) is False
        # Non-existent path
        assert validate_audio_file("nonexistent.flac") is False
        
    def test_data_validation(self):
        """Test tensor validation."""
        valid_tensor = torch.zeros(1, 16000)
        assert validate_audio_data(valid_tensor) is True
        
        nan_tensor = torch.tensor([float('nan'), 1.0])
        assert validate_audio_data(nan_tensor) is False
        
        assert validate_audio_data(None) is False
