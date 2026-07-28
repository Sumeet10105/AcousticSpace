"""Tests for dataset module."""

import pytest
import numpy as np


class TestDataset:
    """Dataset tests."""
    
    def test_dataset_length(self):
        """Test dataset length."""
        from src.data.dataset import AudioSpoofingDataset
        import os
        
        # Test on dev protocol
        data_path = "d:/Infotact_Solutions/AcousticSpace/datasets/LA/LA/ASVspoof2019_LA_dev/flac"
        metadata_path = "d:/Infotact_Solutions/AcousticSpace/datasets/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
        
        # Load small dataset (e.g. 5 samples)
        dataset = AudioSpoofingDataset(data_path, metadata_path, split='dev', max_samples=5)
        
        assert len(dataset) == 5
        assert len(dataset.labels) == 5
        assert len(dataset.speaker_ids) == 5
        
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        from src.data.dataset import AudioSpoofingDataset
        import torch
        
        data_path = "d:/Infotact_Solutions/AcousticSpace/datasets/LA/LA/ASVspoof2019_LA_dev/flac"
        metadata_path = "d:/Infotact_Solutions/AcousticSpace/datasets/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
        
        dataset = AudioSpoofingDataset(data_path, metadata_path, split='dev', max_samples=2)
        waveform, label = dataset[0]
        
        assert isinstance(waveform, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.item() in [0, 1]

