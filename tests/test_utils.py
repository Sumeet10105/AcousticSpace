"""Unit tests for utility and helper modules."""

import os
import tempfile
import pytest
import numpy as np
import torch
import soundfile as sf
import pandas as pd

from src.utils.helper import to_tensor, to_numpy, move_to_device
from src.utils.io import save_pickle, load_pickle, save_json, load_json
from src.utils.seed import set_seed
from src.data_loader import load_audio
from src.build_manifest import build_manifest
import src.predict as predict


class TestUtilities:
    """Tests for system utilities, helpers, serialization, and manifest scripts."""
    
    def test_helper_conversions(self):
        """Test numpy to/from torch conversions and device moving."""
        arr = np.array([1, 2, 3], dtype=np.float32)
        tensor = to_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))
        
        # Test converting back
        arr_back = to_numpy(tensor)
        assert isinstance(arr_back, np.ndarray)
        assert np.allclose(arr, arr_back)
        
        # Identity tests
        assert to_tensor(tensor) is tensor
        assert to_numpy(arr) is arr
        
        # Move to device
        moved = move_to_device(tensor, "cpu")
        assert moved.device.type == "cpu"
        
        dct = {"a": tensor, "b": [tensor]}
        moved_dct = move_to_device(dct, "cpu")
        assert moved_dct["a"].device.type == "cpu"
        assert moved_dct["b"][0].device.type == "cpu"
        
        # Scalar fallback
        assert move_to_device(42, "cpu") == 42
        
    def test_pickle_and_json_io(self):
        """Test pickle and json load/save utilities."""
        data = {"hello": "world", "values": [1, 2, 3]}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pkl_file = os.path.join(temp_dir, "data.pkl")
            json_file = os.path.join(temp_dir, "data.json")
            
            # Test pickle
            save_pickle(data, pkl_file)
            assert os.path.exists(pkl_file)
            data_pkl = load_pickle(pkl_file)
            assert data_pkl == data
            
            # Test JSON
            save_json(data, json_file)
            assert os.path.exists(json_file)
            data_json = load_json(json_file)
            assert data_json == data
            
    def test_seed_control(self):
        """Test seed utility results in reproducible torch outputs."""
        set_seed(123)
        a = torch.randn(5)
        
        set_seed(123)
        b = torch.randn(5)
        
        assert torch.allclose(a, b)
        
    def test_data_loader_audio(self):
        """Test src/data_loader.py legacy downmix function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "audio.flac")
            sf.write(filepath, [0.1, 0.2, 0.3], 16000)
            
            audio, sr = load_audio(filepath)
            assert sr == 16000
            assert len(audio) == 3
            
    def test_predict_reexports(self):
        """Verify predict package exposes core functions."""
        assert hasattr(predict, "predict_single_file")
        assert hasattr(predict, "predict_batch")
        assert hasattr(predict, "predict_directory")
        
    def test_manifest_builder(self, monkeypatch):
        """Test manifest builder parsing ASVspoof space-separated protocols."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock protocol text
            protocol_dir = os.path.join(temp_dir, "ASVspoof2019_LA_cm_protocols")
            os.makedirs(protocol_dir)
            protocol_path = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.train.trn.txt")
            
            with open(protocol_path, "w") as f:
                f.write("LA_0001 LA_T_0000001 - - bonafide\n")
                f.write("LA_0002 LA_T_0000002 - A01 spoof\n")
                
            # Create the corresponding empty flac files
            audio_dir = os.path.join(temp_dir, "ASVspoof2019_LA_train/flac")
            os.makedirs(audio_dir)
            sf.write(os.path.join(audio_dir, "LA_T_0000001.flac"), [0.1, -0.1], 16000)
            sf.write(os.path.join(audio_dir, "LA_T_0000002.flac"), [0.1, -0.1], 16000)
            
            # Monkeypatch the config constant of build_manifest
            import src.build_manifest as bm
            monkeypatch.setattr(bm, "LA_ROOT", temp_dir)
            
            # Execute manifest creation
            df = bm.build_manifest(split="train")
            assert len(df) == 2
            assert df.iloc[0]["label"] == "real"
            assert df.iloc[1]["label"] == "fake"
            assert df.iloc[1]["attack_type"] == "A01"
