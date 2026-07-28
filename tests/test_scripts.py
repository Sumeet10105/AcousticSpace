"""Tests for the training/evaluation/inference CLI scripts."""

import os
import subprocess
import sys
import tempfile
import pytest
import soundfile as sf
import numpy as np


class TestCLIScripts:
    """CLI Scripts integration tests."""
    
    def test_train_script_manifest_check(self):
        """Test that training script errors out gracefully when manifest is missing."""
        # Run train.py with a non-existent config pointing to missing manifest
        cmd = [
            sys.executable,
            "training/train.py",
            "--config", "configs/config.yaml",
            "--train_config", "configs/train.yaml",
            "--model_config", "configs/model.yaml"
        ]
        
        # Should exit with status code 1 since datasets manifest doesn't exist
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        assert "Metadata manifest not found" in res.stderr or "Metadata manifest not found" in res.stdout
        
    def test_inference_script_missing_args(self):
        """Test that inference script fails when both file and directory are missing."""
        cmd = [
            sys.executable,
            "training/inference.py",
            "--model_path", "nonexistent.pt"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        assert "specify either --audio_file or --audio_dir" in res.stderr or "specify either --audio_file or --audio_dir" in res.stdout
        
    def test_export_script_missing_checkpoint(self):
        """Test that export script fails when checkpoint does not exist."""
        cmd = [
            sys.executable,
            "training/export_model.py",
            "--model_path", "nonexistent.pt",
            "--export_format", "onnx"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1
        assert "Checkpoint weights file not found" in res.stderr or "Checkpoint weights file not found" in res.stdout
