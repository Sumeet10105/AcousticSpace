"""Tests for the feature extraction modules."""

import pytest
import torch

from src.features import (
    compute_spectrogram,
    compute_mel_spectrogram,
    compute_mfcc,
    extract_spectral_features,
    extract_chroma,
    extract_rms_energy,
    extract_rir_features,
    extract_breathing_features,
    extract_waveform_features,
    align_temporal_dimension,
    standardize_features,
    fuse_features,
    generate_spoof_explanation,
)


@pytest.fixture
def dummy_audio():
    """Generates a dummy 1-second 16kHz mono audio waveform."""
    torch.manual_seed(42)
    return torch.randn(1, 16000)


class TestFeatures:
    """Feature extraction tests."""
    
    def test_spectrogram(self, dummy_audio):
        """Test spectrogram computation."""
        spec = compute_spectrogram(dummy_audio, n_fft=512, hop_length=160)
        assert spec.shape[0] == 1  # 1 channel
        assert spec.shape[1] == 257  # n_fft // 2 + 1 frequency bins
        assert spec.shape[2] == 101  # (16000 // 160) + 1 frames
        assert not torch.isnan(spec).any()
        
    def test_mel_spectrogram(self, dummy_audio):
        """Test mel spectrogram computation."""
        mel_spec = compute_mel_spectrogram(
            dummy_audio, 
            sample_rate=16000, 
            n_fft=512, 
            hop_length=160, 
            n_mels=64
        )
        assert mel_spec.shape[0] == 1
        assert mel_spec.shape[1] == 64  # n_mels
        assert mel_spec.shape[2] == 101  # frames
        
    def test_mfcc(self, dummy_audio):
        """Test MFCC computation."""
        mfcc = compute_mfcc(
            dummy_audio, 
            sample_rate=16000, 
            n_mfcc=13, 
            n_fft=512, 
            hop_length=160, 
            n_mels=64
        )
        assert mfcc.shape[0] == 1
        assert mfcc.shape[1] == 13  # n_mfcc
        assert mfcc.shape[2] == 101  # frames
        
    def test_spectral_features(self, dummy_audio):
        """Test spectral features extraction."""
        spec_feats = extract_spectral_features(
            dummy_audio, 
            sample_rate=16000, 
            n_fft=512, 
            hop_length=160
        )
        assert spec_feats.shape[0] == 1
        assert spec_feats.shape[1] == 10  # 1 (centroid) + 1 (bandwidth) + 1 (rolloff) + 7 (contrast)
        assert spec_feats.shape[2] == 101
        
    def test_chroma(self, dummy_audio):
        """Test chroma extraction."""
        chroma = extract_chroma(
            dummy_audio, 
            sample_rate=16000, 
            n_fft=512, 
            hop_length=160, 
            n_chroma=12
        )
        assert chroma.shape[0] == 1
        assert chroma.shape[1] == 12  # n_chroma
        assert chroma.shape[2] == 101
        
    def test_rms_energy(self, dummy_audio):
        """Test RMS energy extraction."""
        rms = extract_rms_energy(dummy_audio, frame_length=512, hop_length=160)
        assert rms.shape[0] == 1
        assert rms.shape[1] == 1  # 1 dim of RMS
        assert rms.shape[2] == 97
        
    def test_rir_features(self, dummy_audio):
        """Test Room Impulse Response feature estimation."""
        rir = extract_rir_features(
            dummy_audio, 
            sample_rate=16000, 
            frame_length=512, 
            hop_length=160
        )
        assert rir.shape[0] == 1
        assert rir.shape[1] == 4  # rt60, edt, c50, c80
        assert rir.shape[2] == 97
        
    def test_breathing_features(self, dummy_audio):
        """Test breathing pattern features."""
        breath = extract_breathing_features(
            dummy_audio, 
            sample_rate=16000, 
            frame_length=512, 
            hop_length=160
        )
        assert breath.shape[0] == 1
        assert breath.shape[1] == 4  # zcr, flatness, high_ratio, slope_centroid
        assert breath.shape[2] == 97  # Librosa frame sizing might differ slightly (16000 - 512) // 160 + 1 = 97
        
    def test_waveform_features(self, dummy_audio):
        """Test time-domain statistical descriptors."""
        wave_feats = extract_waveform_features(dummy_audio, frame_length=512, hop_length=160)
        assert wave_feats.shape[0] == 1
        assert wave_feats.shape[1] == 5  # zcr, mean, std, skewness, kurtosis
        assert wave_feats.shape[2] == 97
        
    def test_feature_fusion_and_alignment(self, dummy_audio):
        """Test that different sized features align and concatenate properly."""
        # 101 frames
        mfcc = compute_mfcc(dummy_audio, n_mfcc=13)
        # 97 frames (due to padding/different frame sizing in librosa)
        breath = extract_breathing_features(dummy_audio, frame_length=512, hop_length=160)
        
        # Fusing without specifying target_time_steps should align to maximum (101)
        fused = fuse_features([mfcc, breath])
        assert fused.shape[0] == 1
        assert fused.shape[1] == 13 + 4  # 17 combined features
        assert fused.shape[2] == 101  # maximum of 101 and 97
        
        # Fusing with specific target time steps (e.g. 50)
        fused_50 = fuse_features([mfcc, breath], target_time_steps=50)
        assert fused_50.shape[2] == 50
        
        # Test standardisation
        std_feats = standardize_features(fused_50)
        assert std_feats.shape == fused_50.shape
        # mean should be ~0, std should be ~1
        assert torch.allclose(torch.mean(std_feats, dim=2), torch.zeros(1, 17), atol=1e-3)
        
    def test_explanation_generation(self):
        """Test generating detailed textual audit explanations."""
        # 113 features, 100 frames
        fused_raw = torch.zeros(1, 113, 100)
        
        # Test Fake explanation
        explanation_fake = generate_spoof_explanation(fused_raw, "FAKE", 0.95)
        assert "Classified as FAKE" in explanation_fake
        assert "confidence: 95.00%" in explanation_fake
        
        # Test Real explanation
        explanation_real = generate_spoof_explanation(fused_raw, "REAL", 0.99)
        assert "Classified as REAL" in explanation_real
        assert "confidence: 99.00%" in explanation_real
