"""Single file audio deepfake inference pipeline."""

import logging
import time
import os
import psutil
import torch
from typing import Any, Dict, Optional

from src.preprocessing import load_audio, pad_or_truncate_to_duration, trim_silence
from src.features import (
    compute_mel_spectrogram,
    standardize_features,
    compute_mfcc,
    extract_chroma,
    extract_spectral_features,
    extract_rms_energy,
    extract_rir_features,
    extract_breathing_features,
    extract_waveform_features,
    fuse_features,
)
from src.models import Predictor

logger = logging.getLogger(__name__)


def predict_single_file(
    audio_path: str,
    model: torch.nn.Module,
    device: str = "cpu",
    threshold: float = 0.5,
    duration_seconds: float = 4.0
) -> Dict[str, Any]:
    """Runs the inference pipeline on a single audio file and measures performance.

    Pipeline stages:
    1. Audio Loading (16 kHz, mono)
    2. Silence Trimming
    3. Padding / Truncating to fixed duration
    4. Feature Extraction (Mel Spectrogram)
    5. Standardization
    6. Forward pass & Probability calculation
    7. Metric Measurements (Latency, Memory)

    Args:
        audio_path: Path to the audio file to analyze.
        model: Loaded PyTorch model architecture.
        device: Device to run inference on (default: 'cpu').
        threshold: Decision threshold for classification (default: 0.5).
        duration_seconds: Target standardized audio duration in seconds.

    Returns:
        Dictionary containing prediction results and performance metrics.
    """
    # Start resource tracking
    start_time = time.perf_counter()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / (1024 ** 2)  # MB
    
    try:
        # 1. Load Audio
        waveform, sr = load_audio(audio_path, target_sr=16000, mono=True)
        
        # 2. Trim silence
        waveform = trim_silence(waveform, threshold_db=-45.0)
        
        # 3. Standardize duration
        waveform = pad_or_truncate_to_duration(waveform, sample_rate=sr, duration_seconds=duration_seconds)
        
        # 4. Feature engineering (MFCC, Mel-spectrogram, Chroma, Spectral, RMS, RIR, Breathing, Waveform)
        mfcc = compute_mfcc(waveform, sample_rate=sr, n_mfcc=13)
        mel = compute_mel_spectrogram(waveform, sample_rate=sr, n_mels=64)
        chroma = extract_chroma(waveform, sample_rate=sr)
        spectral = extract_spectral_features(waveform, sample_rate=sr)
        rms = extract_rms_energy(waveform)
        rir = extract_rir_features(waveform, sample_rate=sr)
        breath = extract_breathing_features(waveform, sample_rate=sr)
        wave_stats = extract_waveform_features(waveform)
        
        # 5. Fuse features (align to 100 time steps)
        fused_raw = fuse_features([mfcc, mel, chroma, spectral, rms, rir, breath, wave_stats], target_time_steps=100)
        
        # Standardize features
        fused = standardize_features(fused_raw.clone())
        
        # Select appropriate feature representation for the model type
        if hasattr(model, "predict_proba"):
            features = fused
        else:
            features = fused[:, :64, :]
        
        # 6. Predict
        if hasattr(model, "predict_proba"):
            # Traditional classifier path
            import pickle
            features_flat = features.cpu().numpy().reshape(1, -1)
            scaler_path = "best_model/scaler.pkl" if os.path.exists("best_model/scaler.pkl") else "models/scaler.pkl"
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, "rb") as sf:
                        scaler = pickle.load(sf)
                    features_flat = scaler.transform(features_flat)
                except Exception as se:
                    logger.warning(f"Failed to load/apply scaler: {se}")
                    
            prob_fake = float(model.predict_proba(features_flat)[0, 1])
            if prob_fake >= threshold:
                prediction = "FAKE"
                confidence = prob_fake
            else:
                prediction = "REAL"
                confidence = 1.0 - prob_fake
                
            prediction_result = {
                "prediction": prediction,
                "confidence": confidence,
                "probability": prob_fake
            }
        else:
            # PyTorch Deep Learning model path
            predictor = Predictor(model, device=device, threshold=threshold)
            prediction_result = predictor.predict(features)

        
        # End resource tracking
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss / (1024 ** 2)  # MB
        
        latency_ms = (end_time - start_time) * 1000.0
        memory_used_mb = max(0.0, end_memory - start_memory)
        
        # Build final response dict
        result = {
            "filepath": audio_path,
            "prediction": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
            "probability": prediction_result["probability"],
            "latency_ms": latency_ms,
            "memory_used_mb": memory_used_mb,
            "features": features,
            "fused_raw": fused_raw
        }
        
        # Simple feature importance heuristic:
        # Measure which parts of the spectrogram are most active
        # (average power per frequency bin)
        with torch.no_grad():
            feat_avg = torch.mean(features[0], dim=1)  # Mean energy across time frames per Mel bin
            result["feature_importance"] = feat_avg.cpu().numpy()
            
        logger.info(
            f"Prediction completed for {audio_path}: {result['prediction']} "
            f"({result['confidence']:.4f}) in {result['latency_ms']:.2f} ms."
        )
        return result
        
    except Exception as e:
        logger.error(f"Inference pipeline failed for {audio_path}: {str(e)}")
        raise e
