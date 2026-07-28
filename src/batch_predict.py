"""Batch audio deepfake inference pipeline."""

import logging
import os
import torch
from typing import Any, Dict, List

from src.single_predict import predict_single_file

logger = logging.getLogger(__name__)


def predict_batch(
    audio_paths: List[str],
    model: torch.nn.Module,
    device: str = "cpu",
    threshold: float = 0.5,
    duration_seconds: float = 4.0
) -> Dict[str, Any]:
    """Runs inference on a list of audio files.

    Skips and logs files that are missing or corrupted.

    Args:
        audio_paths: List of absolute or relative paths to audio files.
        model: Loaded PyTorch model architecture.
        device: Device to run inference on (default: 'cpu').
        threshold: Classification decision threshold.
        duration_seconds: Target standardized audio duration.

    Returns:
        Dictionary containing summary statistics and a list of individual file predictions.
    """
    logger.info(f"Starting batch prediction on {len(audio_paths)} files.")
    results = []
    
    total_latency = 0.0
    successful_count = 0
    fake_count = 0
    real_count = 0
    
    for path in audio_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found during batch inference, skipping: {path}")
            continue
            
        try:
            res = predict_single_file(
                audio_path=path,
                model=model,
                device=device,
                threshold=threshold,
                duration_seconds=duration_seconds
            )
            results.append(res)
            
            # Update counters
            successful_count += 1
            total_latency += res["latency_ms"]
            
            if res["prediction"] == "FAKE":
                fake_count += 1
            else:
                real_count += 1
                
        except Exception as e:
            logger.warning(f"Skipping corrupted file {path} during batch inference: {str(e)}")
            continue
            
    avg_latency = total_latency / successful_count if successful_count > 0 else 0.0
    
    summary = {
        "total_files": len(audio_paths),
        "successful_predictions": successful_count,
        "fake_predictions": fake_count,
        "real_predictions": real_count,
        "average_latency_ms": avg_latency,
        "predictions": results
    }
    
    logger.info(
        f"Batch prediction completed: {successful_count}/{len(audio_paths)} succeeded. "
        f"Fake={fake_count}, Real={real_count}. Avg Latency={avg_latency:.2f} ms."
    )
    return summary


def predict_directory(
    directory_path: str,
    model: torch.nn.Module,
    device: str = "cpu",
    threshold: float = 0.5,
    duration_seconds: float = 4.0,
    supported_extensions: tuple = (".flac", ".wav", ".mp3")
) -> Dict[str, Any]:
    """Finds all audio files in a directory and runs batch prediction on them.

    Args:
        directory_path: Path to the directory to scan.
        model: Loaded PyTorch model architecture.
        device: Device to run inference on.
        threshold: Classification decision threshold.
        duration_seconds: Target standardized audio duration.
        supported_extensions: File extensions to filter for.

    Returns:
        Summary dictionary of prediction results.
    """
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        raise FileNotFoundError(f"Directory not found: {directory_path}")
        
    audio_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                audio_paths.append(os.path.join(root, file))
                
    return predict_batch(
        audio_paths=audio_paths,
        model=model,
        device=device,
        threshold=threshold,
        duration_seconds=duration_seconds
    )
