"""Inference interface exposing single-file and batch prediction functions."""

from src.single_predict import predict_single_file
from src.batch_predict import predict_batch, predict_directory

__all__ = [
    "predict_single_file",
    "predict_batch",
    "predict_directory",
]
