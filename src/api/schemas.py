"""API request/response schemas."""

from pydantic import BaseModel
from typing import Optional


class AudioUpload(BaseModel):
    """Audio upload request schema."""
    file_name: str
    file_size: int


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    prediction: str  # "real" or "fake"
    confidence: float
    score: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
