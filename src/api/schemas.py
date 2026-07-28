"""API request/response schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field


class AudioUpload(BaseModel):
    """Audio upload request schema."""

    file_name: str
    file_size: int


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    prediction: str = Field(..., description="Classification label: real or fake")
    confidence: float = Field(..., ge=0.0, le=1.0)
    score: float = Field(..., ge=0.0, le=1.0, description="Probability of fake/spoof class")
    timestamp: str
    latency_ms: Optional[float] = None
    model_name: Optional[str] = None
    feature_importance: Optional[List[float]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool = False
