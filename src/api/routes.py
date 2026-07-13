"""API routes."""

from fastapi import APIRouter, File, UploadFile
from .schemas import PredictionResponse, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0"
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict audio spoofing.
    
    Args:
        file: Audio file to analyze
        
    Returns:
        Prediction result
    """
    # TODO: Implement prediction endpoint
    pass


@router.get("/models")
async def list_models():
    """List available models."""
    # TODO: Implement model listing
    pass
