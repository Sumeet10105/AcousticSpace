import os
import json
import pickle
import tempfile
import torch
import logging
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException
from .schemas import PredictionResponse, HealthResponse
from src.models import ResNetCNN, CRNN, AudioSpectrogramTransformer
from src.single_predict import predict_single_file

logger = logging.getLogger(__name__)
router = APIRouter()

_best_model = None
_best_model_config = None

def get_best_model():
    global _best_model, _best_model_config
    if _best_model is not None:
        return _best_model, _best_model_config
        
    config_path = "best_model/best_model_config.json"
    if not os.path.exists(config_path):
        raise HTTPException(
            status_code=503, 
            detail="Best model not available. Model training and evaluation comparisons must be completed first."
        )
        
    try:
        with open(config_path, "r") as f:
            _best_model_config = json.load(f)
            
        model_name = _best_model_config["model_name"]
        model_type = _best_model_config["type"]
        
        if model_type == "deep_learning":
            weights_path = "best_model/best_model.pt"
            if model_name == "cnn":
                model = ResNetCNN(num_classes=2, pretrained=False)
            elif model_name == "crnn":
                model = CRNN(num_classes=2, n_mels=64)
            elif model_name == "ast":
                model = AudioSpectrogramTransformer(num_classes=2, pretrained=False)
            else:
                raise ValueError(f"Unknown deep learning model: {model_name}")
                
            model.load_model(weights_path)
            model.eval()
            _best_model = model
        else:
            # Traditional classifier path
            pkl_path = "best_model/best_model.pkl"
            with open(pkl_path, "rb") as sf:
                _best_model = pickle.load(sf)
                
        return _best_model, _best_model_config
    except Exception as e:
        logger.error(f"Failed to load best model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load best model: {str(e)}")


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
    # Create temporary file to store uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as tmp:
        try:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read upload file: {str(e)}")
        
    try:
        # Load best model
        model, config = get_best_model()
        
        # Run prediction
        result = predict_single_file(tmp_path, model, device="cpu")
        
        # Clean up temp file
        os.remove(tmp_path)
        
        # Format response: map REAL to real, FAKE to fake
        pred_label = result["prediction"].lower()  # "real" or "fake"
        
        return PredictionResponse(
            prediction=pred_label,
            confidence=float(result["confidence"]),
            score=float(result["probability"]),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/models")
async def list_models():
    """List available models."""
    models_list = []
    
    # Check baseline models
    baselines = ['random_forest', 'xgboost', 'lightgbm', 'svm', 'mlp']
    for b in baselines:
        if os.path.exists(f"models/{b}.pkl"):
            models_list.append({"name": b, "type": "traditional_ml", "status": "available"})
            
    # Check deep learning models
    dls = ['cnn', 'crnn', 'ast']
    for d in dls:
        if os.path.exists(f"models/{d}_best.pt"):
            models_list.append({"name": d, "type": "deep_learning", "status": "available"})
            
    # Get current best model info
    best_info = None
    if os.path.exists("best_model/best_model_config.json"):
        try:
            with open("best_model/best_model_config.json", "r") as f:
                best_info = json.load(f)
        except Exception:
            pass
            
    return {
        "models": models_list,
        "best_model": best_info
    }

