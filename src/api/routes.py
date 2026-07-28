import json
import logging
import os
import pickle
import tempfile
from datetime import datetime

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.data.validator import validate_uploaded_file
from src.models import AudioSpectrogramTransformer, CRNN, ResNetCNN
from src.single_predict import predict_single_file

from .schemas import HealthResponse, PredictionResponse

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "50"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

_best_model = None
_best_model_config = None


def _resolve_device() -> str:
    preferred = os.getenv("INFERENCE_DEVICE", "auto")
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return preferred


def get_best_model():
    global _best_model, _best_model_config
    if _best_model is not None:
        return _best_model, _best_model_config

    config_path = "best_model/best_model_config.json"
    if not os.path.exists(config_path):
        raise HTTPException(
            status_code=503,
            detail=(
                "Best model not available. Run model comparison notebook or "
                "place artifacts in best_model/."
            ),
        )

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            _best_model_config = json.load(handle)

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
            pkl_path = "best_model/best_model.pkl"
            with open(pkl_path, "rb") as handle:
                _best_model = pickle.load(handle)

        return _best_model, _best_model_config
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to load best model: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to load best model: {exc}") from exc


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = os.path.exists("best_model/best_model_config.json")
    return {
        "status": "healthy",
        "version": "0.1.0",
        "model_loaded": model_loaded,
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict whether uploaded audio is real or deepfake/spoof."""
    suffix = os.path.splitext(file.filename or ".flac")[1] or ".flac"
    content = await file.read()
    valid, reason = validate_uploaded_file(file.filename or "upload", content, MAX_UPLOAD_MB)
    if not valid:
        raise HTTPException(status_code=400, detail=reason)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        model, config = get_best_model()
        device = _resolve_device()
        result = predict_single_file(tmp_path, model, device=device)

        pred_label = result["prediction"].lower()
        importance = result.get("feature_importance")
        if importance is not None:
            importance = importance.tolist()

        return PredictionResponse(
            prediction=pred_label,
            confidence=float(result["confidence"]),
            score=float(result["probability"]),
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=float(result.get("latency_ms", 0.0)),
            model_name=config.get("model_name"),
            feature_importance=importance,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/models")
async def list_models():
    """List available trained models and current best model metadata."""
    models_list = []

    for baseline in ["random_forest", "xgboost", "lightgbm", "svm", "mlp", "logistic_regression"]:
        if os.path.exists(f"models/{baseline}.pkl"):
            models_list.append({"name": baseline, "type": "traditional_ml", "status": "available"})

    for dl_model in ["cnn", "crnn", "ast"]:
        if os.path.exists(f"models/{dl_model}_best.pt"):
            models_list.append({"name": dl_model, "type": "deep_learning", "status": "available"})

    best_info = None
    config_path = "best_model/best_model_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                best_info = json.load(handle)
        except Exception:
            pass

    return {"models": models_list, "best_model": best_info}
