"""Tests for API module."""

import io
import json
import os
import pickle
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def synthetic_flac(tmp_path):
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "sample.flac"
    sf.write(path, audio, sample_rate)
    return path


@pytest.fixture
def mock_best_model(tmp_path, monkeypatch):
    """Install a lightweight sklearn model in a temporary best_model directory."""
    workdir = tmp_path / "workspace"
    best_dir = workdir / "best_model"
    best_dir.mkdir(parents=True)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    features = np.random.randn(20, 113 * 100)
    labels = np.random.randint(0, 2, size=20)
    model.fit(features, labels)

    with open(best_dir / "best_model.pkl", "wb") as handle:
        pickle.dump(model, handle)

    with open(best_dir / "best_model_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"model_name": "random_forest", "type": "traditional_ml", "input_shape": "fused_flat"},
            handle,
        )

    monkeypatch.chdir(workdir)
    return workdir


class TestAPI:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["version"] == "0.1.0"
        assert "model_loaded" in payload

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "AcousticSpace API"

    def test_models_endpoint(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        assert "models" in response.json()

    def test_predict_rejects_empty_file(self, client):
        response = client.post(
            "/predict",
            files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
        )
        assert response.status_code == 400

    def test_predict_rejects_unsupported_extension(self, client):
        response = client.post(
            "/predict",
            files={"file": ("bad.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_endpoint(self, client, synthetic_flac, mock_best_model):
        with open(synthetic_flac, "rb") as handle:
            response = client.post(
                "/predict",
                files={"file": ("sample.flac", handle, "audio/flac")},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["prediction"] in {"real", "fake"}
        assert 0.0 <= payload["confidence"] <= 1.0
        assert 0.0 <= payload["score"] <= 1.0
        assert "timestamp" in payload
        assert payload["model_name"] == "random_forest"
