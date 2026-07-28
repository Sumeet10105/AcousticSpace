# AcousticSpace

Deepfake audio detection via **Room Impulse Response (RIR)**, environmental acoustics, reverberation, breathing cadence, and spectrogram analysis — not voice artifacts alone.

## Features

- Multi-modal feature extraction: MFCC, mel spectrogram, chroma, spectral, RMS, RIR, breathing, waveform stats
- Models: Random Forest, XGBoost, LightGBM, SVM, Logistic Regression, MLP, CNN, CRNN, HuggingFace AST
- FastAPI inference API with health, predict, and model listing endpoints
- React analyst dashboard (`frontend/`)
- Explainable AI: Grad-CAM, Integrated Gradients, Attention Rollout
- Docker deployment and GitHub Actions CI

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset (ASVspoof 2019 LA)

Download ASVspoof 2019 LA and place files under `datasets/`. Generate a manifest:

```bash
python scripts/generate_manifest.py \
  --protocol datasets/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt \
  --data-root datasets/LA/LA/ASVspoof2019_LA_train/flac \
  --output datasets/metadata/manifest.csv
```

### 3. Train

```bash
python training/train.py --config configs/config.yaml
```

### 4. Run API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Analyst Dashboard

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL=http://localhost:8000` in `frontend/.env`.

### 6. Docker

```bash
docker compose up --build
```

## Project Layout

```
src/           Core library (features, models, API, XAI)
training/      CLI train/evaluate/inference/export
notebooks/     Research pipeline (01–09)
tests/         Pytest suite
frontend/      React analyst dashboard
configs/       YAML configuration
best_model/    Production model artifacts
models/        All trained checkpoints
scripts/       Utility scripts (manifest generation)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health and model availability |
| `/predict` | POST | Upload audio, get classification + confidence |
| `/models` | GET | List available trained models |

## Environment Variables

Copy `.env.example` to `.env`. Key variables:

- `MAX_UPLOAD_MB` — upload size limit (default 50)
- `INFERENCE_DEVICE` — `auto`, `cpu`, or `cuda`
- `ALLOWED_ORIGINS` — CORS origins for API

## Tests

```bash
pytest tests/ -v --cov=src
```

## License

MIT — see [LICENSE](LICENSE).
