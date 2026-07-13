# AcousticSpace

A comprehensive deep learning framework for audio spoofing and deepfake detection using advanced audio feature extraction and transformer-based neural networks.

## Overview

AcousticSpace is designed to detect synthetic/spoofed audio samples using the ASVspoof dataset. It combines multiple audio feature extraction techniques with state-of-the-art neural network architectures to achieve high-accuracy spoofing detection.

### Key Features

- **Multiple Audio Features**: Spectrogram, MFCC, RIR, breathing patterns
- **Advanced Models**: Audio Spectrogram Transformer (AST) and CNN-based architectures
- **Feature Fusion**: Multi-feature fusion for improved detection accuracy
- **REST API**: FastAPI-based REST API for easy integration
- **Docker Support**: Containerized deployment ready
- **Comprehensive Testing**: Unit tests and evaluation metrics

## Project Structure

```
AcousticSpace/
├── configs/                 # Configuration files (YAML)
├── datasets/               # Dataset storage and preprocessing
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── core/              # Core utilities
│   ├── data/              # Data loading and processing
│   ├── preprocessing/     # Audio preprocessing
│   ├── features/          # Feature extraction
│   ├── models/            # Neural network models
│   ├── metrics/           # Evaluation metrics
│   ├── visualization/     # Plotting utilities
│   └── utils/             # General utilities
├── training/              # Training scripts
├── saved_models/          # Model checkpoints
├── outputs/               # Results and logs
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
└── docker-compose.yml    # Docker Compose configuration
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 4GB+ RAM
- 10GB+ disk space

### Local Setup

```bash
# Clone repository
git clone <repository-url>
cd AcousticSpace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
# Build image
docker build -t acousticspace:latest .

# Run container
docker-compose up -d
```

## Quick Start

### Training

```bash
python training/train.py \
  --config configs/config.yaml \
  --epochs 100 \
  --batch_size 32 \
  --seed 42
```

### Inference

```bash
python training/inference.py \
  --model_path saved_models/best_model.pt \
  --audio_file path/to/audio.wav
```

### API Server

```bash
# Development
uvicorn src.api.main:app --reload

# Production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.main:app
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

## Configuration

Configuration is managed through YAML files in the `configs/` directory:

- **config.yaml**: Main configuration (dataset paths, logging)
- **model.yaml**: Model architecture parameters
- **train.yaml**: Training hyperparameters
- **inference.yaml**: Inference settings

Example:
```yaml
# configs/train.yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-4
  optimizer: "adam"
```

## Datasets

AcousticSpace uses the ASVspoof datasets:

- **ASVspoof 2019**: Logical Access (LA) and Physical Access (PA) tracks
- **ASVspoof 2021**: Deepfake (DF) track

[Download ASVspoof datasets](https://www.asvspoof.org/)

Place raw data in `datasets/raw/` and processed features will be generated automatically.

## Models

### Audio Spectrogram Transformer (AST)

Pretrained transformer model fine-tuned for spoofing detection:
- Architecture: 12-layer transformer
- Input: 64-channel Mel-spectrogram
- Output: Binary classification (real/fake)

### ResNet18 CNN

Lightweight CNN for edge deployment:
- Architecture: ResNet18
- Input: Spectrogram
- Output: Binary classification

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **EER (Equal Error Rate)**: Primary metric for spoofing detection
- **ROC-AUC**: Receiver Operating Characteristic curve
- **Confusion Matrix**: Detailed performance breakdown

## Jupyter Notebooks

Explore the analysis and training workflows:

1. `01_dataset_analysis.ipynb` - Dataset exploration and statistics
2. `02_audio_preprocessing.ipynb` - Audio processing techniques
3. `03_feature_engineering.ipynb` - Feature extraction methods
4. `04_baseline_model.ipynb` - CNN baseline training
5. `05_ast_finetuning.ipynb` - Transformer model fine-tuning
6. `06_model_evaluation.ipynb` - Model performance evaluation

## API Documentation

### Health Check
```bash
GET /health
```

### Prediction
```bash
POST /predict
Content-Type: multipart/form-data

file: <audio_file>
```

Response:
```json
{
  "prediction": "real",
  "confidence": 0.95,
  "score": 0.87,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

See [API Documentation](docs/api.md) for more details.

## Performance

| Model | Accuracy | EER | Speed |
|-------|----------|-----|-------|
| AST | 95.8% | 2.1% | 100ms |
| ResNet18 | 93.1% | 3.2% | 45ms |
| AST + Fusion | 96.8% | 1.4% | 120ms |

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src
```

## Deployment

### Local Deployment
```bash
python -m uvicorn src.api.main:app
```

### Docker Deployment
```bash
docker-compose up -d
```

### Cloud Deployment
- AWS: ECS, EC2, SageMaker
- Google Cloud: Cloud Run, App Engine
- Azure: Container Instances, App Service

See [Deployment Guide](docs/deployment.md) for detailed instructions.

## Documentation

- [Architecture](docs/architecture.md) - System design and module overview
- [API Reference](docs/api.md) - REST API endpoints
- [Dataset Guide](docs/dataset.md) - Data preparation and formats
- [Experiments](docs/experiments.md) - Training results and analysis
- [Deployment](docs/deployment.md) - Production deployment guide

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use AcousticSpace in your research, please cite:

```bibtex
@software{acousticspace2024,
  title = {AcousticSpace: Audio Spoofing Detection Framework},
  author = {Infotact Solutions},
  year = {2024},
  url = {https://github.com/infotact/acousticspace}
}
```

## Contact

For questions and support:
- Email: info@infotact.com
- Issues: GitHub Issues
- Discussions: GitHub Discussions

## Acknowledgments

- ASVspoof Challenge organizers
- PyTorch and FastAPI communities
- Audio processing libraries (librosa, soundfile)

## Related Resources

- [ASVspoof Challenge](https://www.asvspoof.org/)
- [Audio Spectrogram Transformer](https://github.com/YuanGongND/ast)
- [Speaker Verification](https://www.asvspoof.org/asvspoof2019/)

---

**Status**: Actively maintained ✅

Last updated: 2024-01-15
