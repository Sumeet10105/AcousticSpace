# AcousticSpace Architecture

## Overview

AcousticSpace is a deep learning-based audio spoofing detection system. It uses multiple neural network architectures and audio feature extraction techniques to identify synthetic/spoofed audio samples.

## System Architecture

### High-Level Architecture
```
Audio Input
    ↓
Preprocessing (normalize, resample, silence removal)
    ↓
Feature Extraction (spectrogram, MFCC, RIR, breathing patterns)
    ↓
Feature Fusion
    ↓
Neural Network (AST / CNN)
    ↓
Classification (Real or Fake)
```

### Module Structure

#### Core Module (`src/core/`)
- **config.py**: Configuration management
- **logger.py**: Logging utilities
- **exceptions.py**: Custom exceptions

#### Data Module (`src/data/`)
- **dataset.py**: PyTorch Dataset implementation
- **dataloader.py**: DataLoader utilities
- **splitter.py**: Train/val/test splitting
- **validator.py**: Data validation

#### Preprocessing Module (`src/preprocessing/`)
- **audio_loader.py**: Audio file loading
- **normalize.py**: Audio normalization
- **resample.py**: Sample rate resampling
- **silence_removal.py**: Silence detection and removal

#### Features Module (`src/features/`)
- **spectrogram.py**: Mel-spectrogram computation
- **mfcc.py**: MFCC feature extraction
- **rir.py**: Room impulse response features
- **breathing.py**: Breathing pattern features
- **feature_fusion.py**: Multi-feature fusion

#### Models Module (`src/models/`)
- **cnn.py**: CNN-based models (ResNet)
- **ast.py**: Audio Spectrogram Transformer
- **trainer.py**: Training loop
- **evaluator.py**: Model evaluation
- **predictor.py**: Inference pipeline

#### Metrics Module (`src/metrics/`)
- **accuracy.py**: Accuracy computation
- **confusion_matrix.py**: Confusion matrix
- **roc.py**: ROC curve
- **eer.py**: Equal Error Rate

#### Visualization Module (`src/visualization/`)
- **waveform.py**: Waveform plotting
- **spectrogram.py**: Spectrogram visualization
- **plots.py**: General plotting utilities

#### API Module (`src/api/`)
- **main.py**: FastAPI application
- **routes.py**: API endpoints
- **schemas.py**: Request/response schemas

### Data Flow

1. **Ingestion**: Raw audio files from ASVspoof datasets
2. **Preprocessing**: Normalize, resample, remove silence
3. **Feature Extraction**: Compute multiple feature types
4. **Training**: Train neural network model
5. **Evaluation**: Evaluate on test set
6. **Inference**: Make predictions on new audio

## Model Architectures

### Audio Spectrogram Transformer (AST)
- Pretrained on AudioSet
- Fine-tuned for spoofing detection
- Transformer-based architecture

### ResNet CNN
- Vision transformer applied to spectrograms
- Convolutional blocks for local feature extraction
- Global pooling for classification

## Training Pipeline

1. Load configuration from `configs/train.yaml`
2. Initialize model and optimizer
3. Create data loaders from ASVspoof dataset
4. Training loop with validation
5. Save best model checkpoint
6. Evaluate on test set

## Inference Pipeline

1. Load trained model checkpoint
2. Preprocess input audio
3. Extract features
4. Forward through model
5. Apply decision threshold
6. Return prediction and confidence
