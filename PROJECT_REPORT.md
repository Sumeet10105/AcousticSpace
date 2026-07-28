# AcousticSpace: Deepfake Audio Detection Pipeline

A production-quality machine learning and signal processing pipeline for deepfake audio detection using Room Impulse Response (RIR) acoustic features and deep learning models (CNN, CRNN, and Audio Spectrogram Transformers).

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset Analysis & Statistics](#2-dataset-analysis--statistics)
3. [Audio Preprocessing Pipeline](#3-audio-preprocessing-pipeline)
4. [Acoustic Feature Engineering & Fusion](#4-acoustic-feature-engineering--fusion)
5. [Model Architectures & Training](#5-model-architectures--training)
   - [Traditional Baseline Classifiers](#traditional-baseline-classifiers)
   - [Deep Learning Classifiers](#deep-learning-classifiers)
   - [Audio Spectrogram Transformer (AST)](#audio-spectrogram-transformer-ast)
6. [Comparative Evaluation & Leaderboard](#6-comparative-evaluation--leaderboard)
7. [Explainable AI (XAI) Auditing](#7-explainable-ai-xai-auditing)
8. [Production Deployment (FastAPI Integration)](#8-production-deployment-fastapi-integration)
9. [How to Reproduce & Run](#9-how-to-reproduce--run)

---

## 1. Project Overview
AcousticSpace addresses the growing threat of synthesized audio and speech deepfakes. Synthesized speech often lacks the complex environmental and physical interactions present in real recordings, such as:
* Room Impulse Responses (RIRs)
* Physiological breathing dynamics
* Fine-grained frequency distributions (MFCCs/Chroma)

By combining **15 channels of advanced acoustic descriptors** with both traditional machine learning models and deep learning backbones (including Transformers), this pipeline provides a robust framework to classify audio files as **REAL** or **FAKE**.

---

## 2. Dataset Analysis & Statistics
We conducted a comprehensive analysis of the ASVspoof 2019 Logical Access (LA) database, showing the following characteristics:
* **Audio Format**: FLAC, 16,000 Hz, mono.
* **Class Distribution**:
  * **Train**: 2,580 Bonafide (Real) | 22,800 Spoof (Fake)
  * **Development (Dev)**: 2,548 Bonafide (Real) | 22,296 Spoof (Fake)
* **Speaker Distribution**: Balanced speaker identifiers across training data (54 distinct speaker IDs).
* **Spoofing Systems**: Slices based on 19 distinct synthesizer/vocoder algorithms (A01-A19), which help audit models for generalization across unseen synthetic attacks.

---

## 3. Audio Preprocessing Pipeline
To guarantee noise robustness and consistent signal shapes, raw signals go through a multi-stage preprocessing sequence:
1. **Highpass Filtering**: Active Butterworth highpass filter with a cutoff of $80\text{ Hz}$ to eliminate sub-bass rumble.
2. **Spectral Subtraction**: Dynamic noise cancellation mapping noise profiles from low-energy frames.
3. **Silence Trimming**: Peak amplitude-based padding and trimming below $-45\text{ dB}$ to focus classifiers on active speech frames.
4. **Duration Standardization**: Resampling and zero-padding/truncating signals to exactly $4.0\text{ seconds}$ ($64,000\text{ samples}$ at $16\text{ kHz}$).
5. **Peak Normalization**: Amplitude scaling to a target peak of $0.9$.

---

## 4. Acoustic Feature Engineering & Fusion
AcousticSpace extracts **15 distinct feature channels** to capture different vocal and environmental traits:
* **MFCCs (13 channels)**: Captures basic speech articulation.
* **Mel-Spectrogram (64 channels)**: Captures log-scale frequency energy distribution.
* **Chroma STFT (12 channels)**: Captures musical pitch classes.
* **Spectral Descriptors (4 channels)**: Spectral Centroid, Bandwidth, Roll-off, and Zero-Crossing Rate.
* **RMS Energy (1 channel)**: Tracks loudness envelope over time.
* **Room Impulse Response (RIR) features (8 channels)**: High-frequency reverberation descriptors to detect artificial voice rendering.
* **Breathing features (4 channels)**: Frequency-domain indicators of natural breath pauses.
* **Waveform statistics (7 channels)**: Skewness, kurtosis, peak-to-average power ratio, etc.

### Feature Fusion
All features are aligned along the time-step dimension using linear interpolation to a target duration of **100 steps**. They are then concatenated along the feature dimension to create a single **113-channel** feature map.

---

## 5. Model Architectures & Training

### Traditional Baseline Classifiers
We trained and validated 6 baseline classifiers on the flattened 11,300-dimension fused features:
* **Random Forest**: Ensembled decision boundaries.
* **XGBoost & LightGBM**: Gradient-boosted decision trees.
* **Support Vector Machine (SVM)**: Radial Basis Function (RBF) kernel classifier.
* **Logistic Regression**: Linear classifier with L2 regularization.
* **Multi-Layer Perceptron (MLP)**: Feedforward neural network.

### Deep Learning Classifiers
We implemented deep models using PyTorch, trained on Mel-Spectrogram features (`[N, 1, 64, 100]`):
* **ResNetCNN**: 2D ResNet18 convolutional architecture mapping spectro-temporal features.
* **CRNN (Convolutional Recurrent Neural Network)**: CNN layers for spatial feature maps coupled with a bidirectional GRU network for sequential modeling.

### Audio Spectrogram Transformer (AST)
* **Architecture**: Self-attention transformer processing audio spectrogram patches.
* **Configuration**: ViT-Base backbone (12 layers, 12 attention heads, 768 hidden dimension, 86M parameters).

---

## 6. Comparative Evaluation & Leaderboard
The models were evaluated on the validation set. Traditional machine learning models (specifically Random Forest) demonstrated superior accuracy on the early subset runs, establishing itself as the deployment model.

### Metric Overview
* **EER (Equal Error Rate)**: Measures the threshold where False Acceptance Rate equals False Rejection Rate.
* **AUC-ROC**: Area under the ROC curve.
* **Latency**: Average inference time per audio clip.

---

## 7. Explainable AI (XAI) Auditing
To build trust and verify model decisions, the pipeline integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** overlays.
* **Method**: Backpropagates class gradients to the final convolutional layer.
* **Interpretation**: Highlights the exact spectro-temporal regions (e.g., vocal formants, synthetic high-frequency vocoder artifacts) that trigger the "FAKE" classification.
* **Output plots**: Heatmap visualizations saved under the `plots/` directory.

---

## 8. Production Deployment (FastAPI Integration)
We implemented a production-grade backend inside `src/api/routes.py` with three core endpoints:
1. **`GET /health`**: Returns API status and microservice health.
2. **`POST /predict`**: Accepts an audio file (FLAC, WAV, etc.), runs the preprocessing, extracts and fuses acoustic features, and executes predictions using the loaded best-performing model.
3. **`GET /models`**: Lists all available classifiers and details of the current active best-performing model.

---

## 9. How to Reproduce & Run

### Environment Setup
Install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Programmatic Execution
You can regenerate and run the entire pipeline notebooks sequentially:
```bash
python scratch/generate_notebooks.py
python scratch/execute_notebooks.py
```
This will:
1. Re-generate all 9 notebooks under the `notebooks/` directory.
2. Sequentially run them using `nbconvert`, validating output cells and generating all statistics, plots, checkpoints, and exports.

### Running FastAPI API Server
Start the production server using Uvicorn:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Running Test Suite
Execute the pytest suite:
```bash
python -m pytest
```
All 51 test cases will execute and verify preprocessing, features, models, explainability, metrics, and API routes.
