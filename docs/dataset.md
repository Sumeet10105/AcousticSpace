# Dataset Documentation

## Overview

AcousticSpace uses the ASVspoof datasets for training and evaluation. These are industry-standard benchmarks for spoofing and deepfake detection.

## ASVspoof Datasets

### ASVspoof 2019
Largest and most comprehensive audio spoofing dataset.

**Tracks:**
- **LA (Logical Access)**: Speaker verification system attacks
- **PA (Physical Access)**: Voice conversion and speech synthesis

**Statistics:**
- Train: ~16,000 utterances
- Dev: ~10,000 utterances
- Eval: ~13,000 utterances

**Audio:**
- Sample rate: 16 kHz
- Format: 16-bit PCM WAV
- Duration: ~10 seconds average

### ASVspoof 2021
Extended dataset with additional spoofing methods.

**Tracks:**
- **DF (Deepfake)**: Deepfake voices

## Data Structure

```
datasets/
├── raw/
│   ├── ASVspoof2019/
│   │   ├── LA/
│   │   │   ├── ASVspoof2019_LA_train/
│   │   │   ├── ASVspoof2019_LA_dev/
│   │   │   └── ASVspoof2019_LA_eval/
│   │   └── PA/
│   └── ASVspoof2021/
│       └── DF/
├── processed/
│   ├── waveforms/
│   ├── spectrograms/
│   ├── mfcc/
│   ├── rir/
│   ├── breathing/
│   └── features/
└── metadata/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── labels.csv
```

## Metadata Format

### CSV Structure
```
filename,label,split,duration
LA_T_1000001.wav,bonafide,train,10.2
LA_T_1000002.wav,spoof,train,9.8
```

**Fields:**
- `filename`: Audio file name
- `label`: bonafide (real) or spoof (fake)
- `split`: train/val/test
- `duration`: Audio duration in seconds

## Data Preprocessing

1. **Loading**: Load WAV files at 16 kHz
2. **Normalization**: Normalize to [-1, 1] range
3. **Silence Removal**: Remove leading/trailing silence
4. **Resampling**: Convert to target sample rate if needed

## Feature Extraction

### Mel-Spectrogram
- N_FFT: 512
- Hop Length: 160
- N_Mels: 64
- F_min: 50 Hz
- F_max: 8 kHz

### MFCC
- N_MFCC: 13
- N_FFT: 512
- Hop Length: 160

### RIR Features
- Impulse response analysis
- Energy decay curves

### Breathing Patterns
- Low-frequency energy detection
- Periodicity analysis

## Data Splits

- **Train**: 80% - For model training
- **Val**: 10% - For validation during training
- **Test**: 10% - For final evaluation

## Data Augmentation

During training:
- MixUp: Mix two samples with random weights
- SpecAugment: Mask time and frequency regions
- Random pitch shifting
- Background noise addition

## Getting Started

1. Download ASVspoof datasets from official website
2. Place in `datasets/raw/` directory
3. Run preprocessing script to generate features
4. Create metadata CSV files
5. Start training
