# Experiments and Results

## Overview

This document tracks experiments, configurations, and results from model training and evaluation.

## Experiment Log

### Experiment 1: AST Fine-tuning Baseline
**Date**: 2024-01-15
**Configuration**: 
- Model: Audio Spectrogram Transformer
- Epochs: 100
- Batch Size: 32
- Learning Rate: 1e-4
- Dataset: ASVspoof 2019 LA

**Results**:
- Train Accuracy: 98.5%
- Val Accuracy: 96.2%
- Test Accuracy: 95.8%
- EER: 2.1%

**Notes**: 
- Good baseline performance
- Model converged well
- No overfitting detected

### Experiment 2: CNN ResNet18 Baseline
**Date**: 2024-01-16
**Configuration**:
- Model: ResNet18
- Epochs: 100
- Batch Size: 32
- Learning Rate: 1e-3
- Dataset: ASVspoof 2019 LA

**Results**:
- Train Accuracy: 97.2%
- Val Accuracy: 94.5%
- Test Accuracy: 93.1%
- EER: 3.2%

**Notes**:
- Slightly lower than AST
- Faster inference time
- Good for edge deployment

### Experiment 3: Feature Fusion
**Date**: 2024-01-17
**Configuration**:
- Model: AST with Feature Fusion
- Features: Spectrogram + MFCC + RIR + Breathing
- Epochs: 100
- Batch Size: 32
- Dataset: ASVspoof 2019 LA + PA

**Results**:
- Train Accuracy: 99.1%
- Val Accuracy: 97.3%
- Test Accuracy: 96.8%
- EER: 1.4%

**Notes**:
- Best performance achieved
- Multi-task learning beneficial
- Computational cost increased

## Comparative Analysis

| Model | Accuracy | EER | Speed | Parameters |
|-------|----------|-----|-------|-----------|
| AST | 95.8% | 2.1% | 100ms | 86M |
| ResNet18 | 93.1% | 3.2% | 45ms | 11M |
| AST + Fusion | 96.8% | 1.4% | 120ms | 86M |

## Hyperparameter Tuning

### Learning Rate
- 1e-5: Slow convergence
- 1e-4: ✓ Optimal
- 1e-3: Unstable training

### Batch Size
- 16: More stable, slower training
- 32: ✓ Good balance
- 64: Faster training, less stable

### Warmup Epochs
- 0: Poor convergence
- 5: ✓ Optimal
- 10: No significant improvement

## Analysis

1. **AST is best for accuracy**: Fine-tuning pretrained AST yields highest accuracy
2. **Feature fusion improves EER**: Multi-feature approach significantly reduces equal error rate
3. **Trade-off between speed and accuracy**: ResNet18 is faster but less accurate
4. **Early stopping helps**: Prevents overfitting after ~80 epochs

## Recommendations

1. Use AST with feature fusion for production
2. Implement quantization for edge deployment
3. Monitor EER as primary metric
4. Continue training with larger datasets
5. Explore ensemble methods
