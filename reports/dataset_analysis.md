# ASVspoof 2019 Logical Access (LA) Dataset Analysis Report

This report presents a thorough, production-quality analysis of the ASVspoof 2019 Logical Access dataset. The analysis automatically discovered the folder hierarchy, metadata, protocol file structure, train/dev/test splits, audio formats, labels, speaker IDs, attack IDs, spoof systems, duplicates, and class imbalance.

## Folder Hierarchy & Dataset Discovery
The dataset is structured inside `datasets/LA/LA/` as follows:
- **`ASVspoof2019_LA_cm_protocols/`**: Contains key ascii text protocol mapping files for training, development, and evaluation.
- **`ASVspoof2019_LA_train/flac/`**: FLAC audio files for training (`LA_T_*.flac`).
- **`ASVspoof2019_LA_dev/flac/`**: FLAC audio files for development (`LA_D_*.flac`).
- **`ASVspoof2019_LA_eval/flac/`**: FLAC audio files for evaluation (`LA_E_*.flac`).

## Protocol & Metadata Format
Each protocol file has columns structured as:
`SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY`
1. **SPEAKER_ID**: E.g., `LA_0079`
2. **AUDIO_FILE_NAME**: E.g., `LA_T_1138215`
3. **SYSTEM_ID**: Speech spoofing system ID (`A01` to `A19`), or `-` for bonafide speech.
4. **KEY**: `bonafide` (genuine speech) or `spoof` (spoofed speech).

---

## Dataset Statistics

### 1. Data Splits & File Counts
| Split | Protocol Entries | FLAC Files Found | Missing Files | Speaker Count |
|---|---|---|---|---|
| **Train** | 25380 | 25380 | 0 | 20 |
| **Dev** | 24844 | 24844 | 0 | 20 |
| **Eval** | 71237 | 71237 | 0 | 67 |

### 2. Class Distributions & Imbalance
| Split | Bonafide (Real) | Spoof (Fake) | Imbalance Ratio (Spoof:Bonafide) |
|---|---|---|---|
| **Train** | 2580 | 22800 | 8.84:1 |
| **Dev** | 2548 | 22296 | 8.75:1 |
| **Eval** | 7355 | 63882 | 8.69:1 |

### 3. Spoofing Attack System Distribution
* **Train Split**:
  - Contains systems: A01 to A06.
  - Distribution: A01: 3800, A02: 3800, A03: 3800, A04: 3800, A05: 3800, A06: 3800
* **Dev Split**:
  - Contains systems: A01 to A06.
  - Distribution: A01: 3716, A02: 3716, A03: 3716, A04: 3716, A05: 3716, A06: 3716
* **Eval Split**:
  - Contains systems: A07 to A19 (unseen spoofing algorithms to evaluate generalization).
  - Distribution: A07: 4914, A08: 4914, A09: 4914, A10: 4914, A11: 4914, A12: 4914, A13: 4914, A14: 4914, A15: 4914, A16: 4914, A17: 4914, A18: 4914, A19: 4914

---

## Audio File Specifications
Based on a randomized header scan of the audio files:
* **Audio Format**: FLAC
* **Sample Rate**: 16000 Hz (mono channel)
* **Bit Depth**: 16-bit
* **Duration Statistics (seconds)**:
  - **Train**: Mean = 3.45s, Min = 0.76s, Max = 12.78s
  - **Dev**: Mean = 3.54s, Min = 0.75s, Max = 9.77s
  - **Eval**: Mean = 3.12s, Min = 0.70s, Max = 10.82s

## Cleanliness & Integrity Check
* **Corrupted Files**: None detected.
* **Duplicate Files**: Checked 1000 random files, found 0 duplicates.
* **Missing Labels**: None. Every protocol line has a speaker, system, and class label.

---

## Visualizations Generated
1. **Class Distribution**: Saved to `plots/class_distribution.png`
2. **Speaker Counts**: Saved to `plots/speaker_distribution_train.png`
3. **Attack Systems**: Saved to `plots/attack_distribution_dev.png`
4. **Audio Durations**: Saved to `plots/duration_histogram.png`
5. **Waveforms**: Saved to `plots/waveform_bonafide.png` and `plots/waveform_spoof.png`
6. **Spectrograms**: Saved to `plots/spectrogram_bonafide.png` and `plots/spectrogram_spoof.png`
7. **Mel Spectrograms**: Saved to `plots/melspectrogram_bonafide.png` and `plots/melspectrogram_spoof.png`
8. **MFCCs**: Saved to `plots/mfcc_bonafide.png` and `plots/mfcc_spoof.png`
