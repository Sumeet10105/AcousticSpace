import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_audio
from src.spectral_features import extract_log_mel

manifest_path = project_root / "dataset" / "manifest.csv"
df = pd.read_csv(manifest_path)

mel_list = []
labels = []
splits = []
filepaths = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting mel spectrograms"):
    try:
        y, sr = load_audio(row["filepath"])
        mel = extract_log_mel(y, sr)
        mel_list.append(mel)
        labels.append(row["label"])
        splits.append(row["split"])
        filepaths.append(row["filepath"])
    except Exception as e:
        print(f"Failed on {row['filepath']}: {e}")

mel_array = np.stack(mel_list)   # shape: (num_samples, 64, 126)
labels_array = np.array(labels)
splits_array = np.array(splits)

out_path = project_root / "dataset" / "mel_features.npz"
np.savez_compressed(
    out_path,
    mel=mel_array,
    label=labels_array,
    split=splits_array,
    filepath=np.array(filepaths)
)
print(f"Saved {len(mel_array)} mel spectrograms to {out_path}")
print("Mel array shape:", mel_array.shape)