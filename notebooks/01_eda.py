import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_audio

# Load manifest
manifest_path = project_root / "dataset" / "manifest.csv"
df = pd.read_csv(manifest_path)

# Basic information
print("Dataset Shape:", df.shape)
print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nSplit Distribution:")
print(df.groupby("split")["label"].value_counts())

# Load one real and one fake sample
y_real, sr_real = load_audio(df[df["label"] == "real"].iloc[0]["filepath"])
y_fake, sr_fake = load_audio(df[df["label"] == "fake"].iloc[0]["filepath"])

print("\nReal Sample Length:", len(y_real))
print("Fake Sample Length:", len(y_fake))
print("Sample Rate:", sr_real)