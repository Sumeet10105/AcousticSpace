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

print(df[df["split"] == "eval"]["label"].value_counts())

import soundfile as sf

def check_integrity(df):
    bad_files = []
    for fp in df["filepath"]:
        try:
            info = sf.info(fp)
        except Exception as e:
            bad_files.append((fp, str(e)))
    return bad_files

sample_df = df.sample(200, random_state=42)
bad = check_integrity(sample_df)
print(f"Corrupt/missing files: {len(bad)} out of {len(sample_df)}")

#bad_full = check_integrity(df)
#print(f"Corrupt/missing files: {len(bad_full)} out of {len(df)}")


#import json
#with open(project_root / "docs" / "eda_bad_files.json", "w") as f:
#    json.dump(bad_full, f, indent=2)

srs = [sf.info(fp).samplerate for fp in df["filepath"].sample(200, random_state=42)]
print(pd.Series(srs).value_counts())

from src.feature_extraction import extract_rir_features

t60_real = extract_rir_features(y_real, sr_real)
t60_fake = extract_rir_features(y_fake, sr_fake)
print("Real T60:", t60_real)
print("Fake T60:", t60_fake)

features_real = extract_rir_features(y_real, sr_real)
features_fake = extract_rir_features(y_fake, sr_fake)
print("Real features:", features_real)
print("Fake features:", features_fake)

import pandas as pd
feat_df = pd.read_csv(project_root / "dataset" / "rir_features.csv")
print(feat_df.groupby("label")["t60"].describe())
print(feat_df.groupby("label")["decay_slope"].describe())

print("Total rows:", len(feat_df))
print("\nNaN counts:")
print(feat_df.isna().sum())

print("\nLabel counts:")
print(feat_df["label"].value_counts())

print("\nT60 by label (mean):")
print(feat_df.groupby("label")["t60"].mean())

print("\nDecay slope by label (mean):")
print(feat_df.groupby("label")["decay_slope"].mean())

print((feat_df["t60"] > 10).sum(), "rows with T60 > 10s")
print((feat_df["decay_slope"] < -100).sum(), "rows with slope < -100 dB/s")

feat_df_clean = feat_df[
    (feat_df["t60"] <= 10) &
    (feat_df["decay_slope"] >= -100)
]
print(f"Kept {len(feat_df_clean)} of {len(feat_df)} rows")

feat_df_clean.to_csv(project_root / "dataset" / "rir_features.csv", index=False)

from src.spectral_features import extract_log_mel

mel_real = extract_log_mel(y_real, sr_real)
mel_fake = extract_log_mel(y_fake, sr_fake)

print("Real mel shape:", mel_real.shape)
print("Fake mel shape:", mel_fake.shape)