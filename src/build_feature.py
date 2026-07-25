import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_audio
from src.feature_extraction import extract_rir_features

manifest_path = project_root / "dataset" / "manifest.csv"
df = pd.read_csv(manifest_path)


records = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
    try:
        y, sr = load_audio(row["filepath"])
        feats = extract_rir_features(y, sr)
        feats["filepath"] = row["filepath"]
        feats["label"] = row["label"]
        feats["split"] = row["split"]
        records.append(feats)
    except Exception as e:
        print(f"Failed on {row['filepath']}: {e}")

features_df = pd.DataFrame(records)
out_path = project_root / "dataset" / "rir_features.csv"
features_df.to_csv(out_path, index=False)
print(f"Saved {len(features_df)} feature rows to {out_path}")

