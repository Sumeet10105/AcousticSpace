# src/build_manifest.py
import os
import pandas as pd

LA_ROOT = "dataset/raw/LA"   # wherever you unzipped LA.zip

def build_manifest(split="train"):
    """
    split: 'train', 'dev', or 'eval'
    """
    protocol_map = {
        "train": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "dev":   "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "eval":  "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    }
    audio_dir_map = {
        "train": "ASVspoof2019_LA_train/flac",
        "dev":   "ASVspoof2019_LA_dev/flac",
        "eval":  "ASVspoof2019_LA_eval/flac",
    }

    protocol_path = os.path.join(LA_ROOT, protocol_map[split])
    audio_dir = os.path.join(LA_ROOT, audio_dir_map[split])

    rows = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            speaker_id, filename, _, attack_type, label = parts
            filepath = os.path.join(audio_dir, filename + ".flac")
            rows.append({
                "filepath": filepath,
                "label": "real" if label == "bonafide" else "fake",
                "attack_type": attack_type,   # "-" for real, A01-A19 for fake
                "speaker_id": speaker_id,
                "split": split,
            })

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    all_splits = [build_manifest(s) for s in ["train", "dev", "eval"]]
    full_df = pd.concat(all_splits, ignore_index=True)
    full_df.to_csv("dataset/manifest.csv", index=False)
    print(f"Wrote {len(full_df)} entries to data/manifest.csv")
    print(full_df["label"].value_counts())