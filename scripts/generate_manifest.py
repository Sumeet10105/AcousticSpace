#!/usr/bin/env python3
"""Generate train/val/test manifest CSV from ASVspoof protocol files."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data.splitter import speaker_stratified_split


def parse_protocol(protocol_path: str, data_root: str) -> list:
    records = []
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id, audio_file, _, system_id, key = parts[:5]
            label = 0 if key == "bonafide" else 1
            filepath = os.path.join(data_root, f"{audio_file}.flac")
            records.append(
                {
                    "filepath": filepath,
                    "label": label,
                    "speaker_id": speaker_id,
                    "system_id": system_id,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ASVspoof manifest CSV")
    parser.add_argument("--protocol", required=True, help="Path to ASVspoof protocol file")
    parser.add_argument("--data-root", required=True, help="Directory containing FLAC files")
    parser.add_argument("--output", default="datasets/metadata/manifest.csv")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = parse_protocol(args.protocol, args.data_root)
    train, val, test = speaker_stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    for split_name, split_records in [("train", train), ("val", val), ("test", test)]:
        for record in split_records:
            record["split"] = split_name

    manifest = train + val + test
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(output_path, index=False)
    print(f"Wrote {len(manifest)} rows to {output_path}")


if __name__ == "__main__":
    main()
