"""Dataset class for audio spoofing detection."""

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.audio_loader import load_audio

logger = logging.getLogger(__name__)


class AudioSpoofingDataset(Dataset):
    """
    Dataset class for audio spoofing detection.

    Supports:
    - ASVspoof protocol files
    - CSV metadata files
    - Lazy loading
    - Optional transforms
    - Speaker filtering
    - Path validation
    - CPU-friendly dataset subsampling
    """

    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        validate_paths: bool = True,
        speakers: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.split = split
        self.transform = transform
        self.validate_paths = validate_paths
        self.speakers = speakers
        self.max_samples = max_samples

        self.data: List[Dict] = []

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata from either CSV or ASVspoof protocol."""

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        ext = os.path.splitext(self.metadata_path)[1].lower()

        if ext == ".csv":
            self._load_csv()

        else:
            self._load_protocol()

        logger.info(
            f"Loaded {len(self.data)} samples for split '{self.split}'."
        )

    def _load_protocol(self):
        """Load official ASVspoof protocol."""

        df = pd.read_csv(
            self.metadata_path,
            sep=r"\s+",
            header=None,
            names=[
                "speaker_id",
                "audio_file",
                "unused",
                "system_id",
                "label",
            ],
        )

        if self.max_samples:
            df = df.sample(
                n=min(self.max_samples, len(df)),
                random_state=42,
            )

        for _, row in df.iterrows():

            filepath = os.path.join(
                self.data_path,
                f"{row.audio_file}.flac",
            )

            if self.validate_paths and not os.path.exists(filepath):
                continue

            if (
                self.speakers
                and row.speaker_id not in self.speakers
            ):
                continue

            self.data.append(
                {
                    "filepath": filepath,
                    "label": 0
                    if row.label.lower() == "bonafide"
                    else 1,
                    "speaker_id": row.speaker_id,
                    "attack_type": row.system_id,
                    "split": self.split,
                }
            )

    def _load_csv(self):
        """Load CSV manifest."""

        df = pd.read_csv(self.metadata_path)

        split_map = {
            "train": "train",
            "val": "dev",
            "dev": "dev",
            "test": "eval",
            "eval": "eval",
        }

        target = split_map.get(
            self.split.lower(),
            self.split,
        )

        if "split" in df.columns:
            df = df[
                df["split"].str.lower() == target.lower()
            ]

        if self.speakers and "speaker_id" in df.columns:
            df = df[
                df["speaker_id"].isin(self.speakers)
            ]

        if self.max_samples:
            df = df.sample(
                n=min(self.max_samples, len(df)),
                random_state=42,
            )

        label_map = {
            "bonafide": 0,
            "real": 0,
            "spoof": 1,
            "fake": 1,
        }

        for _, row in df.iterrows():

            filepath = row["filepath"]

            if not os.path.isabs(filepath):
                filepath = os.path.join(
                    self.data_path,
                    filepath,
                )

            filepath = os.path.normpath(filepath)

            if self.validate_paths and not os.path.exists(filepath):
                continue

            self.data.append(
                {
                    "filepath": filepath,
                    "label": label_map.get(
                        str(row["label"]).lower(),
                        1,
                    ),
                    "speaker_id": row.get(
                        "speaker_id",
                        "unknown",
                    ),
                    "attack_type": row.get(
                        "attack_type",
                        "-",
                    ),
                    "split": row.get(
                        "split",
                        self.split,
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if idx >= len(self.data):
            raise IndexError(
                "Dataset index out of range."
            )

        sample = self.data[idx]

        try:
            waveform, _ = load_audio(
                sample["filepath"],
                target_sr=16000,
                mono=True,
            )

        except Exception as e:

            logger.warning(
                f"Failed to load {sample['filepath']}: {e}"
            )

            waveform = torch.zeros(
                1,
                64000,
            )

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

    def get_metadata(
        self,
        idx: int,
    ) -> Dict:

        if idx >= len(self.data):
            raise IndexError(
                "Dataset index out of range."
            )

        return self.data[idx]