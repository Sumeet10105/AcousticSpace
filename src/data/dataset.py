"""Dataset class for audio spoofing detection."""

<<<<<<< HEAD
import os
import logging
from typing import Callable, List, Optional, Tuple
=======
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple
>>>>>>> 01d9eff5e902e9e68f29504ca915f1ccf7624734

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.audio_loader import load_audio
<<<<<<< HEAD
from src.preprocessing.padding import pad_or_truncate_to_duration
from src.preprocessing.silence_removal import trim_silence
=======
>>>>>>> 01d9eff5e902e9e68f29504ca915f1ccf7624734

logger = logging.getLogger(__name__)


class AudioSpoofingDataset(Dataset):
    """PyTorch dataset for ASVspoof-style audio spoofing detection.

    Supports two metadata formats:
    1. Manifest CSV (``manifest_path``): columns ``filepath``, ``label``, ``split``
    2. ASVspoof protocol file (``data_path`` + ``metadata_path``): space-separated protocol
    """
<<<<<<< HEAD

    def __init__(
        self,
        data_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        target_sr: int = 16000,
        duration_seconds: float = 4.0,
        augment: bool = False,
=======
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
>>>>>>> 01d9eff5e902e9e68f29504ca915f1ccf7624734
    ):
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.manifest_path = manifest_path
        self.split = split
        self.transform = transform
        self.validate_paths = validate_paths
        self.speakers = speakers
        self.max_samples = max_samples
<<<<<<< HEAD
        self.target_sr = target_sr
        self.duration_seconds = duration_seconds
        self.augment = augment

        self.data: List[str] = []
        self.labels: List[int] = []
        self.speaker_ids: List[str] = []
        self.system_ids: List[str] = []

        if manifest_path is not None:
            self._load_from_manifest(manifest_path, split)
        elif data_path is not None and metadata_path is not None:
            self._load_from_protocol(data_path, metadata_path)
        else:
            raise ValueError(
                "Provide either manifest_path or both data_path and metadata_path."
            )

        if max_samples is not None and len(self.data) > max_samples:
            indices = pd.Series(range(len(self.data))).sample(
                n=max_samples, random_state=42
            ).tolist()
            self.data = [self.data[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.speaker_ids = [self.speaker_ids[i] for i in indices]
            self.system_ids = [self.system_ids[i] for i in indices]

    def _load_from_manifest(self, manifest_path: str, split: str) -> None:
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        df = pd.read_csv(manifest_path)
        required = {"filepath", "label", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        df = df[df["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)
        if df.empty:
            logger.warning("No samples found for split '%s' in %s", split, manifest_path)

        for _, row in df.iterrows():
            filepath = str(row["filepath"])
            label = int(row["label"])
            self.data.append(filepath)
            self.labels.append(label)
            self.speaker_ids.append(str(row.get("speaker_id", "unknown")))
            self.system_ids.append(str(row.get("system_id", "unknown")))

    def _load_from_protocol(self, data_path: str, metadata_path: str) -> None:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Protocol file not found: {metadata_path}")

        df = pd.read_csv(
            metadata_path,
            sep=" ",
            header=None,
            names=["speaker_id", "audio_file", "col3", "system_id", "key"],
        )

        for _, row in df.iterrows():
            filename = f"{row['audio_file']}.flac"
            filepath = os.path.join(data_path, filename)
            label = 0 if row["key"] == "bonafide" else 1
            self.data.append(filepath)
            self.labels.append(label)
            self.speaker_ids.append(str(row["speaker_id"]))
            self.system_ids.append(str(row["system_id"]))

    def _maybe_augment(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return waveform
        from src.preprocessing.augmentation import add_gaussian_noise, random_gain

        waveform = random_gain(waveform, min_gain=0.8, max_gain=1.2)
        waveform = add_gaussian_noise(waveform, noise_factor=0.005)
        return waveform
=======

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
>>>>>>> 01d9eff5e902e9e68f29504ca915f1ccf7624734

    def __len__(self) -> int:
        return len(self.data)

<<<<<<< HEAD
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filepath = self.data[idx]
        label = self.labels[idx]

        try:
            waveform, sr = load_audio(filepath, target_sr=self.target_sr, mono=True)
            waveform = trim_silence(waveform, threshold_db=-45.0)
            waveform = pad_or_truncate_to_duration(
                waveform,
                sample_rate=sr,
                duration_seconds=self.duration_seconds,
            )
            waveform = self._maybe_augment(waveform)
        except Exception as exc:
            logger.warning("Failed to load %s: %s. Using silent fallback.", filepath, exc)
            target_samples = int(self.target_sr * self.duration_seconds)
            waveform = torch.zeros(1, target_samples)
=======
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
>>>>>>> 01d9eff5e902e9e68f29504ca915f1ccf7624734

        if self.transform:
            waveform = self.transform(waveform)

<<<<<<< HEAD
        return waveform, torch.tensor(label, dtype=torch.long)
=======
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
>>>>>>> 01d9eff5e902e9e68f29504ca915f1ccf7624734
