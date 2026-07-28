"""Dataset class for audio spoofing detection."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.audio_loader import load_audio
from src.preprocessing.padding import pad_or_truncate_to_duration
from src.preprocessing.silence_removal import trim_silence

logger = logging.getLogger(__name__)


class AudioSpoofingDataset(Dataset):
    """PyTorch dataset for ASVspoof-style audio spoofing detection.

    Supports manifest CSV files and ASVspoof-style protocol files while keeping
    compatibility with older constructor signatures used by the project tests.
    """

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
        validate_paths: bool = True,
        speakers: Optional[List[str]] = None,
    ):
        self.data_path = data_path or ""
        self.metadata_path = metadata_path
        self.manifest_path = manifest_path
        self.split = split
        self.transform = transform
        self.validate_paths = validate_paths
        self.speakers = speakers or []
        self.max_samples = max_samples
        self.target_sr = target_sr
        self.duration_seconds = duration_seconds
        self.augment = augment

        self.data: List[Dict[str, Any]] = []
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
                n=max_samples,
                random_state=42,
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

        split_map = {
            "train": "train",
            "val": "val",
            "dev": "val",
            "test": "test",
            "eval": "test",
        }
        target_split = split_map.get(split.lower(), split.lower())

        df = df[df["split"].astype(str).str.lower() == target_split].reset_index(drop=True)
        if df.empty:
            logger.warning("No samples found for split '%s' in %s", split, manifest_path)

        for _, row in df.iterrows():
            filepath = str(row["filepath"])
            if not os.path.isabs(filepath) and self.data_path:
                filepath = os.path.join(self.data_path, filepath)
            filepath = os.path.normpath(filepath)

            if self.validate_paths and not os.path.exists(filepath):
                continue

            if self.speakers and str(row.get("speaker_id", "unknown")) not in self.speakers:
                continue

            label = self._normalize_label(row["label"])
            self.data.append(
                {
                    "filepath": filepath,
                    "label": label,
                    "speaker_id": str(row.get("speaker_id", "unknown")),
                    "system_id": str(row.get("system_id", "unknown")),
                    "split": str(row.get("split", split)),
                }
            )
            self.labels.append(label)
            self.speaker_ids.append(str(row.get("speaker_id", "unknown")))
            self.system_ids.append(str(row.get("system_id", "unknown")))

    def _load_from_protocol(self, data_path: str, metadata_path: str) -> None:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Protocol file not found: {metadata_path}")

        df = pd.read_csv(
            metadata_path,
            sep=r"\s+",
            header=None,
            names=["speaker_id", "audio_file", "unused", "system_id", "key"],
        )

        for _, row in df.iterrows():
            filename = f"{row['audio_file']}.flac"
            filepath = os.path.join(data_path, filename)
            label = 0 if str(row["key"]).lower() == "bonafide" else 1

            if self.validate_paths and not os.path.exists(filepath):
                continue

            if self.speakers and str(row["speaker_id"]) not in self.speakers:
                continue

            self.data.append(
                {
                    "filepath": filepath,
                    "label": label,
                    "speaker_id": str(row["speaker_id"]),
                    "system_id": str(row["system_id"]),
                    "split": self.split,
                }
            )
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

    @staticmethod
    def _normalize_label(value: Any) -> int:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)

        text = str(value).strip().lower()
        if text in {"bonafide", "real", "0", "false", "neg"}:
            return 0
        if text in {"spoof", "fake", "1", "true", "pos"}:
            return 1
        return int(text) if text.isdigit() else 1

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.data):
            raise IndexError("Dataset index out of range.")

        sample = self.data[idx]
        label = int(sample["label"])
        filepath = sample["filepath"]

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
            waveform = torch.zeros(1, target_samples, dtype=torch.float32)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, torch.tensor(label, dtype=torch.long)

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data):
            raise IndexError("Dataset index out of range.")
        return self.data[idx]
