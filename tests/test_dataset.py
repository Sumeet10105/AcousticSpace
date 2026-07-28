<<<<<<< HEAD
"""Tests for dataset module."""

import os

import pandas as pd
import pytest
import soundfile as sf
import torch


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a tiny manifest-backed dataset for unit tests."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    rows = []
    for idx in range(4):
        path = audio_dir / f"sample_{idx}.flac"
        sf.write(path, [0.1, -0.1, 0.2, -0.2], 16000)
        rows.append(
            {
                "filepath": str(path),
                "label": idx % 2,
                "split": "train" if idx < 3 else "val",
                "speaker_id": f"spk_{idx % 2}",
                "system_id": "A01",
            }
        )

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


class TestDataset:
    def test_dataset_length(self, synthetic_dataset):
        from src.data.dataset import AudioSpoofingDataset

        dataset = AudioSpoofingDataset(manifest_path=str(synthetic_dataset), split="train")
        assert len(dataset) == 3
        assert len(dataset.labels) == 3
        assert len(dataset.speaker_ids) == 3

    def test_dataset_getitem(self, synthetic_dataset):
        from src.data.dataset import AudioSpoofingDataset

        dataset = AudioSpoofingDataset(
            manifest_path=str(synthetic_dataset),
            split="train",
            max_samples=2,
        )
        waveform, label = dataset[0]

        assert isinstance(waveform, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert label.item() in [0, 1]
        assert waveform.ndim == 2

    def test_protocol_mode(self, tmp_path):
        from src.data.dataset import AudioSpoofingDataset

        audio_dir = tmp_path / "flac"
        audio_dir.mkdir()
        audio_path = audio_dir / "LA_T_0000001.flac"
        sf.write(audio_path, [0.1, -0.1, 0.2], 16000)

        protocol = tmp_path / "protocol.txt"
        with open(protocol, "w", encoding="utf-8") as handle:
            handle.write("LA_0001 LA_T_0000001 - - bonafide\n")

        dataset = AudioSpoofingDataset(
            data_path=str(audio_dir),
            metadata_path=str(protocol),
            split="train",
        )
        assert len(dataset) == 1
        waveform, label = dataset[0]
        assert label.item() == 0

    def test_speaker_stratified_split(self):
        from src.data.splitter import speaker_stratified_split

        records = [
            {"speaker_id": f"spk_{i}", "filepath": f"/tmp/{i}.flac"}
            for i in range(10)
        ]
        train, val, test = speaker_stratified_split(records, train_ratio=0.6, val_ratio=0.2)
        train_speakers = {row["speaker_id"] for row in train}
        val_speakers = {row["speaker_id"] for row in val}
        test_speakers = {row["speaker_id"] for row in test}
        assert train_speakers.isdisjoint(val_speakers)
        assert train_speakers.isdisjoint(test_speakers)
        assert val_speakers.isdisjoint(test_speakers)
=======
class TestDataset:
    """Dataset tests."""

    def test_dataset_loading_and_splits(self, temp_dataset):
        """Test dataset split filtering."""
        temp_dir, manifest_path, _ = temp_dataset

        dataset_train = AudioSpoofingDataset(
            data_path=temp_dir,
            metadata_path=manifest_path,
            split="train",
            validate_paths=True,
        )

        assert len(dataset_train) == 2

        dataset_val = AudioSpoofingDataset(
            data_path=temp_dir,
            metadata_path=manifest_path,
            split="val",
            validate_paths=True,
        )

        assert len(dataset_val) == 1

        dataset_test = AudioSpoofingDataset(
            data_path=temp_dir,
            metadata_path=manifest_path,
            split="test",
            validate_paths=True,
        )

        assert len(dataset_test) == 1

    def test_dataset_getitem(self, temp_dataset):
        """Test loading a dataset sample."""
        temp_dir, manifest_path, _ = temp_dataset

        dataset = AudioSpoofingDataset(
            data_path=temp_dir,
            metadata_path=manifest_path,
            split="train",
            validate_paths=True,
        )

        waveform, label = dataset[0]

        assert isinstance(waveform, torch.Tensor)
        assert waveform.shape[0] == 1
        assert waveform.shape[1] == 16000
        assert label in [0, 1]

    def test_max_samples(self, temp_dataset):
        """Test max_samples parameter."""
        temp_dir, manifest_path, _ = temp_dataset

        dataset = AudioSpoofingDataset(
            data_path=temp_dir,
            metadata_path=manifest_path,
            split="train",
            max_samples=1,
            validate_paths=True,
        )

        assert len(dataset) == 1

    def test_speaker_filtering(self, temp_dataset):
        """Test filtering by speaker IDs."""
        temp_dir, manifest_path, _ = temp_dataset

        dataset = AudioSpoofingDataset(
            data_path=temp_dir,
            metadata_path=manifest_path,
            split="train",
            speakers=["SP_001"],
            validate_paths=True,
        )

        assert len(dataset) == 2

    def test_corrupted_audio_fallback(self, temp_dataset):
        """Ensure corrupted audio files don't crash loading."""
        temp_dir, manifest_path, _ = temp_dataset

        dataset = AudioSpoofingDataset(
            data_path=temp_dir,
            metadata_path=manifest_path,
            split="train",
            validate_paths=True,
        )

        waveform, label = dataset[1]

        assert isinstance(waveform, torch.Tensor)
        assert label in [0, 1]
>>>>>>> 01d9eff5e902e9e68f29504ca915f1ccf7624734
