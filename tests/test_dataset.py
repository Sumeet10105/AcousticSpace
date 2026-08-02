"""Tests for dataset module."""

import pandas as pd
import pytest
import soundfile as sf
import torch

from src.data.dataset import AudioSpoofingDataset
from src.data.splitter import speaker_stratified_split


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


@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset fixture for split and loading tests."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    rows = [
        {
            "filepath": str(audio_dir / "sample_0.flac"),
            "label": "bonafide",
            "split": "train",
            "speaker_id": "SP_001",
            "attack_type": "A01",
        },
        {
            "filepath": str(audio_dir / "sample_1.flac"),
            "label": "spoof",
            "split": "train",
            "speaker_id": "SP_001",
            "attack_type": "A02",
        },
        {
            "filepath": str(audio_dir / "sample_2.flac"),
            "label": "bonafide",
            "split": "val",
            "speaker_id": "SP_002",
            "attack_type": "A03",
        },
        {
            "filepath": str(audio_dir / "sample_3.flac"),
            "label": "spoof",
            "split": "test",
            "speaker_id": "SP_003",
            "attack_type": "A04",
        },
    ]

    for idx, row in enumerate(rows):
        sf.write(row["filepath"], [0.1, -0.1, 0.2, -0.2], 16000)

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return str(audio_dir), str(manifest), rows


class TestDataset:
    def test_dataset_length(self, synthetic_dataset):
        dataset = AudioSpoofingDataset(manifest_path=str(synthetic_dataset), split="train")
        assert len(dataset) == 3
        assert len(dataset.labels) == 3
        assert len(dataset.speaker_ids) == 3

    def test_dataset_getitem(self, synthetic_dataset):
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

    def test_dataset_loading_and_splits(self, temp_dataset):
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

    def test_dataset_getitem_from_metadata(self, temp_dataset):
        temp_dir, manifest_path, _ = temp_dataset

        dataset = AudioSpoofingDataset(
            data_path=temp_dir,
            metadata_path=manifest_path,
            split="train",
            validate_paths=True,
            duration_seconds=1.0,
        )

        waveform, label = dataset[0]

        assert isinstance(waveform, torch.Tensor)
        assert waveform.shape[0] == 1
        assert waveform.shape[1] == 16000
        assert label.item() in [0, 1]

    def test_max_samples(self, temp_dataset):
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
        temp_dir, manifest_path, _ = temp_dataset

        dataset = AudioSpoofingDataset(
            data_path=temp_dir,
            metadata_path=manifest_path,
            split="train",
            validate_paths=True,
        )

        waveform, label = dataset[1]

        assert isinstance(waveform, torch.Tensor)
        assert label.item() in [0, 1]

    def test_audio_validator_functions(self):
        """Test audio validators in src/data/validator.py."""
        from src.data.validator import validate_audio_file, validate_audio_data, validate_uploaded_file
        import numpy as np
        
        assert not validate_audio_file(None)
        assert not validate_audio_file("")
        assert not validate_audio_file("nonexistent_file.wav")
        
        assert not validate_audio_data(None)
        assert not validate_audio_data("invalid_type")
        assert not validate_audio_data(torch.randn(5))
        assert not validate_audio_data(torch.tensor([[float("nan")]]))
        
        assert not validate_audio_data(np.array([]))
        assert not validate_audio_data(np.array([[float("inf")]]))
        assert validate_audio_data(np.array([[0.1, 0.2]]))
        
        ok, msg = validate_uploaded_file("", b"")
        assert not ok
        assert msg == "Missing filename"
        
        ok, msg = validate_uploaded_file("test.wav", b"")
        assert not ok
        assert msg == "Empty file"
        
        ok, msg = validate_uploaded_file("test.wav", b"a" * (51 * 1024 * 1024), max_size_mb=50.0)
        assert not ok
        assert "exceeds" in msg
        
        ok, msg = validate_uploaded_file("test.txt", b"abc")
        assert not ok
        assert "Unsupported" in msg
        
        ok, msg = validate_uploaded_file("test.wav", b"abc")
        assert ok

    def test_data_splitter_edge_cases(self):
        """Test splitter.py fallback cases and list splitting."""
        from src.data.splitter import train_val_test_split, speaker_aware_split
        import pandas as pd
        
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        train_l, val_l, test_l = train_val_test_split(lst, 0.8, 0.1)
        assert len(train_l) == 8
        assert len(val_l) == 1
        assert len(test_l) == 1
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        train_d, val_d, test_d = speaker_aware_split(df, speaker_col="nonexistent")
        assert len(train_d) == 2
