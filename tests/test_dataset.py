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