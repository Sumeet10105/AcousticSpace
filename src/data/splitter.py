"""Data splitting utilities with speaker-aware stratification."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def train_val_test_split(
    data: Sequence[Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Sequential split into train, validation, and test sets."""
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = list(data[:train_size])
    val_data = list(data[train_size : train_size + val_size])
    test_data = list(data[train_size + val_size :])
    return train_data, val_data, test_data


def speaker_stratified_split(
    records: List[Dict[str, Any]],
    speaker_key: str = "speaker_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records by speaker to reduce data leakage across splits."""
    rng = np.random.default_rng(seed)
    speakers = sorted({str(record[speaker_key]) for record in records})
    rng.shuffle(speakers)

    n_speakers = len(speakers)
    train_end = int(n_speakers * train_ratio)
    val_end = train_end + int(n_speakers * val_ratio)

    train_speakers = set(speakers[:train_end])
    val_speakers = set(speakers[train_end:val_end])
    test_speakers = set(speakers[val_end:])

    train_records, val_records, test_records = [], [], []
    for record in records:
        speaker = str(record[speaker_key])
        if speaker in train_speakers:
            train_records.append(record)
        elif speaker in val_speakers:
            val_records.append(record)
        else:
            test_records.append(record)

    return train_records, val_records, test_records
