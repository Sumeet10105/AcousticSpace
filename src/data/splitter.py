"""Data splitting utilities with speaker-aware stratification."""

import logging
import random
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_val_test_split(
    data: Union[pd.DataFrame, Sequence[Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[Union[pd.DataFrame, List[Any]], Union[pd.DataFrame, List[Any]], Union[pd.DataFrame, List[Any]]]:
    """Sequential split into train, validation, and test sets."""
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    if isinstance(data, pd.DataFrame):
        train_data = data.iloc[:train_size].reset_index(drop=True)
        val_data = data.iloc[train_size : train_size + val_size].reset_index(drop=True)
        test_data = data.iloc[train_size + val_size :].reset_index(drop=True)
    else:
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


def speaker_aware_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    speaker_col: str = "speaker_id",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train, validation, and test sets by speaker."""
    if speaker_col not in df.columns:
        logger.warning("Speaker column '%s' not found. Falling back to simple split.", speaker_col)
        return train_val_test_split(df, train_ratio, val_ratio)

    unique_speakers = sorted(df[speaker_col].astype(str).unique())
    rng = random.Random(random_seed)
    rng.shuffle(unique_speakers)

    num_speakers = len(unique_speakers)
    train_count = int(num_speakers * train_ratio)
    val_count = int(num_speakers * val_ratio)

    train_speakers = set(unique_speakers[:train_count])
    val_speakers = set(unique_speakers[train_count : train_count + val_count])
    test_speakers = set(unique_speakers[train_count + val_count :])

    train_df = df[df[speaker_col].astype(str).isin(train_speakers)].reset_index(drop=True)
    val_df = df[df[speaker_col].astype(str).isin(val_speakers)].reset_index(drop=True)
    test_df = df[df[speaker_col].astype(str).isin(test_speakers)].reset_index(drop=True)

    logger.info(
        "Speaker-aware split: %s train speakers (%s samples), %s val speakers (%s samples), %s test speakers (%s samples).",
        len(train_speakers),
        len(train_df),
        len(val_speakers),
        len(val_df),
        len(test_speakers),
        len(test_df),
    )

    return train_df, val_df, test_df
