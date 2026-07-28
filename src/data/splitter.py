"""Data splitting utilities, including speaker-aware splitting to prevent speaker leakage."""

import logging
import random
from typing import Dict, List, Tuple, Union
import pandas as pd

logger = logging.getLogger(__name__)


def train_val_test_split(
    data: Union[pd.DataFrame, List], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1
) -> Tuple[Union[pd.DataFrame, List], Union[pd.DataFrame, List], Union[pd.DataFrame, List]]:
    """Split data into train, validation, and test sets.

    Args:
        data: Data to split (either pandas DataFrame or a list).
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.

    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    if isinstance(data, pd.DataFrame):
        train_data = data.iloc[:train_size].reset_index(drop=True)
        val_data = data.iloc[train_size:train_size + val_size].reset_index(drop=True)
        test_data = data.iloc[train_size + val_size:].reset_index(drop=True)
    else:
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
    return train_data, val_data, test_data


def speaker_aware_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    speaker_col: str = "speaker_id",
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into train, validation, and test sets based on speakers.

    Ensures that all samples from a given speaker are strictly in one split
    to prevent speaker leakage.

    Args:
        df: Input DataFrame containing audio metadata.
        train_ratio: Ratio for the training set (default: 0.8).
        val_ratio: Ratio for the validation set (default: 0.1).
        speaker_col: Column name identifying speaker IDs.
        random_seed: Seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if speaker_col not in df.columns:
        logger.warning(f"Speaker column '{speaker_col}' not found. Falling back to simple split.")
        return train_val_test_split(df, train_ratio, val_ratio)

    # Get unique speakers and shuffle them
    unique_speakers = sorted(df[speaker_col].unique())
    rng = random.Random(random_seed)
    rng.shuffle(unique_speakers)

    num_speakers = len(unique_speakers)
    train_count = int(num_speakers * train_ratio)
    val_count = int(num_speakers * val_ratio)

    train_speakers = set(unique_speakers[:train_count])
    val_speakers = set(unique_speakers[train_count:train_count + val_count])
    test_speakers = set(unique_speakers[train_count + val_count:])

    # Split df based on speaker membership
    train_df = df[df[speaker_col].isin(train_speakers)].reset_index(drop=True)
    val_df = df[df[speaker_col].isin(val_speakers)].reset_index(drop=True)
    test_df = df[df[speaker_col].isin(test_speakers)].reset_index(drop=True)

    logger.info(
        f"Speaker-aware split: {len(train_speakers)} train speakers ({len(train_df)} samples), "
        f"{len(val_speakers)} val speakers ({len(val_df)} samples), "
        f"{len(test_speakers)} test speakers ({len(test_df)} samples)."
    )

    return train_df, val_df, test_df
