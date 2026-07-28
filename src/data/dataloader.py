"""Data loader utilities for deep learning pipelines."""

import logging
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def create_dataloader(
    dataset: Dataset, 
    batch_size: int = 32, 
    shuffle: bool = True, 
    num_workers: int = 4, 
    pin_memory: bool = True
) -> DataLoader:
    """Create a PyTorch DataLoader for the given dataset.
    
    Args:
        dataset: Dataset instance containing the audio samples.
        batch_size: Batch size for training or inference.
        shuffle: Whether to shuffle the data at every epoch.
        num_workers: Number of subprocesses to use for data loading.
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning them.
        
    Returns:
        A PyTorch DataLoader instance.
    """
    logger.info(
        f"Creating dataloader with batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}."
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
