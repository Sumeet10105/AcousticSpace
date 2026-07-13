"""Data loader utilities."""

from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
    """
    Create a DataLoader for the given dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
