"""Data splitting utilities."""


def train_val_test_split(data, train_ratio=0.8, val_ratio=0.1):
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Data to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data
