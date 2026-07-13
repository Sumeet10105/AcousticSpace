import pickle
from pathlib import Path
from typing import Any


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Path to load file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: dict, filepath: str) -> None:
    """Save data to JSON file."""
    import json
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> dict:
    """Load data from JSON file."""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)
