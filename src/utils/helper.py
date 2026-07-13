import numpy as np
import torch


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    if isinstance(data, torch.Tensor):
        return data
    return torch.from_numpy(data).float()


def to_numpy(data: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    if isinstance(data, np.ndarray):
        return data
    return data.detach().cpu().numpy()


def move_to_device(data: Any, device: str) -> Any:
    """Move data to specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [move_to_device(item, device) for item in data]
    return data
