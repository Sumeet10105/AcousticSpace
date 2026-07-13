import os
from pathlib import Path
from typing import Optional

import yaml


class Config:
    """Configuration manager for AcousticSpace."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = Path(config_path) if config_path else Path("configs/config.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config or {}
    
    def get(self, key: str, default=None):
        """Get configuration value by dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str):
        """Get configuration value using dictionary notation."""
        return self.get(key)
    
    def __repr__(self) -> str:
        """String representation of config."""
        return str(self.config)


# Global config instance
_config = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
