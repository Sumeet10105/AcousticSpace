"""Unit tests for the core modules (config, logger, exceptions)."""

import os
import tempfile
import pytest
import logging
from src.core.config import Config, get_config
from src.core.logger import get_logger
from src.core.exceptions import (
    AcousticSpaceException,
    ConfigurationError,
    DataError,
    ModelError,
    PreprocessingError,
    APIError,
)


class TestCoreModules:
    """Tests for core settings, logger, and exception classes."""
    
    def test_config_loading_and_retrieval(self):
        """Test YAML config parser features."""
        # Create a temp yaml config
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as tmp:
            tmp.write(
                "project: AcousticSpace\n"
                "nested:\n"
                "  value: 42\n"
            )
            config_path = tmp.name
            
        try:
            config = Config(config_path)
            assert config.get("project") == "AcousticSpace"
            assert config.get("nested.value") == 42
            assert config["nested.value"] == 42
            assert config.get("nested.missing", "default") == "default"
            assert "AcousticSpace" in repr(config)
            
            # Global config test
            global_config = get_config(config_path)
            assert global_config.get("project") == "AcousticSpace"
        finally:
            if os.path.exists(config_path):
                os.remove(config_path)
                
    def test_config_missing_file(self):
        """Test Config raises FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError):
            Config("nonexistent_config_file.yaml")
            
    def test_logger_setup(self):
        """Test get_logger registers handlers correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            logger = get_logger("test_logger", log_file=log_file)
            
            assert isinstance(logger, logging.Logger)
            assert logger.name == "test_logger"
            
            # Log something
            logger.info("Test log statement")
            assert os.path.exists(log_file)
            with open(log_file, "r") as f:
                content = f.read()
            assert "Test log statement" in content
            
    def test_custom_exceptions(self):
        """Test custom exception types raise and inherit from base exception."""
        with pytest.raises(AcousticSpaceException):
            raise ConfigurationError("config invalid")
            
        with pytest.raises(AcousticSpaceException):
            raise DataError("data invalid")
            
        with pytest.raises(AcousticSpaceException):
            raise ModelError("model invalid")
            
        with pytest.raises(AcousticSpaceException):
            raise PreprocessingError("preprocessing invalid")
            
        with pytest.raises(AcousticSpaceException):
            raise APIError("api invalid")
