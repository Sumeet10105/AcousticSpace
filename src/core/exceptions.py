"""Custom exceptions for AcousticSpace."""


class AcousticSpaceException(Exception):
    """Base exception for AcousticSpace."""
    pass


class ConfigurationError(AcousticSpaceException):
    """Raised when configuration is invalid."""
    pass


class DataError(AcousticSpaceException):
    """Raised when data processing fails."""
    pass


class ModelError(AcousticSpaceException):
    """Raised when model operation fails."""
    pass


class PreprocessingError(AcousticSpaceException):
    """Raised when preprocessing fails."""
    pass


class APIError(AcousticSpaceException):
    """Raised when API operation fails."""
    pass
