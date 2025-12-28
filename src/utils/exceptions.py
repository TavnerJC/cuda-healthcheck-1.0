"""
Custom exceptions for CUDA Healthcheck Tool.

Defines specific exception types for different failure scenarios
to enable better error handling and debugging.
"""


class CudaHealthcheckError(Exception):
    """Base exception for all CUDA Healthcheck errors."""

    pass


class CudaDetectionError(CudaHealthcheckError):
    """Raised when CUDA detection fails."""

    pass


class DatabricksConnectionError(CudaHealthcheckError):
    """Raised when cannot connect to Databricks."""

    pass


class ClusterNotRunningError(CudaHealthcheckError):
    """Raised when cluster is not in RUNNING state."""

    pass


class ClusterNotFoundError(CudaHealthcheckError):
    """Raised when specified cluster cannot be found."""

    pass


class DeltaTableError(CudaHealthcheckError):
    """Raised when Delta table operations fail."""

    pass


class CompatibilityError(CudaHealthcheckError):
    """Raised when critical compatibility issues are detected."""

    pass


class BreakingChangeError(CudaHealthcheckError):
    """Raised when breaking changes prevent operation."""

    pass


class ConfigurationError(CudaHealthcheckError):
    """Raised when configuration is invalid or missing."""

    pass
