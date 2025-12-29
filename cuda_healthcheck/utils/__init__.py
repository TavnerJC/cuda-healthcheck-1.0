"""Utility modules for CUDA Healthcheck."""

from .error_recovery import (
    ErrorRecoveryContext,
    GracefulDegradation,
    PartialResultCollector,
    safe_detection,
)
from .logging_config import get_logger, setup_logging
from .performance import (
    BatchProcessor,
    LRUCache,
    PerformanceTimer,
    cached,
    memoize,
    timed,
)
from .retry import retry_on_failure
from .validation import (
    safe_float_conversion,
    safe_int_conversion,
    safe_str_conversion,
    sanitize_cluster_name,
    validate_cluster_id,
    validate_cuda_version,
    validate_environment_variables,
    validate_table_path,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "retry_on_failure",
    "validate_cuda_version",
    "validate_cluster_id",
    "validate_table_path",
    "validate_environment_variables",
    "sanitize_cluster_name",
    "safe_int_conversion",
    "safe_float_conversion",
    "safe_str_conversion",
    "GracefulDegradation",
    "PartialResultCollector",
    "safe_detection",
    "ErrorRecoveryContext",
    "LRUCache",
    "cached",
    "memoize",
    "PerformanceTimer",
    "timed",
    "BatchProcessor",
]
