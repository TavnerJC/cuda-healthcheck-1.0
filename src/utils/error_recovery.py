"""
Error Recovery Utilities.

Provides fallback mechanisms and graceful degradation strategies
for handling failures in CUDA detection and Databricks operations.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class DetectionResult:
    """Result of a detection operation with fallback support."""

    value: Any
    success: bool
    method_used: str
    fallback_used: bool = False
    error_message: Optional[str] = None


class GracefulDegradation:
    """
    Provides graceful degradation for detection operations.

    Tries multiple methods and falls back to safer defaults when needed.
    """

    def __init__(self) -> None:
        self.failures: List[str] = []

    def try_with_fallbacks(
        self,
        primary_func: Callable[[], T],
        fallback_funcs: List[Callable[[], T]],
        default_value: T,
        operation_name: str = "operation",
    ) -> DetectionResult:
        """
        Try primary function, then fallbacks, finally use default.

        Args:
            primary_func: Primary detection function
            fallback_funcs: List of fallback functions to try
            default_value: Default value if all fail
            operation_name: Name of operation for logging

        Returns:
            DetectionResult with value and metadata

        Example:
            ```python
            degradation = GracefulDegradation()
            result = degradation.try_with_fallbacks(
                primary_func=detect_via_nvidia_smi,
                fallback_funcs=[detect_via_nvcc, detect_via_env],
                default_value="Unknown",
                operation_name="CUDA version detection"
            )
            ```
        """
        # Try primary function
        try:
            logger.debug(f"Attempting {operation_name} via primary method...")
            value = primary_func()
            if value is not None:
                logger.info(f"{operation_name} succeeded with primary method")
                return DetectionResult(
                    value=value,
                    success=True,
                    method_used="primary",
                    fallback_used=False,
                )
        except Exception as e:
            error_msg = f"Primary method failed: {e}"
            logger.warning(error_msg)
            self.failures.append(error_msg)

        # Try fallbacks
        for i, fallback_func in enumerate(fallback_funcs, 1):
            try:
                logger.debug(f"Attempting {operation_name} via fallback {i}...")
                value = fallback_func()
                if value is not None:
                    logger.info(f"{operation_name} succeeded with fallback method {i}")
                    return DetectionResult(
                        value=value,
                        success=True,
                        method_used=f"fallback_{i}",
                        fallback_used=True,
                    )
            except Exception as e:
                error_msg = f"Fallback {i} failed: {e}"
                logger.debug(error_msg)
                self.failures.append(error_msg)

        # All failed, use default
        logger.warning(f"{operation_name} failed all methods, using default: {default_value}")
        return DetectionResult(
            value=default_value,
            success=False,
            method_used="default",
            fallback_used=True,
            error_message="; ".join(self.failures[-3:]),  # Last 3 errors
        )


class PartialResultCollector:
    """
    Collects partial results from multiple detections.

    Allows continuing operation even if some detections fail.
    """

    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
        self.warnings: List[str] = []

    def add_result(self, key: str, value: Any, required: bool = False) -> None:
        """
        Add a detection result.

        Args:
            key: Result key
            value: Result value
            required: If True, None value is treated as error
        """
        if value is None and required:
            error_msg = f"Required result '{key}' is None"
            logger.error(error_msg)
            self.errors[key] = error_msg
        else:
            self.results[key] = value
            logger.debug(f"Added result for '{key}': {value}")

    def add_error(self, key: str, error: Exception) -> None:
        """Add an error for a specific detection."""
        error_msg = str(error)
        self.errors[key] = error_msg
        logger.error(f"Error in '{key}': {error_msg}")

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(warning)

    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return len(self.errors) > 0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collection."""
        return {
            "results": self.results,
            "errors": self.errors,
            "warnings": self.warnings,
            "total_results": len(self.results),
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "has_errors": self.has_critical_errors(),
        }


def safe_detection(func: Callable[[], T], default: T, error_msg: str = "Detection failed") -> T:
    """
    Safely run a detection function with fallback to default.

    Args:
        func: Detection function to run
        default: Default value if detection fails
        error_msg: Error message to log

    Returns:
        Detection result or default value

    Example:
        ```python
        cuda_version = safe_detection(
            func=detector.detect_cuda_version,
            default="Unknown",
            error_msg="CUDA version detection failed"
        )
        ```
    """
    try:
        result = func()
        return result if result is not None else default
    except Exception as e:
        logger.warning(f"{error_msg}: {e}")
        return default


def create_minimal_result(operation: str, error: Exception) -> Dict[str, Any]:
    """
    Create a minimal result dictionary for failed operations.

    Args:
        operation: Name of the operation that failed
        error: The exception that occurred

    Returns:
        Dictionary with error information
    """
    return {
        "status": "error",
        "operation": operation,
        "error": str(error),
        "error_type": type(error).__name__,
        "partial_results": {},
    }


def merge_partial_results(*result_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple partial result dictionaries.

    Args:
        *result_dicts: Variable number of result dictionaries

    Returns:
        Merged dictionary with all non-None values

    Example:
        ```python
        merged = merge_partial_results(
            {"cuda_version": "12.4", "gpus": []},
            {"libraries": ["pytorch"], "cuda_version": None}
        )
        # Result: {"cuda_version": "12.4", "gpus": [], "libraries": ["pytorch"]}
        ```
    """
    merged = {}
    for result_dict in result_dicts:
        for key, value in result_dict.items():
            # Only add non-None values or if key doesn't exist
            if value is not None or key not in merged:
                merged[key] = value
    return merged


class ErrorRecoveryContext:
    """
    Context manager for operations with automatic error recovery.

    Example:
        ```python
        with ErrorRecoveryContext("CUDA detection") as recovery:
            # Risky operation
            result = detect_cuda()
            recovery.set_result(result)

        if recovery.failed:
            # Handle failure
            fallback_result = recovery.get_fallback()
        ```
    """

    def __init__(self, operation_name: str, fallback_value: Any = None):
        self.operation_name = operation_name
        self.fallback_value = fallback_value
        self.result = None
        self.error: Optional[Exception] = None
        self.failed = False

    def __enter__(self) -> "ErrorRecoveryContext":
        logger.debug(f"Starting {self.operation_name}")
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Any
    ) -> bool:
        if exc_type is not None:
            self.failed = True
            self.error = exc_val  # type: ignore[assignment]
            logger.error(f"{self.operation_name} failed: {exc_val}", exc_info=True)
            # Suppress exception if fallback is available
            return self.fallback_value is not None
        return False

    def set_result(self, result: Any) -> None:
        """Set the result of the operation."""
        self.result = result

    def get_result(self) -> Any:
        """Get result or fallback."""
        return self.result if not self.failed else self.fallback_value

    def get_fallback(self) -> Any:
        """Get the fallback value."""
        return self.fallback_value


def validate_or_fallback(
    value: Any,
    validator: Callable[[Any], bool],
    fallback: Any,
    value_name: str = "value",
) -> Any:
    """
    Validate a value and return fallback if validation fails.

    Args:
        value: Value to validate
        validator: Validation function
        fallback: Fallback value
        value_name: Name of value for logging

    Returns:
        Original value if valid, fallback otherwise

    Example:
        ```python
        from src.utils.validation import validate_cuda_version

        version = validate_or_fallback(
            value=detected_version,
            validator=validate_cuda_version,
            fallback="Unknown",
            value_name="CUDA version"
        )
        ```
    """
    try:
        if validator(value):
            return value
        else:
            logger.warning(f"{value_name} validation failed, using fallback: {fallback}")
            return fallback
    except Exception as e:
        logger.warning(f"Error validating {value_name}: {e}, using fallback: {fallback}")
        return fallback


__all__ = [
    "DetectionResult",
    "GracefulDegradation",
    "PartialResultCollector",
    "safe_detection",
    "create_minimal_result",
    "merge_partial_results",
    "ErrorRecoveryContext",
    "validate_or_fallback",
]
