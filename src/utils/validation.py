"""
Validation utilities for CUDA Healthcheck Tool.

Provides input validation, data sanitization, and validation helpers
to ensure data integrity and prevent errors.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .exceptions import ConfigurationError, CudaHealthcheckError
from .logging_config import get_logger

logger = get_logger(__name__)


def validate_cuda_version(version: str) -> bool:
    """
    Validate CUDA version string format.

    Args:
        version: CUDA version string (e.g., "12.4", "13.0")

    Returns:
        True if valid, False otherwise

    Example:
        ```python
        assert validate_cuda_version("12.4") == True
        assert validate_cuda_version("invalid") == False
        ```
    """
    if not version or not isinstance(version, str):
        return False

    # Match pattern: X.Y or X.Y.Z
    pattern = r"^\d+\.\d+(\.\d+)?$"
    return bool(re.match(pattern, version.strip()))


def validate_cluster_id(cluster_id: str) -> bool:
    """
    Validate Databricks cluster ID format.

    Args:
        cluster_id: Cluster ID string

    Returns:
        True if valid format, False otherwise
    """
    if not cluster_id or not isinstance(cluster_id, str):
        return False

    # Cluster IDs are typically alphanumeric with hyphens
    pattern = r"^[a-zA-Z0-9\-_]+$"
    return bool(re.match(pattern, cluster_id.strip()))


def validate_table_path(table_path: str) -> bool:
    """
    Validate Delta table path format.

    Args:
        table_path: Table path (e.g., "catalog.schema.table")

    Returns:
        True if valid format, False otherwise
    """
    if not table_path or not isinstance(table_path, str):
        return False

    # Pattern: catalog.schema.table or schema.table
    parts = table_path.strip().split(".")
    if len(parts) < 2 or len(parts) > 3:
        return False

    # Each part should be valid identifier
    identifier_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    return all(re.match(identifier_pattern, part) for part in parts)


def validate_databricks_host(host: str) -> bool:
    """
    Validate Databricks host URL format.

    Args:
        host: Databricks host URL

    Returns:
        True if valid format, False otherwise
    """
    if not host or not isinstance(host, str):
        return False

    # Should start with https:// and contain databricks
    host = host.strip()
    if not host.startswith("https://"):
        return False

    # Should look like a valid URL
    url_pattern = r"^https://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}/?"
    return bool(re.match(url_pattern, host))


def validate_token(token: str) -> bool:
    """
    Validate Databricks token format.

    Args:
        token: Databricks personal access token

    Returns:
        True if valid format, False otherwise
    """
    if not token or not isinstance(token, str):
        return False

    token = token.strip()

    # Databricks tokens typically start with 'dapi'
    if token.startswith("dapi"):
        # Should be alphanumeric
        return len(token) > 10 and token.replace("dapi", "").isalnum()

    # Also accept other token formats (for service principals, etc.)
    return len(token) > 10


def validate_file_path(path: str, must_exist: bool = False) -> bool:
    """
    Validate file path.

    Args:
        path: File path to validate
        must_exist: If True, check if file exists

    Returns:
        True if valid, False otherwise
    """
    if not path or not isinstance(path, str):
        return False

    try:
        path_obj = Path(path)

        if must_exist:
            return path_obj.exists() and path_obj.is_file()

        # Just check if it's a valid path format
        return True

    except (ValueError, OSError):
        return False


def validate_environment_variables(required_vars: List[str]) -> Dict[str, str]:
    """
    Validate that required environment variables are set.

    Args:
        required_vars: List of required environment variable names

    Returns:
        Dictionary of variable names to values

    Raises:
        ConfigurationError: If required variables are missing

    Example:
        ```python
        try:
            env_vars = validate_environment_variables(['DATABRICKS_HOST', 'DATABRICKS_TOKEN'])
            host = env_vars['DATABRICKS_HOST']
        except ConfigurationError as e:
            print(f"Missing config: {e}")
        ```
    """
    missing = []
    env_vars = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        else:
            env_vars[var] = value

    if missing:
        raise ConfigurationError(
            f"Required environment variables not set: {', '.join(missing)}\n"
            f"Please set these variables or see docs/ENVIRONMENT_VARIABLES.md"
        )

    return env_vars


def sanitize_cluster_name(name: str) -> str:
    """
    Sanitize cluster name for safe use in file names, etc.

    Args:
        name: Cluster name to sanitize

    Returns:
        Sanitized name
    """
    if not name:
        return "unnamed_cluster"

    # Replace spaces and special chars with underscores
    sanitized = re.sub(r"[^\w\-]", "_", name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    return sanitized or "unnamed_cluster"


def validate_compatibility_score(score: Union[int, float]) -> bool:
    """
    Validate compatibility score is in valid range.

    Args:
        score: Compatibility score

    Returns:
        True if valid (0-100), False otherwise
    """
    try:
        score_num = float(score)
        return 0 <= score_num <= 100
    except (ValueError, TypeError):
        return False


def validate_library_info(lib_info: Dict[str, Any]) -> bool:
    """
    Validate library information dictionary structure.

    Args:
        lib_info: Library information dictionary

    Returns:
        True if valid structure, False otherwise
    """
    required_keys = ["name", "version", "is_compatible"]

    if not isinstance(lib_info, dict):
        return False

    # Check required keys exist
    if not all(key in lib_info for key in required_keys):
        return False

    # Validate types
    if not isinstance(lib_info.get("name"), str):
        return False
    if not isinstance(lib_info.get("version"), str):
        return False
    if not isinstance(lib_info.get("is_compatible"), bool):
        return False

    # Warnings should be a list if present
    if "warnings" in lib_info and not isinstance(lib_info["warnings"], list):
        return False

    return True


def validate_gpu_info(gpu_info: Dict[str, Any]) -> bool:
    """
    Validate GPU information dictionary structure.

    Args:
        gpu_info: GPU information dictionary

    Returns:
        True if valid structure, False otherwise
    """
    required_keys = ["name", "compute_capability", "memory_total_mb"]

    if not isinstance(gpu_info, dict):
        return False

    # Check required keys
    if not all(key in gpu_info for key in required_keys):
        return False

    # Validate compute capability format
    if "compute_capability" in gpu_info:
        cc = str(gpu_info.get("compute_capability", ""))
        if not re.match(r"^\d+\.\d+$", cc):
            return False

    # Validate memory is positive number
    if "memory_total_mb" in gpu_info:
        try:
            memory = int(gpu_info.get("memory_total_mb", 0))
            if memory <= 0:
                return False
        except (ValueError, TypeError):
            return False

    return True


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Integer value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.debug(f"Could not convert {value} to int, using default {default}")
        return default


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.debug(f"Could not convert {value} to float, using default {default}")
        return default


def safe_str_conversion(value: Any, default: str = "") -> str:
    """
    Safely convert value to string.

    Args:
        value: Value to convert
        default: Default value if None

    Returns:
        String value or default
    """
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        logger.debug(f"Could not convert {value} to string, using default")
        return default


def validate_and_sanitize_input(
    value: Any,
    expected_type: type,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allowed_values: Optional[List[Any]] = None,
) -> Any:
    """
    Validate and sanitize input value.

    Args:
        value: Value to validate
        expected_type: Expected type
        min_length: Minimum length for strings
        max_length: Maximum length for strings
        allowed_values: List of allowed values

    Returns:
        Validated and sanitized value

    Raises:
        CudaHealthcheckError: If validation fails
    """
    # Type check
    if not isinstance(value, expected_type):
        raise CudaHealthcheckError(
            f"Invalid type: expected {expected_type.__name__}, got {type(value).__name__}"
        )

    # String-specific validations
    if expected_type == str and isinstance(value, str):
        value = value.strip()

        if min_length and len(value) < min_length:
            raise CudaHealthcheckError(
                f"Value too short: minimum length {min_length}, got {len(value)}"
            )

        if max_length and len(value) > max_length:
            raise CudaHealthcheckError(
                f"Value too long: maximum length {max_length}, got {len(value)}"
            )

    # Allowed values check
    if allowed_values and value not in allowed_values:
        raise CudaHealthcheckError(f"Invalid value: must be one of {allowed_values}, got {value}")

    return value


def check_command_available(command: str) -> bool:
    """
    Check if a command is available in PATH.

    Args:
        command: Command name to check

    Returns:
        True if command is available, False otherwise
    """
    import shutil

    return shutil.which(command) is not None


def validate_json_serializable(obj: Any) -> bool:
    """
    Check if object is JSON serializable.

    Args:
        obj: Object to check

    Returns:
        True if serializable, False otherwise
    """
    import json

    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


# Export all validation functions
__all__ = [
    "validate_cuda_version",
    "validate_cluster_id",
    "validate_table_path",
    "validate_databricks_host",
    "validate_token",
    "validate_file_path",
    "validate_environment_variables",
    "sanitize_cluster_name",
    "validate_compatibility_score",
    "validate_library_info",
    "validate_gpu_info",
    "safe_int_conversion",
    "safe_float_conversion",
    "safe_str_conversion",
    "validate_and_sanitize_input",
    "check_command_available",
    "validate_json_serializable",
]
