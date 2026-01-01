"""
Databricks Runtime Version Detection.

This module provides robust detection of Databricks runtime versions
using multiple fallback methods.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def detect_databricks_runtime() -> Dict[str, Any]:
    """
    Detect Databricks runtime version using multiple fallback methods.

    This function attempts to detect the Databricks runtime version by:
    1. Checking DATABRICKS_RUNTIME_VERSION environment variable (primary)
    2. Parsing /databricks/environment.yml file (fallback)
    3. Checking /Workspace directory indicator (basic detection)
    4. Checking IPython config for DATABRICKS marker (notebook context)

    Returns:
        Dictionary with runtime information:
        - runtime_version (float): Runtime version (e.g., 14.3, 15.2, 16.4)
        - runtime_version_string (str): Full version string (e.g., "14.3.x-gpu-ml-scala2.12")
        - is_databricks (bool): True if Databricks detected
        - is_ml_runtime (bool): True if ML runtime detected
        - is_gpu_runtime (bool): True if GPU runtime detected
        - is_serverless (bool): True if Serverless Compute detected
        - cuda_version (Optional[str]): Expected CUDA version for this runtime
        - detection_method (str): How the runtime was detected

    Examples:
        >>> # Databricks ML Runtime 14.3
        >>> result = detect_databricks_runtime()
        >>> print(result)
        {
            "runtime_version": 14.3,
            "runtime_version_string": "14.3.x-gpu-ml-scala2.12",
            "is_databricks": True,
            "is_ml_runtime": True,
            "is_gpu_runtime": True,
            "is_serverless": False,
            "cuda_version": "12.2",
            "detection_method": "env_var"
        }

        >>> # Databricks ML Runtime 15.2
        >>> result = detect_databricks_runtime()
        >>> print(result)
        {
            "runtime_version": 15.2,
            "runtime_version_string": "15.2.x-gpu-ml-scala2.12",
            "is_databricks": True,
            "is_ml_runtime": True,
            "is_gpu_runtime": True,
            "is_serverless": False,
            "cuda_version": "12.4",
            "detection_method": "env_var"
        }

        >>> # Databricks ML Runtime 16.4
        >>> result = detect_databricks_runtime()
        >>> print(result)
        {
            "runtime_version": 16.4,
            "runtime_version_string": "16.4.x-gpu-ml-scala2.12",
            "is_databricks": True,
            "is_ml_runtime": True,
            "is_gpu_runtime": True,
            "is_serverless": False,
            "cuda_version": "12.6",
            "detection_method": "env_var"
        }

        >>> # Databricks Serverless GPU Compute
        >>> result = detect_databricks_runtime()
        >>> print(result)
        {
            "runtime_version": None,
            "runtime_version_string": "serverless-gpu-v4",
            "is_databricks": True,
            "is_ml_runtime": False,
            "is_gpu_runtime": True,
            "is_serverless": True,
            "cuda_version": "12.6",
            "detection_method": "env_var"
        }

        >>> # Non-Databricks environment
        >>> result = detect_databricks_runtime()
        >>> print(result)
        {
            "runtime_version": None,
            "runtime_version_string": None,
            "is_databricks": False,
            "is_ml_runtime": False,
            "is_gpu_runtime": False,
            "is_serverless": False,
            "cuda_version": None,
            "detection_method": "unknown"
        }
    """
    # Try method 1: Environment variable (primary)
    result = _detect_from_env_var()
    if result["is_databricks"]:
        logger.info(f"Databricks runtime detected via env_var: {result['runtime_version_string']}")
        return result

    # Try method 2: Parse /databricks/environment.yml
    result = _detect_from_environment_file()
    if result["is_databricks"]:
        logger.info(f"Databricks runtime detected via file: {result['runtime_version_string']}")
        return result

    # Try method 3: Check /Workspace indicator
    result = _detect_from_workspace_indicator()
    if result["is_databricks"]:
        logger.info("Databricks detected via workspace indicator")
        return result

    # Try method 4: Check IPython config
    result = _detect_from_ipython()
    if result["is_databricks"]:
        logger.info("Databricks detected via IPython config")
        return result

    # No Databricks detected
    logger.debug("No Databricks environment detected")
    return _create_result(is_databricks=False, detection_method="unknown")


def _detect_from_env_var() -> Dict[str, Any]:
    """
    Detect runtime from DATABRICKS_RUNTIME_VERSION environment variable.

    This is the primary detection method.
    """
    runtime_str = os.getenv("DATABRICKS_RUNTIME_VERSION")

    if not runtime_str:
        return _create_result(is_databricks=False, detection_method="env_var")

    # Parse runtime string
    parsed = _parse_runtime_string(runtime_str)

    return _create_result(
        is_databricks=True,
        runtime_version=parsed["runtime_version"],
        runtime_version_string=runtime_str,
        is_ml_runtime=parsed["is_ml_runtime"],
        is_gpu_runtime=parsed["is_gpu_runtime"],
        is_serverless=parsed["is_serverless"],
        cuda_version=parsed["cuda_version"],
        detection_method="env_var",
    )


def _detect_from_environment_file() -> Dict[str, Any]:
    """
    Detect runtime from /databricks/environment.yml file.

    This is a fallback method when env var is not available.
    """
    env_file = Path("/databricks/environment.yml")

    if not env_file.exists():
        return _create_result(is_databricks=False, detection_method="file")

    try:
        with open(env_file, "r") as f:
            env_data = yaml.safe_load(f)

        # Look for runtime version in various places
        runtime_str = None

        if isinstance(env_data, dict):
            # Check common keys
            for key in ["runtime_version", "databricks_runtime_version", "version"]:
                if key in env_data:
                    runtime_str = env_data[key]
                    break

            # Check nested structure
            if not runtime_str and "databricks" in env_data:
                databricks_section = env_data["databricks"]
                if isinstance(databricks_section, dict):
                    runtime_str = databricks_section.get("runtime_version")

        if not runtime_str:
            logger.warning("Could not find runtime version in environment.yml")
            return _create_result(
                is_databricks=True,
                detection_method="file",
            )

        # Parse runtime string
        parsed = _parse_runtime_string(str(runtime_str))

        return _create_result(
            is_databricks=True,
            runtime_version=parsed["runtime_version"],
            runtime_version_string=str(runtime_str),
            is_ml_runtime=parsed["is_ml_runtime"],
            is_gpu_runtime=parsed["is_gpu_runtime"],
            is_serverless=parsed["is_serverless"],
            cuda_version=parsed["cuda_version"],
            detection_method="file",
        )

    except Exception as e:
        logger.warning(f"Failed to parse /databricks/environment.yml: {e}")
        return _create_result(is_databricks=True, detection_method="file")


def _detect_from_workspace_indicator() -> Dict[str, Any]:
    """
    Detect Databricks by checking for /Workspace directory.

    This is a basic detection method that only confirms Databricks presence.
    """
    workspace_path = Path("/Workspace")

    if not workspace_path.exists():
        return _create_result(is_databricks=False, detection_method="workspace")

    # Databricks detected, but no version info
    return _create_result(
        is_databricks=True,
        detection_method="workspace",
    )


def _detect_from_ipython() -> Dict[str, Any]:
    """
    Detect Databricks from IPython configuration.

    This works in Databricks notebooks where IPython is available.
    """
    try:
        import IPython

        ipython = IPython.get_ipython()
        if ipython and hasattr(ipython, "config"):
            # Check for DATABRICKS in config
            config_str = str(ipython.config)
            if "DATABRICKS" in config_str:
                return _create_result(
                    is_databricks=True,
                    detection_method="ipython",
                )

    except (ImportError, AttributeError) as e:
        logger.debug(f"IPython detection failed: {e}")

    return _create_result(is_databricks=False, detection_method="ipython")


def _parse_runtime_string(runtime_str: str) -> Dict[str, Any]:
    """
    Parse Databricks runtime version string.

    Args:
        runtime_str: Runtime version string (e.g., "14.3.x-gpu-ml-scala2.12")

    Returns:
        Dictionary with parsed components
    """
    result = {
        "runtime_version": None,
        "is_ml_runtime": False,
        "is_gpu_runtime": False,
        "is_serverless": False,
        "cuda_version": None,
    }

    if not runtime_str:
        return result

    runtime_lower = runtime_str.lower()

    # Check for serverless
    if "serverless" in runtime_lower:
        result["is_serverless"] = True
        result["is_gpu_runtime"] = "gpu" in runtime_lower

        # Serverless versioning (e.g., "serverless-gpu-v4")
        if "v4" in runtime_lower or "12.6" in runtime_lower:
            result["cuda_version"] = "12.6"
        elif "v3" in runtime_lower or "12.4" in runtime_lower:
            result["cuda_version"] = "12.4"

        return result

    # Check for ML runtime
    result["is_ml_runtime"] = "-ml" in runtime_lower

    # Check for GPU runtime
    result["is_gpu_runtime"] = "-gpu" in runtime_lower

    # Extract version number (e.g., "14.3" from "14.3.x-gpu-ml-scala2.12")
    version_match = re.match(r"^(\d+)\.(\d+)", runtime_str)
    if version_match:
        major = int(version_match.group(1))
        minor = int(version_match.group(2))
        result["runtime_version"] = float(f"{major}.{minor}")

        # Map runtime version to expected CUDA version
        result["cuda_version"] = _get_cuda_version_for_runtime(result["runtime_version"])

    return result


def _get_cuda_version_for_runtime(runtime_version: Optional[float]) -> Optional[str]:
    """
    Get expected CUDA version for a given Databricks runtime version.

    This mapping is based on Databricks ML Runtime release notes.

    Args:
        runtime_version: Runtime version (e.g., 14.3, 15.2, 16.4)

    Returns:
        Expected CUDA version string (e.g., "12.2", "12.4", "12.6")
    """
    if runtime_version is None:
        return None

    # Mapping based on Databricks ML Runtime documentation
    # https://docs.databricks.com/release-notes/runtime/index.html
    cuda_mapping = {
        16.4: "12.6",  # ML Runtime 16.4 (Dec 2025)
        16.0: "12.6",  # ML Runtime 16.0
        15.4: "12.4",  # ML Runtime 15.4
        15.3: "12.4",  # ML Runtime 15.3
        15.2: "12.4",  # ML Runtime 15.2
        15.1: "12.4",  # ML Runtime 15.1
        15.0: "12.4",  # ML Runtime 15.0
        14.3: "12.2",  # ML Runtime 14.3
        14.2: "12.2",  # ML Runtime 14.2
        14.1: "12.2",  # ML Runtime 14.1
        14.0: "12.2",  # ML Runtime 14.0
        13.3: "11.8",  # ML Runtime 13.3
        13.2: "11.8",  # ML Runtime 13.2
        13.1: "11.8",  # ML Runtime 13.1
        13.0: "11.8",  # ML Runtime 13.0
    }

    # Find the closest runtime version (in case of patch versions)
    closest_version = min(
        cuda_mapping.keys(),
        key=lambda v: abs(v - runtime_version),
        default=None,
    )

    if closest_version and abs(closest_version - runtime_version) < 0.5:
        return cuda_mapping[closest_version]

    logger.warning(f"Unknown CUDA version for runtime {runtime_version}")
    return None


def _create_result(
    is_databricks: bool,
    runtime_version: Optional[float] = None,
    runtime_version_string: Optional[str] = None,
    is_ml_runtime: bool = False,
    is_gpu_runtime: bool = False,
    is_serverless: bool = False,
    cuda_version: Optional[str] = None,
    detection_method: str = "unknown",
) -> Dict[str, Any]:
    """
    Create a standardized result dictionary.

    Args:
        is_databricks: Whether Databricks was detected
        runtime_version: Parsed runtime version (e.g., 14.3)
        runtime_version_string: Full runtime string
        is_ml_runtime: Whether this is an ML runtime
        is_gpu_runtime: Whether this is a GPU runtime
        is_serverless: Whether this is Serverless Compute
        cuda_version: Expected CUDA version
        detection_method: How the runtime was detected

    Returns:
        Standardized result dictionary
    """
    return {
        "runtime_version": runtime_version,
        "runtime_version_string": runtime_version_string,
        "is_databricks": is_databricks,
        "is_ml_runtime": is_ml_runtime,
        "is_gpu_runtime": is_gpu_runtime,
        "is_serverless": is_serverless,
        "cuda_version": cuda_version,
        "detection_method": detection_method,
    }


def get_runtime_info_summary() -> str:
    """
    Get a human-readable summary of the detected runtime.

    Returns:
        Formatted string summarizing the runtime environment

    Example:
        >>> print(get_runtime_info_summary())
        Databricks ML Runtime 16.4 (GPU, CUDA 12.6)
        Detected via: env_var
    """
    result = detect_databricks_runtime()

    if not result["is_databricks"]:
        return "Not running in Databricks environment"

    parts = []

    if result["is_serverless"]:
        parts.append("Databricks Serverless GPU Compute")
        if result["runtime_version_string"]:
            parts.append(f"({result['runtime_version_string']})")
    else:
        parts.append("Databricks")
        if result["is_ml_runtime"]:
            parts.append("ML Runtime")
        else:
            parts.append("Runtime")

        if result["runtime_version"]:
            parts.append(str(result["runtime_version"]))

        flags = []
        if result["is_gpu_runtime"]:
            flags.append("GPU")
        if result["cuda_version"]:
            flags.append(f"CUDA {result['cuda_version']}")

        if flags:
            parts.append(f"({', '.join(flags)})")

    summary = " ".join(parts)
    summary += f"\nDetected via: {result['detection_method']}"

    return summary


# Convenience function for backward compatibility
def is_databricks_environment() -> bool:
    """
    Check if code is running in a Databricks environment.

    Returns:
        True if running in Databricks, False otherwise

    Example:
        >>> if is_databricks_environment():
        ...     print("Running in Databricks!")
    """
    result = detect_databricks_runtime()
    return result["is_databricks"]
