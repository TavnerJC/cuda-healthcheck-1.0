"""
CUDA Detection Module for Databricks Clusters.

Provides classes and functions for detecting CUDA versions, GPU properties,
and library compatibility.

Example:
    ```python
    from cuda_healthcheck.cuda_detector import CUDADetector

    detector = CUDADetector()
    environment = detector.detect_environment()
    print(f"CUDA Version: {environment.cuda_driver_version}")
    ```
"""

from .detector import (
    CUDADetector,
    CUDAEnvironment,
    GPUInfo,
    LibraryInfo,
    detect_cuda_environment,
)

__all__ = [
    "CUDADetector",
    "detect_cuda_environment",
    "GPUInfo",
    "LibraryInfo",
    "CUDAEnvironment",
]

__version__ = "1.0.0"
