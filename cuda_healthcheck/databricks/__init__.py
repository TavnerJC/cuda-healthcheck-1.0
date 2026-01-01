"""
Databricks Integration Module.

Provides high-level Databricks healthcheck functionality and low-level
API connector for cluster operations and Delta table management.

Supports both Classic ML Runtime clusters and Serverless GPU Compute.

Example:
    ```python
    # High-level healthcheck (recommended)
    from cuda_healthcheck.databricks import DatabricksHealthchecker, get_healthchecker

    checker = get_healthchecker()
    result = checker.run_healthcheck()
    checker.display_results()
    checker.export_results_to_delta("main.cuda.healthcheck_results")
    ```

Example:
    ```python
    # Low-level connector
    from cuda_healthcheck.databricks import DatabricksConnector

    connector = DatabricksConnector()
    cluster_info = connector.get_cluster_info("cluster-123")
    spark_conf = connector.get_spark_config("cluster-123")
    ```

Example:
    ```python
    # Auto-detecting GPU discovery (serverless-aware)
    from cuda_healthcheck.databricks import detect_gpu_auto, is_serverless_environment

    if is_serverless_environment():
        print("Running on Serverless GPU Compute")

    gpu_info = detect_gpu_auto()  # Automatically uses correct method
    print(f"Found {gpu_info.get('gpu_count', 0)} GPUs")
    ```
"""

from .connector import ClusterInfo, DatabricksConnector, is_databricks_environment
from .databricks_integration import (
    DatabricksHealthchecker,
    HealthcheckResult,
    get_healthchecker,
)
from .runtime_detector import (
    check_driver_compatibility,
    detect_databricks_runtime,
    get_driver_version_for_runtime,
    get_runtime_info_summary,
)
from .serverless import (
    detect_gpu_auto,
    detect_gpu_direct,
    detect_gpu_distributed,
    is_serverless_environment,
)

__all__ = [
    # High-level healthcheck
    "DatabricksHealthchecker",
    "get_healthchecker",
    "HealthcheckResult",
    # Low-level connector
    "DatabricksConnector",
    "is_databricks_environment",
    "ClusterInfo",
    # Runtime detection
    "detect_databricks_runtime",
    "get_runtime_info_summary",
    "get_driver_version_for_runtime",
    "check_driver_compatibility",
    # Serverless-aware GPU detection
    "detect_gpu_auto",
    "detect_gpu_direct",
    "detect_gpu_distributed",
    "is_serverless_environment",
]

__version__ = "1.0.0"
