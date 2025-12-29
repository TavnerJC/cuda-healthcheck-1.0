"""
Databricks Integration Module.

Provides high-level Databricks healthcheck functionality and low-level
API connector for cluster operations and Delta table management.

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
"""

from .connector import ClusterInfo, DatabricksConnector, is_databricks_environment
from .databricks_integration import (
    DatabricksHealthchecker,
    HealthcheckResult,
    get_healthchecker,
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
]

__version__ = "1.0.0"
