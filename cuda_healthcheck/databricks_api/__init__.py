"""
Databricks Cluster Scanner Module (Legacy).

This module provides cluster scanning functionality.
For new code, prefer using src.databricks.DatabricksHealthchecker.

Example:
    ```python
    from cuda_healthcheck.databricks_api import ClusterScanner

    scanner = ClusterScanner()
    results = scanner.scan_all_clusters()
    ```
"""

from .cluster_scanner import ClusterHealthcheck, ClusterScanner, scan_clusters

__all__ = ["ClusterScanner", "scan_clusters", "ClusterHealthcheck"]

__version__ = "1.0.0"
