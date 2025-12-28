"""
Databricks Cluster Scanner Module (Legacy).

This module provides cluster scanning functionality.
For new code, prefer using src.databricks.DatabricksHealthchecker.

Example:
    ```python
    from src.databricks_api import ClusterScanner

    scanner = ClusterScanner()
    results = scanner.scan_all_clusters()
    ```
"""

from .cluster_scanner import ClusterScanner, scan_clusters, ClusterHealthcheck

__all__ = ["ClusterScanner", "scan_clusters", "ClusterHealthcheck"]

__version__ = "1.0.0"
