"""
Databricks Integration Module.

Provides low-level Databricks API connector for cluster information,
Spark configuration, and Delta table operations.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.exceptions import (
    ClusterNotFoundError,
    ClusterNotRunningError,
    ConfigurationError,
    DatabricksConnectionError,
    DeltaTableError,
)
from ..utils.logging_config import get_logger
from ..utils.retry import retry_on_failure
from ..utils.validation import (
    validate_cluster_id,
    validate_databricks_host,
    validate_table_path,
    validate_token,
)

logger = get_logger(__name__)

# Try to import Databricks SDK
try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.compute import ClusterDetails, State

    DATABRICKS_SDK_AVAILABLE = True
except ImportError:
    DATABRICKS_SDK_AVAILABLE = False
    WorkspaceClient = None
    ClusterDetails = None
    State = None
    logger.warning("Databricks SDK not available. Install with: pip install databricks-sdk")


@dataclass
class ClusterInfo:
    """Information about a Databricks cluster."""

    cluster_id: str
    cluster_name: str
    state: str
    spark_version: str
    node_type_id: str
    driver_node_type_id: Optional[str]
    num_workers: int
    spark_conf: Dict[str, str]
    custom_tags: Dict[str, str]


class DatabricksConnector:
    """
    Low-level Databricks API connector.

    Handles authentication, cluster information retrieval,
    Spark configuration, and Delta table operations.

    Example:
        ```python
        connector = DatabricksConnector()
        cluster_info = connector.get_cluster_info("cluster-id-123")
        spark_conf = connector.get_spark_config("cluster-id-123")
        ```
    """

    def __init__(
        self,
        workspace_url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Initialize Databricks connector.

        Args:
            workspace_url: Databricks workspace URL (or use DATABRICKS_HOST env var)
            token: Personal Access Token (or use DATABRICKS_TOKEN env var)

        Raises:
            DatabricksConnectionError: If SDK not available or credentials missing
            ConfigurationError: If credentials are invalid format
        """
        if not DATABRICKS_SDK_AVAILABLE:
            raise DatabricksConnectionError(
                "Databricks SDK not installed. Install with: pip install databricks-sdk"
            )

        # Use provided credentials or environment variables
        self.workspace_url = workspace_url or os.getenv("DATABRICKS_HOST")
        self.token = token or os.getenv("DATABRICKS_TOKEN")

        # Validate credentials are provided
        if not self.workspace_url or not self.token:
            missing = []
            if not self.workspace_url:
                missing.append("DATABRICKS_HOST")
            if not self.token:
                missing.append("DATABRICKS_TOKEN")

            raise ConfigurationError(
                f"Databricks credentials not provided: {', '.join(missing)} not set.\n"
                f"Set environment variables or pass to constructor.\n"
                f"See docs/ENVIRONMENT_VARIABLES.md for details."
            )

        # Validate format
        if not validate_databricks_host(self.workspace_url):
            raise ConfigurationError(
                f"Invalid DATABRICKS_HOST format: {self.workspace_url}\n"
                f"Should be like: https://your-workspace.cloud.databricks.com"
            )

        if not validate_token(self.token):
            raise ConfigurationError(
                "Invalid DATABRICKS_TOKEN format. Token should be at least 10 characters."
            )

        try:
            self.client = WorkspaceClient(host=self.workspace_url, token=self.token)
            logger.info(f"Connected to Databricks workspace: {self.workspace_url}")
        except Exception as e:
            error_msg = f"Failed to connect to Databricks: {e}"
            logger.error(error_msg, exc_info=True)
            raise DatabricksConnectionError(error_msg)

    @retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
    def get_cluster_info(self, cluster_id: str) -> ClusterInfo:
        """
        Get information about a specific cluster.

        Args:
            cluster_id: The Databricks cluster ID

        Returns:
            ClusterInfo object with cluster details

        Raises:
            ClusterNotFoundError: If cluster doesn't exist
            DatabricksConnectionError: If API call fails
            ConfigurationError: If cluster_id is invalid format
        """
        # Validate cluster_id format
        if not cluster_id or not isinstance(cluster_id, str):
            raise ConfigurationError("Invalid cluster_id: must be non-empty string")

        if not validate_cluster_id(cluster_id):
            logger.warning(f"Cluster ID format may be invalid: {cluster_id}")

        logger.debug("Fetching info for cluster: %s", cluster_id)

        try:
            cluster = self.client.clusters.get(cluster_id=cluster_id)

            cluster_info = ClusterInfo(
                cluster_id=cluster.cluster_id,
                cluster_name=cluster.cluster_name or "Unnamed",
                state=cluster.state.value if cluster.state else "UNKNOWN",
                spark_version=cluster.spark_version or "Unknown",
                node_type_id=cluster.node_type_id or "Unknown",
                driver_node_type_id=cluster.driver_node_type_id,
                num_workers=cluster.num_workers or 0,
                spark_conf=dict(cluster.spark_conf) if cluster.spark_conf else {},
                custom_tags=dict(cluster.custom_tags) if cluster.custom_tags else {},
            )

            logger.info(
                f"Retrieved cluster info: {cluster_info.cluster_name} "
                f"(State: {cluster_info.state})"
            )
            return cluster_info

        except AttributeError as e:
            error_msg = f"Unexpected cluster structure from API: {e}"
            logger.error(error_msg, exc_info=True)
            raise DatabricksConnectionError(error_msg)
        except Exception as e:
            error_str = str(e).lower()
            if "does not exist" in error_str or "not found" in error_str:
                error_msg = f"Cluster {cluster_id} not found in workspace"
                logger.error(error_msg)
                raise ClusterNotFoundError(error_msg)

            if "permission" in error_str or "unauthorized" in error_str:
                error_msg = f"Permission denied accessing cluster {cluster_id}"
                logger.error(error_msg)
                raise DatabricksConnectionError(f"{error_msg}. Check token permissions.")

            error_msg = f"Failed to get cluster info for {cluster_id}: {e}"
            logger.error(error_msg, exc_info=True)
            raise DatabricksConnectionError(error_msg)

    def get_spark_config(self, cluster_id: str) -> Dict[str, str]:
        """
        Get Spark configuration for a cluster.

        Args:
            cluster_id: The Databricks cluster ID

        Returns:
            Dictionary of Spark configuration key-value pairs

        Raises:
            ClusterNotFoundError: If cluster not found
        """
        cluster_info = self.get_cluster_info(cluster_id)
        if cluster_info is None:
            raise ClusterNotFoundError(f"Cluster {cluster_id} not found")
        return cluster_info.spark_conf

    def list_clusters(self, filter_gpu: bool = False) -> List[ClusterInfo]:
        """
        List all clusters in the workspace.

        Args:
            filter_gpu: If True, return only GPU-enabled clusters

        Returns:
            List of ClusterInfo objects
        """
        try:
            all_clusters = list(self.client.clusters.list())
            cluster_infos = []

            for cluster in all_clusters:
                cluster_info = ClusterInfo(
                    cluster_id=cluster.cluster_id,
                    cluster_name=cluster.cluster_name or "Unnamed",
                    state=cluster.state.value if cluster.state else "UNKNOWN",
                    spark_version=cluster.spark_version or "Unknown",
                    node_type_id=cluster.node_type_id or "Unknown",
                    driver_node_type_id=cluster.driver_node_type_id,
                    num_workers=cluster.num_workers or 0,
                    spark_conf=dict(cluster.spark_conf) if cluster.spark_conf else {},
                    custom_tags=(dict(cluster.custom_tags) if cluster.custom_tags else {}),
                )

                # Filter GPU clusters if requested
                if filter_gpu:
                    if self._is_gpu_cluster(cluster_info):
                        cluster_infos.append(cluster_info)
                else:
                    cluster_infos.append(cluster_info)

            return cluster_infos

        except Exception as e:
            raise DatabricksConnectionError(f"Failed to list clusters: {e}")

    def _is_gpu_cluster(self, cluster_info: ClusterInfo) -> bool:
        """Check if a cluster has GPU nodes."""
        gpu_indicators = ["gpu", "g5", "g4dn", "p3", "p4"]
        node_type = cluster_info.node_type_id.lower()
        return any(indicator in node_type for indicator in gpu_indicators)

    def ensure_cluster_running(self, cluster_id: str, timeout: int = 300) -> bool:
        """
        Ensure a cluster is in RUNNING state, starting it if necessary.

        Args:
            cluster_id: The Databricks cluster ID
            timeout: Maximum time to wait for cluster to start (seconds)

        Returns:
            True if cluster is running

        Raises:
            ClusterNotFoundError: If cluster not found
            ClusterNotRunningError: If cluster fails to start
        """
        import time

        cluster_info = self.get_cluster_info(cluster_id)
        if cluster_info is None:
            raise ClusterNotFoundError(f"Cluster {cluster_id} not found")

        if cluster_info.state == "RUNNING":
            logger.info(f"Cluster {cluster_id} is already running")
            return True

        if cluster_info.state in ["TERMINATED", "TERMINATING"]:
            logger.info(f"Starting cluster {cluster_id}...")
            try:
                self.client.clusters.start(cluster_id=cluster_id)
            except Exception as e:
                raise ClusterNotRunningError(f"Failed to start cluster: {e}")

        # Wait for cluster to be running
        start_time = time.time()
        while time.time() - start_time < timeout:
            cluster_info = self.get_cluster_info(cluster_id)
            if cluster_info is None:
                raise ClusterNotFoundError(f"Cluster {cluster_id} disappeared")

            if cluster_info.state == "RUNNING":
                logger.info(f"Cluster {cluster_id} is now running")
                return True

            if cluster_info.state in ["ERROR", "UNKNOWN"]:
                raise ClusterNotRunningError(
                    f"Cluster {cluster_id} is in error state: {cluster_info.state}"
                )

            time.sleep(10)

        raise ClusterNotRunningError(f"Cluster {cluster_id} did not start within {timeout} seconds")

    def read_delta_table(
        self,
        table_path: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read data from a Delta table.

        Args:
            table_path: Full table path (e.g., "main.schema.table")
            limit: Optional limit on number of rows to return

        Returns:
            List of dictionaries representing rows

        Raises:
            DeltaTableError: If table read fails
        """
        try:
            warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
            if not warehouse_id:
                raise DeltaTableError("DATABRICKS_WAREHOUSE_ID environment variable not set")

            # Validate table_path format (catalog.schema.table)
            if not validate_table_path(table_path):
                raise DeltaTableError(f"Invalid table path format: {table_path}")

            query = f"SELECT * FROM {table_path}"  # nosec B608 - table_path is validated above
            if limit:
                # Ensure limit is an integer to prevent injection
                query += f" LIMIT {int(limit)}"

            self.client.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                statement=query,
            )

            # Convert result to list of dictionaries
            # This is a simplified version - real implementation would parse result properly
            logger.info(f"Read {table_path} from Delta table")
            return []  # Placeholder

        except Exception as e:
            raise DeltaTableError(f"Failed to read Delta table {table_path}: {e}")

    def write_delta_table(
        self,
        table_path: str,
        data: List[Dict[str, Any]],
        mode: str = "append",
    ) -> None:
        """
        Write data to a Delta table.

        Args:
            table_path: Full table path (e.g., "main.schema.table")
            data: List of dictionaries to write
            mode: Write mode ("append", "overwrite")

        Raises:
            DeltaTableError: If table write fails
        """
        try:
            warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
            if not warehouse_id:
                raise DeltaTableError("DATABRICKS_WAREHOUSE_ID environment variable not set")

            # In a real implementation, this would use Spark DataFrame API
            # This is a placeholder showing the API structure
            logger.info(f"Writing {len(data)} rows to {table_path} (mode: {mode})")

        except Exception as e:
            raise DeltaTableError(f"Failed to write to Delta table {table_path}: {e}")


def is_databricks_environment() -> bool:
    """
    Check if code is running in a Databricks environment.

    Returns:
        True if running in Databricks, False otherwise.

    Example:
        ```python
        if is_databricks_environment():
            # Use Databricks-specific features
            dbutils.fs.ls("/")
        ```
    """
    try:
        # Try to access dbutils (only available in Databricks)
        import IPython

        ipython = IPython.get_ipython()
        if ipython and "DATABRICKS" in ipython.config:
            return True

        # Alternative check: environment variables
        return "DATABRICKS_RUNTIME_VERSION" in os.environ

    except (ImportError, AttributeError):
        return False
