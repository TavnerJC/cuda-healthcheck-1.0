"""
Tests for DatabricksConnector class.

Tests the low-level Databricks API connector functionality including
cluster information retrieval, Spark configuration, and Delta operations.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.databricks.connector import (
    ClusterInfo,
    DatabricksConnector,
    is_databricks_environment,
)
from src.utils.exceptions import (
    ClusterNotFoundError,
    ClusterNotRunningError,
    DatabricksConnectionError,
)


class TestDatabricksConnector:
    """Test suite for DatabricksConnector class."""

    def test_initialization_without_sdk(self):
        """Test initialization when Databricks SDK is not available."""
        with patch("src.databricks.connector.DATABRICKS_SDK_AVAILABLE", False):
            with pytest.raises(DatabricksConnectionError, match="SDK not installed"):
                DatabricksConnector()

    def test_initialization_without_credentials(self, monkeypatch):
        """Test initialization without credentials."""
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

        with pytest.raises(DatabricksConnectionError, match="credentials not provided"):
            DatabricksConnector()

    def test_initialization_with_credentials(self):
        """Test successful initialization with credentials."""
        with patch("src.databricks.connector.WorkspaceClient") as mock_client:
            connector = DatabricksConnector(
                workspace_url="https://test.databricks.com", token="test_token"
            )
            assert connector is not None
            assert connector.workspace_url == "https://test.databricks.com"
            assert connector.token == "test_token"

    def test_get_cluster_info_success(self, mock_databricks_connector, mock_cluster_info):
        """Test successful cluster info retrieval."""
        cluster_info = mock_databricks_connector.get_cluster_info("test-cluster-123")
        assert cluster_info.cluster_id == mock_cluster_info["cluster_id"]
        assert cluster_info.cluster_name == mock_cluster_info["cluster_name"]

    def test_get_cluster_info_not_found(self):
        """Test cluster info retrieval when cluster doesn't exist."""
        with patch("src.databricks.connector.WorkspaceClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.clusters.get.side_effect = Exception("does not exist")
            mock_client_class.return_value = mock_client

            connector = DatabricksConnector(
                workspace_url="https://test.databricks.com", token="test_token"
            )

            with pytest.raises(ClusterNotFoundError, match="not found"):
                connector.get_cluster_info("nonexistent-cluster")

    def test_get_spark_config(self, mock_databricks_connector, mock_cluster_info):
        """Test getting Spark configuration."""
        spark_conf = mock_databricks_connector.get_spark_config("test-cluster-123")
        assert isinstance(spark_conf, dict)
        assert spark_conf == mock_cluster_info["spark_conf"]

    def test_list_clusters(self, mock_databricks_connector):
        """Test listing all clusters."""
        clusters = mock_databricks_connector.list_clusters()
        assert isinstance(clusters, list)
        assert len(clusters) > 0

    def test_list_clusters_filter_gpu(self, mock_databricks_connector):
        """Test listing only GPU clusters."""
        clusters = mock_databricks_connector.list_clusters(filter_gpu=True)
        assert isinstance(clusters, list)

    def test_is_databricks_environment_true(self):
        """Test is_databricks_environment when in Databricks."""
        with patch("os.environ", {"DATABRICKS_RUNTIME_VERSION": "13.3"}):
            assert is_databricks_environment() is True

    def test_is_databricks_environment_false(self, monkeypatch):
        """Test is_databricks_environment when not in Databricks."""
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        # IPython check will also fail in test environment
        result = is_databricks_environment()
        assert result is False


class TestClusterInfo:
    """Test suite for ClusterInfo dataclass."""

    def test_cluster_info_creation(self, mock_cluster_info):
        """Test creating ClusterInfo instance."""
        cluster = ClusterInfo(**mock_cluster_info)
        assert cluster.cluster_id == "test-cluster-123"
        assert cluster.cluster_name == "Test GPU Cluster"
        assert cluster.state == "RUNNING"
        assert cluster.num_workers == 2


class TestRetryBehavior:
    """Test retry functionality in connector."""

    def test_get_cluster_info_with_retry(self):
        """Test that get_cluster_info retries on failure."""
        with patch("src.databricks.connector.WorkspaceClient") as mock_client_class:
            mock_client = MagicMock()

            # Fail twice, then succeed
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] < 3:
                    raise Exception("Temporary failure")
                mock_cluster = MagicMock()
                mock_cluster.cluster_id = "test-123"
                mock_cluster.cluster_name = "Test Cluster"
                mock_cluster.state = MagicMock(value="RUNNING")
                mock_cluster.spark_version = "13.3"
                mock_cluster.node_type_id = "g5.xlarge"
                mock_cluster.driver_node_type_id = "g5.xlarge"
                mock_cluster.num_workers = 2
                mock_cluster.spark_conf = {}
                mock_cluster.custom_tags = {}
                return mock_cluster

            mock_client.clusters.get.side_effect = side_effect
            mock_client_class.return_value = mock_client

            connector = DatabricksConnector(
                workspace_url="https://test.databricks.com", token="test_token"
            )

            # Should succeed after retries
            cluster_info = connector.get_cluster_info("test-123")
            assert cluster_info.cluster_id == "test-123"
            assert call_count[0] == 3  # Called 3 times (2 failures + 1 success)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
