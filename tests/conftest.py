"""
Pytest configuration and shared fixtures for CUDA Healthcheck tests.

Provides mock implementations of Databricks utilities, CUDA detection,
and common test data.
"""

import pytest
from unittest.mock import MagicMock, Mock
from typing import Dict, Any, List
from dataclasses import dataclass


# ============================================================================
# Mock Databricks Utilities
# ============================================================================


class MockNotebook:
    """Mock Databricks notebook utilities."""

    def run(self, path: str, timeout: int = 60, arguments: Dict = None) -> str:
        """Mock notebook.run()."""
        return "success"

    def entry_point(self):
        """Mock entry_point for context."""
        mock_context = MagicMock()
        mock_context.tags().get = lambda key: {
            "clusterId": "test-cluster-123",
            "clusterName": "Test Cluster",
        }.get(key)
        return MagicMock(
            getDbutils=lambda: MagicMock(
                notebook=lambda: MagicMock(getContext=lambda: mock_context)
            )
        )


class MockFS:
    """Mock Databricks filesystem utilities."""

    def ls(self, path: str) -> List[Dict[str, Any]]:
        """Mock fs.ls()."""
        return [
            {"path": f"{path}/file1.txt", "name": "file1.txt", "size": 1024},
            {"path": f"{path}/file2.txt", "name": "file2.txt", "size": 2048},
        ]

    def put(self, path: str, content: str, overwrite: bool = False) -> None:
        """Mock fs.put()."""
        pass

    def rm(self, path: str, recurse: bool = False) -> None:
        """Mock fs.rm()."""
        pass

    def mkdirs(self, path: str) -> None:
        """Mock fs.mkdirs()."""
        pass


class MockJobs:
    """Mock Databricks jobs utilities."""

    def runNow(self, job_id: int, notebook_params: Dict = None) -> Dict[str, Any]:
        """Mock jobs.runNow()."""
        return {"run_id": 12345}


class MockSecrets:
    """Mock Databricks secrets utilities."""

    def __init__(self):
        self._secrets = {
            "cuda-healthcheck": {
                "databricks-token": "dapi1234567890abcdef",
                "api-key": "test-key-123",
            }
        }

    def get(self, scope: str, key: str) -> str:
        """Mock secrets.get()."""
        return self._secrets.get(scope, {}).get(key, "")

    def list(self, scope: str) -> List[Dict[str, str]]:
        """Mock secrets.list()."""
        return [{"key": k} for k in self._secrets.get(scope, {}).keys()]


class MockDbutils:
    """
    Mock implementation of Databricks dbutils.

    Provides all common dbutils functionality for testing without
    requiring a Databricks environment.
    """

    def __init__(self):
        self.notebook = MockNotebook()
        self.fs = MockFS()
        self.jobs = MockJobs()
        self.secrets = MockSecrets()


@pytest.fixture
def mock_dbutils():
    """
    Provides mock dbutils for local testing.

    Example:
        ```python
        def test_databricks_function(mock_dbutils):
            result = mock_dbutils.notebook.run("/path/to/notebook")
            assert result == "success"
        ```
    """
    return MockDbutils()


# ============================================================================
# CUDA Detection Mocks
# ============================================================================


@pytest.fixture(params=["12.4", "12.6", "13.0"])
def cuda_versions(request):
    """
    Parameterized CUDA versions for testing.

    Tests using this fixture will run once for each CUDA version.

    Example:
        ```python
        def test_cuda_compatibility(cuda_versions):
            # This test runs 3 times with versions 12.4, 12.6, 13.0
            assert cuda_versions in ["12.4", "12.6", "13.0"]
        ```
    """
    return request.param


@pytest.fixture
def mock_gpu_info():
    """
    Mock GPU information.

    Returns:
        Dictionary with mock GPU details.
    """
    return {
        "name": "NVIDIA A100-SXM4-40GB",
        "driver_version": "550.54.15",
        "cuda_version": "12.4",
        "compute_capability": "8.0",
        "memory_total_mb": 40960,
        "gpu_index": 0,
    }


@pytest.fixture
def mock_cuda_environment(cuda_versions, mock_gpu_info):
    """
    Mock CUDA environment with parameterized versions.

    Returns:
        Dictionary with complete CUDA environment.
    """
    return {
        "cuda_runtime_version": cuda_versions,
        "cuda_driver_version": cuda_versions,
        "nvcc_version": cuda_versions,
        "gpus": [mock_gpu_info],
        "libraries": [
            {
                "name": "pytorch",
                "version": "2.0.0",
                "cuda_version": cuda_versions,
                "is_compatible": True,
                "warnings": [],
            },
            {
                "name": "tensorflow",
                "version": "2.13.0",
                "cuda_version": cuda_versions,
                "is_compatible": True,
                "warnings": [],
            },
        ],
        "breaking_changes": [],
        "timestamp": "2024-12-28T00:00:00",
    }


@pytest.fixture
def mock_cuda_detector(mock_cuda_environment):
    """
    Mock CUDADetector with pre-configured environment.

    Example:
        ```python
        def test_detection(mock_cuda_detector):
            env = mock_cuda_detector.detect_environment()
            assert env["cuda_driver_version"] is not None
        ```
    """
    mock_detector = MagicMock()
    mock_detector.detect_environment.return_value = mock_cuda_environment
    mock_detector.detect_nvidia_smi.return_value = {
        "success": True,
        "driver_version": "550.54.15",
        "cuda_version": "12.4",
        "gpus": [mock_cuda_environment["gpus"][0]],
    }
    mock_detector.detect_cuda_runtime.return_value = "12.4"
    mock_detector.detect_nvcc_version.return_value = "12.4"
    return mock_detector


# ============================================================================
# Databricks API Mocks
# ============================================================================


@pytest.fixture
def mock_cluster_info():
    """
    Mock Databricks cluster information.

    Returns:
        Dictionary with cluster details.
    """
    return {
        "cluster_id": "test-cluster-123",
        "cluster_name": "Test GPU Cluster",
        "state": "RUNNING",
        "spark_version": "13.3.x-gpu-ml-scala2.12",
        "node_type_id": "g5.xlarge",
        "driver_node_type_id": "g5.xlarge",
        "num_workers": 2,
        "spark_conf": {
            "spark.databricks.delta.preview.enabled": "true",
        },
        "custom_tags": {
            "Environment": "testing",
            "Project": "cuda-healthcheck",
        },
    }


@pytest.fixture
def mock_databricks_connector(mock_cluster_info):
    """
    Mock DatabricksConnector with pre-configured responses.

    Example:
        ```python
        def test_connector(mock_databricks_connector):
            cluster = mock_databricks_connector.get_cluster_info("test-cluster-123")
            assert cluster.cluster_name == "Test GPU Cluster"
        ```
    """
    from dataclasses import dataclass

    @dataclass
    class MockClusterInfo:
        cluster_id: str
        cluster_name: str
        state: str
        spark_version: str
        node_type_id: str
        driver_node_type_id: str
        num_workers: int
        spark_conf: Dict[str, str]
        custom_tags: Dict[str, str]

    mock_connector = MagicMock()
    mock_connector.get_cluster_info.return_value = MockClusterInfo(**mock_cluster_info)
    mock_connector.get_spark_config.return_value = mock_cluster_info["spark_conf"]
    mock_connector.list_clusters.return_value = [MockClusterInfo(**mock_cluster_info)]
    return mock_connector


# ============================================================================
# Breaking Changes Mocks
# ============================================================================


@pytest.fixture
def mock_breaking_changes():
    """
    Mock breaking changes database entries.

    Returns:
        List of breaking change dictionaries.
    """
    return [
        {
            "id": "pytorch-cuda13-rebuild",
            "title": "PyTorch requires rebuild for CUDA 13.x",
            "severity": "CRITICAL",
            "affected_library": "pytorch",
            "cuda_version_from": "12.x",
            "cuda_version_to": "13.0",
            "description": "PyTorch compiled for CUDA 12.x will not work with CUDA 13.x",
            "affected_apis": ["torch.cuda.is_available()"],
            "migration_path": "Install PyTorch CUDA 13.x builds",
            "references": ["https://pytorch.org"],
            "applies_to_compute_capabilities": None,
        },
        {
            "id": "tensorflow-cuda13-support",
            "title": "TensorFlow CUDA 13.x support requires TF 2.18+",
            "severity": "CRITICAL",
            "affected_library": "tensorflow",
            "cuda_version_from": "12.x",
            "cuda_version_to": "13.0",
            "description": "TensorFlow below 2.18 does not support CUDA 13.x",
            "affected_apis": ["tf.config.list_physical_devices()"],
            "migration_path": "Upgrade to TensorFlow 2.18+",
            "references": ["https://tensorflow.org"],
            "applies_to_compute_capabilities": None,
        },
    ]


@pytest.fixture
def mock_breaking_changes_db(mock_breaking_changes):
    """
    Mock BreakingChangesDatabase with pre-loaded changes.

    Example:
        ```python
        def test_breaking_changes(mock_breaking_changes_db):
            changes = mock_breaking_changes_db.get_changes_by_library("pytorch")
            assert len(changes) > 0
        ```
    """
    mock_db = MagicMock()
    mock_db.get_all_changes.return_value = mock_breaking_changes
    mock_db.get_changes_by_library.return_value = [mock_breaking_changes[0]]
    mock_db.score_compatibility.return_value = {
        "compatibility_score": 70,
        "total_issues": 2,
        "critical_issues": 1,
        "warning_issues": 1,
        "info_issues": 0,
        "breaking_changes": {
            "CRITICAL": [mock_breaking_changes[0]],
            "WARNING": [mock_breaking_changes[1]],
            "INFO": [],
        },
        "recommendation": "CAUTION: Environment has compatibility concerns.",
    }
    return mock_db


# ============================================================================
# Test Data
# ============================================================================


@pytest.fixture
def sample_healthcheck_result():
    """
    Sample healthcheck result for testing.

    Returns:
        Dictionary with complete healthcheck result.
    """
    return {
        "healthcheck_id": "healthcheck-20241228-120000",
        "cluster_id": "test-cluster-123",
        "cluster_name": "Test Cluster",
        "timestamp": "2024-12-28T12:00:00",
        "cuda_environment": {
            "cuda_runtime_version": "12.4",
            "cuda_driver_version": "12.4",
            "nvcc_version": "12.4",
            "gpus": [
                {
                    "name": "NVIDIA A100",
                    "driver_version": "550.54.15",
                    "cuda_version": "12.4",
                    "compute_capability": "8.0",
                    "memory_total_mb": 40960,
                    "gpu_index": 0,
                }
            ],
            "libraries": [
                {
                    "name": "pytorch",
                    "version": "2.0.0",
                    "cuda_version": "12.4",
                    "is_compatible": True,
                    "warnings": [],
                }
            ],
        },
        "compatibility_analysis": {
            "compatibility_score": 95,
            "total_issues": 0,
            "critical_issues": 0,
            "warning_issues": 0,
            "info_issues": 0,
        },
        "status": "healthy",
        "recommendations": ["Environment is healthy and well-configured"],
    }


# ============================================================================
# Environment Setup
# ============================================================================


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """
    Automatically set up test environment variables.

    This fixture runs for every test automatically.
    """
    monkeypatch.setenv("CUDA_HEALTHCHECK_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi_test_token_123")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "test-warehouse-123")
