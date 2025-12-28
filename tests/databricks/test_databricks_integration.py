"""
Tests for DatabricksHealthchecker class.

Tests the high-level Databricks healthcheck functionality including
CUDA detection, compatibility analysis, and Delta table export.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.databricks.databricks_integration import (
    DatabricksHealthchecker,
    get_healthchecker,
    HealthcheckResult,
)


class TestDatabricksHealthchecker:
    """Test suite for DatabricksHealthchecker class."""

    def test_initialization(self):
        """Test healthchecker initialization."""
        checker = DatabricksHealthchecker()
        assert checker is not None
        assert checker.cuda_detector is not None
        assert checker.breaking_changes_db is not None

    def test_initialization_with_cluster_id(self):
        """Test healthchecker initialization with specific cluster ID."""
        checker = DatabricksHealthchecker(cluster_id="test-123")
        assert checker.cluster_id == "test-123"

    def test_get_cluster_cuda_version(self, mock_cuda_detector):
        """Test getting CUDA version from cluster."""
        with patch("src.databricks.databricks_integration.CUDADetector") as mock_class:
            mock_class.return_value = mock_cuda_detector
            checker = DatabricksHealthchecker()
            version = checker.get_cluster_cuda_version()
            assert version is not None

    def test_get_cluster_metadata(self):
        """Test getting cluster metadata."""
        checker = DatabricksHealthchecker()
        metadata = checker.get_cluster_metadata()

        assert isinstance(metadata, dict)
        assert "is_databricks" in metadata
        assert "cluster_id" in metadata
        assert "runtime_version" in metadata

    def test_run_healthcheck(self, mock_cuda_environment, mock_breaking_changes_db):
        """Test running complete healthcheck."""
        with patch("src.databricks.databricks_integration.CUDADetector") as mock_detector_class:
            with patch(
                "src.databricks.databricks_integration.BreakingChangesDatabase"
            ) as mock_db_class:
                # Setup mocks
                from dataclasses import dataclass

                @dataclass
                class MockGPU:
                    name: str = "NVIDIA A100"
                    driver_version: str = "550.54.15"
                    cuda_version: str = "12.4"
                    compute_capability: str = "8.0"
                    memory_total_mb: int = 40960
                    gpu_index: int = 0

                @dataclass
                class MockLibrary:
                    name: str = "pytorch"
                    version: str = "2.0.0"
                    cuda_version: str = "12.4"
                    is_compatible: bool = True
                    warnings: list = None

                    def __post_init__(self):
                        if self.warnings is None:
                            self.warnings = []

                @dataclass
                class MockEnvironment:
                    cuda_runtime_version: str = "12.4"
                    cuda_driver_version: str = "12.4"
                    nvcc_version: str = "12.4"
                    gpus: list = None
                    libraries: list = None
                    breaking_changes: list = None
                    timestamp: str = "2024-12-28T00:00:00"

                    def __post_init__(self):
                        if self.gpus is None:
                            self.gpus = [MockGPU()]
                        if self.libraries is None:
                            self.libraries = [MockLibrary()]
                        if self.breaking_changes is None:
                            self.breaking_changes = []

                mock_detector = MagicMock()
                mock_detector.detect_environment.return_value = MockEnvironment()
                mock_detector_class.return_value = mock_detector

                mock_db = MagicMock()
                mock_db.score_compatibility.return_value = {
                    "compatibility_score": 95,
                    "total_issues": 0,
                    "critical_issues": 0,
                    "warning_issues": 0,
                    "info_issues": 0,
                    "breaking_changes": {"CRITICAL": [], "WARNING": [], "INFO": []},
                    "recommendation": "Environment is healthy",
                }
                mock_db_class.return_value = mock_db

                # Run healthcheck
                checker = DatabricksHealthchecker()
                result = checker.run_healthcheck()

                # Verify result
                assert isinstance(result, dict)
                assert "healthcheck_id" in result
                assert "timestamp" in result
                assert "status" in result
                assert result["status"] == "healthy"
                assert "cuda_environment" in result
                assert "compatibility_analysis" in result

    def test_run_healthcheck_with_warnings(self, mock_cuda_environment):
        """Test healthcheck with warning-level issues."""
        with patch("src.databricks.databricks_integration.CUDADetector") as mock_detector_class:
            with patch(
                "src.databricks.databricks_integration.BreakingChangesDatabase"
            ) as mock_db_class:
                from dataclasses import dataclass

                @dataclass
                class MockEnvironment:
                    cuda_runtime_version: str = "12.4"
                    cuda_driver_version: str = "12.4"
                    nvcc_version: str = "12.4"
                    gpus: list = None
                    libraries: list = None
                    breaking_changes: list = None
                    timestamp: str = "2024-12-28T00:00:00"

                    def __post_init__(self):
                        if self.gpus is None:
                            self.gpus = []
                        if self.libraries is None:
                            self.libraries = []
                        if self.breaking_changes is None:
                            self.breaking_changes = []

                mock_detector = MagicMock()
                mock_detector.detect_environment.return_value = MockEnvironment()
                mock_detector_class.return_value = mock_detector

                mock_db = MagicMock()
                mock_db.score_compatibility.return_value = {
                    "compatibility_score": 75,
                    "total_issues": 2,
                    "critical_issues": 0,
                    "warning_issues": 2,
                    "info_issues": 0,
                    "breaking_changes": {"CRITICAL": [], "WARNING": [], "INFO": []},
                    "recommendation": "Review warnings",
                }
                mock_db_class.return_value = mock_db

                checker = DatabricksHealthchecker()
                result = checker.run_healthcheck()

                assert result["status"] == "warning"

    def test_run_healthcheck_with_critical_issues(self):
        """Test healthcheck with critical issues."""
        with patch("src.databricks.databricks_integration.CUDADetector") as mock_detector_class:
            with patch(
                "src.databricks.databricks_integration.BreakingChangesDatabase"
            ) as mock_db_class:
                from dataclasses import dataclass

                @dataclass
                class MockEnvironment:
                    cuda_runtime_version: str = "13.0"
                    cuda_driver_version: str = "13.0"
                    nvcc_version: str = "13.0"
                    gpus: list = None
                    libraries: list = None
                    breaking_changes: list = None
                    timestamp: str = "2024-12-28T00:00:00"

                    def __post_init__(self):
                        if self.gpus is None:
                            self.gpus = []
                        if self.libraries is None:
                            self.libraries = []
                        if self.breaking_changes is None:
                            self.breaking_changes = []

                mock_detector = MagicMock()
                mock_detector.detect_environment.return_value = MockEnvironment()
                mock_detector_class.return_value = mock_detector

                mock_db = MagicMock()
                mock_db.score_compatibility.return_value = {
                    "compatibility_score": 40,
                    "total_issues": 3,
                    "critical_issues": 2,
                    "warning_issues": 1,
                    "info_issues": 0,
                    "breaking_changes": {"CRITICAL": [], "WARNING": [], "INFO": []},
                    "recommendation": "Critical issues detected",
                }
                mock_db_class.return_value = mock_db

                checker = DatabricksHealthchecker()
                result = checker.run_healthcheck()

                assert result["status"] == "critical"

    def test_export_results_to_delta_no_results(self):
        """Test export to Delta when no results available."""
        checker = DatabricksHealthchecker()
        success = checker.export_results_to_delta("main.test.table")
        assert success is False

    def test_display_results_no_results(self, capsys):
        """Test displaying results when none available."""
        checker = DatabricksHealthchecker()
        checker.display_results()
        captured = capsys.readouterr()
        assert "No healthcheck results" in captured.out

    def test_display_results(self, capsys, sample_healthcheck_result):
        """Test displaying healthcheck results."""
        checker = DatabricksHealthchecker()
        checker.display_results(sample_healthcheck_result)
        captured = capsys.readouterr()

        assert "CUDA HEALTHCHECK RESULTS" in captured.out
        assert "healthcheck-20241228-120000" in captured.out
        assert "healthy" in captured.out.lower()


class TestGetHealthchecker:
    """Test suite for get_healthchecker factory function."""

    def test_get_healthchecker_no_credentials(self, monkeypatch):
        """Test factory function without credentials."""
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

        checker = get_healthchecker()
        assert isinstance(checker, DatabricksHealthchecker)
        assert checker.connector is None

    def test_get_healthchecker_with_credentials(self):
        """Test factory function with credentials."""
        # Credentials are set in setup_test_environment fixture
        checker = get_healthchecker()
        assert isinstance(checker, DatabricksHealthchecker)

    def test_get_healthchecker_with_cluster_id(self):
        """Test factory function with cluster ID."""
        checker = get_healthchecker(cluster_id="test-123")
        assert checker.cluster_id == "test-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
