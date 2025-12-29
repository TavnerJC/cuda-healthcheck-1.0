"""
Unit tests for HealthcheckOrchestrator.

Tests can be run locally without any dependencies on Databricks or CUDA.
Uses mocks to simulate CUDA detection and breaking changes database.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.healthcheck.orchestrator import (
    HealthcheckOrchestrator,
    HealthcheckReport,
    run_complete_healthcheck,
)


@dataclass
class MockGPU:
    """Mock GPU for testing."""

    name: str = "NVIDIA A100"
    driver_version: str = "550.54.15"
    cuda_version: str = "12.4"
    compute_capability: str = "8.0"
    memory_total_mb: int = 40960
    gpu_index: int = 0


@dataclass
class MockLibrary:
    """Mock library for testing."""

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
    """Mock CUDA environment for testing."""

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


class TestHealthcheckOrchestratorInitialization:
    """Test HealthcheckOrchestrator initialization."""

    def test_initialization(self):
        """Test basic orchestrator initialization."""
        orchestrator = HealthcheckOrchestrator()
        assert orchestrator is not None
        assert orchestrator.detector is not None
        assert orchestrator.breaking_changes_db is not None
        assert orchestrator.last_environment is None
        assert orchestrator.last_report is None

    def test_has_detector(self):
        """Test that orchestrator has detector instance."""
        orchestrator = HealthcheckOrchestrator()
        assert hasattr(orchestrator, "detector")

    def test_has_breaking_changes_db(self):
        """Test that orchestrator has breaking changes database."""
        orchestrator = HealthcheckOrchestrator()
        assert hasattr(orchestrator, "breaking_changes_db")


class TestCheckCompatibility:
    """Test check_compatibility method."""

    def test_check_compatibility_same_version(self):
        """Test compatibility check with same version."""
        orchestrator = HealthcheckOrchestrator()
        result = orchestrator.check_compatibility("12.4", "12.4")

        assert isinstance(result, dict)
        assert "local_version" in result
        assert "cluster_version" in result
        assert "compatible" in result
        assert result["local_version"] == "12.4"
        assert result["cluster_version"] == "12.4"

    def test_check_compatibility_different_versions(self):
        """Test compatibility check with different versions."""
        orchestrator = HealthcheckOrchestrator()
        result = orchestrator.check_compatibility("12.4", "13.0")

        assert isinstance(result, dict)
        assert "breaking_changes" in result
        assert "total_changes" in result
        assert "recommendation" in result

    def test_check_compatibility_structure(self):
        """Test that compatibility result has correct structure."""
        orchestrator = HealthcheckOrchestrator()
        result = orchestrator.check_compatibility("12.4", "12.6")

        assert "local_version" in result
        assert "cluster_version" in result
        assert "compatible" in result
        assert "has_breaking_changes" in result
        assert "breaking_changes" in result
        assert "total_changes" in result
        assert "recommendation" in result

        # Breaking changes should have severity categories
        assert "critical" in result["breaking_changes"]
        assert "warning" in result["breaking_changes"]
        assert "info" in result["breaking_changes"]


class TestAnalyzeBreakingChanges:
    """Test analyze_breaking_changes method."""

    def test_analyze_breaking_changes_basic(self):
        """Test basic breaking changes analysis."""
        orchestrator = HealthcheckOrchestrator()
        libraries = [
            {
                "name": "pytorch",
                "version": "2.0.0",
                "cuda_version": "12.4",
                "is_compatible": True,
                "warnings": [],
            }
        ]

        result = orchestrator.analyze_breaking_changes(libraries, "12.4")

        assert isinstance(result, dict)
        assert "compatibility_score" in result
        assert "total_issues" in result
        assert "critical_issues" in result
        assert "warning_issues" in result

    def test_analyze_breaking_changes_with_compute_capability(self):
        """Test analysis with compute capability."""
        orchestrator = HealthcheckOrchestrator()
        libraries = [
            {
                "name": "pytorch",
                "version": "2.0.0",
                "cuda_version": "12.4",
                "is_compatible": True,
                "warnings": [],
            }
        ]

        result = orchestrator.analyze_breaking_changes(
            libraries, "12.4", compute_capability="8.0"
        )

        assert "compatibility_score" in result


class TestGenerateReport:
    """Test generate_report method."""

    @patch("src.healthcheck.orchestrator.CUDADetector")
    @patch("src.healthcheck.orchestrator.BreakingChangesDatabase")
    def test_generate_report_structure(self, mock_db_class, mock_detector_class):
        """Test that generated report has correct structure."""
        # Setup mocks
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

        orchestrator = HealthcheckOrchestrator()
        report = orchestrator.generate_report()

        assert isinstance(report, HealthcheckReport)
        assert report.healthcheck_id.startswith("healthcheck-")
        assert report.timestamp is not None
        assert report.cuda_environment is not None
        assert report.compatibility_analysis is not None
        assert report.status in ["healthy", "warning", "critical"]
        assert isinstance(report.recommendations, list)

    @patch("src.healthcheck.orchestrator.CUDADetector")
    @patch("src.healthcheck.orchestrator.BreakingChangesDatabase")
    def test_generate_report_healthy_status(self, mock_db_class, mock_detector_class):
        """Test report generation with healthy status."""
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
            "recommendation": "Healthy",
        }
        mock_db_class.return_value = mock_db

        orchestrator = HealthcheckOrchestrator()
        report = orchestrator.generate_report()

        assert report.status == "healthy"

    @patch("src.healthcheck.orchestrator.CUDADetector")
    @patch("src.healthcheck.orchestrator.BreakingChangesDatabase")
    def test_generate_report_warning_status(self, mock_db_class, mock_detector_class):
        """Test report generation with warning status."""
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
            "breaking_changes": {"CRITICAL": [], "WARNING": [{}, {}], "INFO": []},
            "recommendation": "Review warnings",
        }
        mock_db_class.return_value = mock_db

        orchestrator = HealthcheckOrchestrator()
        report = orchestrator.generate_report()

        assert report.status == "warning"

    @patch("src.healthcheck.orchestrator.CUDADetector")
    @patch("src.healthcheck.orchestrator.BreakingChangesDatabase")
    def test_generate_report_critical_status(self, mock_db_class, mock_detector_class):
        """Test report generation with critical status."""
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
            "breaking_changes": {"CRITICAL": [{}, {}], "WARNING": [{}], "INFO": []},
            "recommendation": "Critical issues",
        }
        mock_db_class.return_value = mock_db

        orchestrator = HealthcheckOrchestrator()
        report = orchestrator.generate_report()

        assert report.status == "critical"

    @patch("src.healthcheck.orchestrator.CUDADetector")
    @patch("src.healthcheck.orchestrator.BreakingChangesDatabase")
    def test_generate_report_stores_last_report(
        self, mock_db_class, mock_detector_class
    ):
        """Test that generate_report stores the report."""
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
            "recommendation": "Healthy",
        }
        mock_db_class.return_value = mock_db

        orchestrator = HealthcheckOrchestrator()
        report = orchestrator.generate_report()

        assert orchestrator.last_report is not None
        assert orchestrator.last_report == report


class TestSaveReportJson:
    """Test save_report_json method."""

    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_save_report_json(self, mock_json_dump, mock_open):
        """Test saving report to JSON."""
        orchestrator = HealthcheckOrchestrator()

        # Create a mock report
        report = HealthcheckReport(
            healthcheck_id="test-123",
            timestamp="2024-12-28T00:00:00",
            cuda_environment={},
            compatibility_analysis={},
            status="healthy",
            recommendations=[],
        )
        orchestrator.last_report = report

        orchestrator.save_report_json(filepath="test.json")

        mock_open.assert_called_once_with("test.json", "w")
        mock_json_dump.assert_called_once()

    def test_save_report_json_no_report(self):
        """Test saving when no report exists."""
        orchestrator = HealthcheckOrchestrator()
        # Should not raise error, just log
        orchestrator.save_report_json()


class TestPrintReportSummary:
    """Test print_report_summary method."""

    def test_print_report_summary(self, capsys):
        """Test printing report summary."""
        orchestrator = HealthcheckOrchestrator()

        report = HealthcheckReport(
            healthcheck_id="test-123",
            timestamp="2024-12-28T00:00:00",
            cuda_environment={
                "cuda_runtime_version": "12.4",
                "cuda_driver_version": "12.4",
                "nvcc_version": "12.4",
            },
            compatibility_analysis={
                "compatibility_score": 95,
                "critical_issues": 0,
                "warning_issues": 0,
            },
            status="healthy",
            recommendations=["Environment is healthy"],
        )

        orchestrator.print_report_summary(report)

        captured = capsys.readouterr()
        assert "CUDA HEALTHCHECK SUMMARY" in captured.out
        assert "test-123" in captured.out
        assert "healthy" in captured.out.lower()

    def test_print_report_summary_no_report(self, capsys):
        """Test printing when no report exists."""
        orchestrator = HealthcheckOrchestrator()
        orchestrator.print_report_summary()

        captured = capsys.readouterr()
        assert "No report available" in captured.out


class TestRunCompleteHealthcheck:
    """Test run_complete_healthcheck convenience function."""

    @patch("src.healthcheck.orchestrator.HealthcheckOrchestrator")
    def test_run_complete_healthcheck(self, mock_orchestrator_class):
        """Test convenience function."""
        mock_orchestrator = MagicMock()
        mock_report = HealthcheckReport(
            healthcheck_id="test-123",
            timestamp="2024-12-28T00:00:00",
            cuda_environment={},
            compatibility_analysis={},
            status="healthy",
            recommendations=[],
        )
        mock_orchestrator.generate_report.return_value = mock_report
        mock_orchestrator_class.return_value = mock_orchestrator

        result = run_complete_healthcheck()

        assert isinstance(result, dict)
        assert result["healthcheck_id"] == "test-123"
        assert result["status"] == "healthy"


class TestRecommendationsGeneration:
    """Test recommendation generation logic."""

    @patch("src.healthcheck.orchestrator.CUDADetector")
    @patch("src.healthcheck.orchestrator.BreakingChangesDatabase")
    def test_recommendations_include_critical_warning(
        self, mock_db_class, mock_detector_class
    ):
        """Test that critical issues generate warning recommendation."""
        mock_detector = MagicMock()
        mock_detector.detect_environment.return_value = MockEnvironment()
        mock_detector_class.return_value = mock_detector

        mock_db = MagicMock()
        mock_db.score_compatibility.return_value = {
            "compatibility_score": 40,
            "total_issues": 2,
            "critical_issues": 2,
            "warning_issues": 0,
            "info_issues": 0,
            "breaking_changes": {"CRITICAL": [{}, {}], "WARNING": [], "INFO": []},
            "recommendation": "Critical",
        }
        mock_db_class.return_value = mock_db

        orchestrator = HealthcheckOrchestrator()
        report = orchestrator.generate_report()

        # Should have critical warning in recommendations
        assert any("CRITICAL" in rec for rec in report.recommendations)

    @patch("src.healthcheck.orchestrator.CUDADetector")
    @patch("src.healthcheck.orchestrator.BreakingChangesDatabase")
    def test_recommendations_for_cuda_13(self, mock_db_class, mock_detector_class):
        """Test recommendations for CUDA 13.x."""
        mock_env = MockEnvironment()
        mock_env.cuda_driver_version = "13.0"

        mock_detector = MagicMock()
        mock_detector.detect_environment.return_value = mock_env
        mock_detector_class.return_value = mock_detector

        mock_db = MagicMock()
        mock_db.score_compatibility.return_value = {
            "compatibility_score": 80,
            "total_issues": 0,
            "critical_issues": 0,
            "warning_issues": 0,
            "info_issues": 0,
            "breaking_changes": {"CRITICAL": [], "WARNING": [], "INFO": []},
            "recommendation": "Good",
        }
        mock_db_class.return_value = mock_db

        orchestrator = HealthcheckOrchestrator()
        report = orchestrator.generate_report()

        # Should mention CUDA 13.x in recommendations
        assert any("13." in rec for rec in report.recommendations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
