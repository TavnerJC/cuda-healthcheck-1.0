"""
Unit tests for Databricks runtime detection.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from cuda_healthcheck.databricks.runtime_detector import (
    _create_result,
    _detect_from_env_var,
    _detect_from_environment_file,
    _detect_from_ipython,
    _detect_from_workspace_indicator,
    _get_cuda_version_for_runtime,
    _parse_runtime_string,
    detect_databricks_runtime,
    get_runtime_info_summary,
    is_databricks_environment,
)


class TestParseRuntimeString:
    """Tests for _parse_runtime_string function."""

    def test_parse_ml_runtime_14_3(self):
        """Test parsing ML Runtime 14.3."""
        result = _parse_runtime_string("14.3.x-gpu-ml-scala2.12")

        assert result["runtime_version"] == 14.3
        assert result["is_ml_runtime"] is True
        assert result["is_gpu_runtime"] is True
        assert result["is_serverless"] is False
        assert result["cuda_version"] == "12.2"

    def test_parse_ml_runtime_15_2(self):
        """Test parsing ML Runtime 15.2."""
        result = _parse_runtime_string("15.2.x-gpu-ml-scala2.12")

        assert result["runtime_version"] == 15.2
        assert result["is_ml_runtime"] is True
        assert result["is_gpu_runtime"] is True
        assert result["is_serverless"] is False
        assert result["cuda_version"] == "12.4"

    def test_parse_ml_runtime_16_4(self):
        """Test parsing ML Runtime 16.4."""
        result = _parse_runtime_string("16.4.x-gpu-ml-scala2.12")

        assert result["runtime_version"] == 16.4
        assert result["is_ml_runtime"] is True
        assert result["is_gpu_runtime"] is True
        assert result["is_serverless"] is False
        assert result["cuda_version"] == "12.6"

    def test_parse_cpu_runtime(self):
        """Test parsing CPU-only runtime."""
        result = _parse_runtime_string("15.2.x-scala2.12")

        assert result["runtime_version"] == 15.2
        assert result["is_ml_runtime"] is False
        assert result["is_gpu_runtime"] is False
        assert result["is_serverless"] is False

    def test_parse_serverless_v4(self):
        """Test parsing Serverless GPU Compute v4."""
        result = _parse_runtime_string("serverless-gpu-v4")

        assert result["runtime_version"] is None
        assert result["is_serverless"] is True
        assert result["is_gpu_runtime"] is True
        assert result["cuda_version"] == "12.6"

    def test_parse_serverless_v3(self):
        """Test parsing Serverless GPU Compute v3."""
        result = _parse_runtime_string("serverless-gpu-v3")

        assert result["runtime_version"] is None
        assert result["is_serverless"] is True
        assert result["is_gpu_runtime"] is True
        assert result["cuda_version"] == "12.4"

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = _parse_runtime_string("")

        assert result["runtime_version"] is None
        assert result["is_ml_runtime"] is False
        assert result["is_gpu_runtime"] is False
        assert result["is_serverless"] is False


class TestGetCudaVersionForRuntime:
    """Tests for _get_cuda_version_for_runtime function."""

    def test_runtime_14_3(self):
        """Test CUDA version for Runtime 14.3."""
        assert _get_cuda_version_for_runtime(14.3) == "12.2"

    def test_runtime_15_2(self):
        """Test CUDA version for Runtime 15.2."""
        assert _get_cuda_version_for_runtime(15.2) == "12.4"

    def test_runtime_16_4(self):
        """Test CUDA version for Runtime 16.4."""
        assert _get_cuda_version_for_runtime(16.4) == "12.6"

    def test_runtime_13_3(self):
        """Test CUDA version for Runtime 13.3."""
        assert _get_cuda_version_for_runtime(13.3) == "11.8"

    def test_unknown_runtime(self):
        """Test unknown runtime version."""
        # Should return None for unknown versions
        result = _get_cuda_version_for_runtime(99.9)
        assert result is None

    def test_none_runtime(self):
        """Test None runtime version."""
        assert _get_cuda_version_for_runtime(None) is None


class TestDetectFromEnvVar:
    """Tests for _detect_from_env_var function."""

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3.x-gpu-ml-scala2.12"})
    def test_detect_ml_runtime_14_3(self):
        """Test detection from env var - ML Runtime 14.3."""
        result = _detect_from_env_var()

        assert result["is_databricks"] is True
        assert result["runtime_version"] == 14.3
        assert result["runtime_version_string"] == "14.3.x-gpu-ml-scala2.12"
        assert result["is_ml_runtime"] is True
        assert result["is_gpu_runtime"] is True
        assert result["cuda_version"] == "12.2"
        assert result["detection_method"] == "env_var"

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "15.2.x-gpu-ml-scala2.12"})
    def test_detect_ml_runtime_15_2(self):
        """Test detection from env var - ML Runtime 15.2."""
        result = _detect_from_env_var()

        assert result["is_databricks"] is True
        assert result["runtime_version"] == 15.2
        assert result["cuda_version"] == "12.4"

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "16.4.x-gpu-ml-scala2.12"})
    def test_detect_ml_runtime_16_4(self):
        """Test detection from env var - ML Runtime 16.4."""
        result = _detect_from_env_var()

        assert result["is_databricks"] is True
        assert result["runtime_version"] == 16.4
        assert result["cuda_version"] == "12.6"

    @patch.dict(os.environ, {}, clear=True)
    def test_no_env_var(self):
        """Test when DATABRICKS_RUNTIME_VERSION is not set."""
        result = _detect_from_env_var()

        assert result["is_databricks"] is False
        assert result["detection_method"] == "env_var"


class TestDetectFromEnvironmentFile:
    """Tests for _detect_from_environment_file function."""

    def test_detect_from_yaml_file(self, tmp_path):
        """Test detection from environment.yml file."""
        # Create temporary environment.yml
        env_file = tmp_path / "environment.yml"
        env_data = {
            "databricks": {"runtime_version": "14.3.x-gpu-ml-scala2.12"},
        }

        with open(env_file, "w") as f:
            yaml.dump(env_data, f)

        with patch("cuda_healthcheck.databricks.runtime_detector.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", mock_open(read_data=yaml.dump(env_data))):
                result = _detect_from_environment_file()

                assert result["is_databricks"] is True
                assert result["runtime_version"] == 14.3
                assert result["detection_method"] == "file"

    def test_file_not_exists(self):
        """Test when environment.yml doesn't exist."""
        with patch("cuda_healthcheck.databricks.runtime_detector.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = _detect_from_environment_file()

            assert result["is_databricks"] is False
            assert result["detection_method"] == "file"

    def test_file_parse_error(self):
        """Test handling of YAML parse errors."""
        with patch("cuda_healthcheck.databricks.runtime_detector.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", mock_open(read_data="invalid: yaml: content:")):
                result = _detect_from_environment_file()

                # Should detect Databricks but not parse version
                assert result["is_databricks"] is True
                assert result["runtime_version"] is None


class TestDetectFromWorkspaceIndicator:
    """Tests for _detect_from_workspace_indicator function."""

    def test_workspace_exists(self):
        """Test when /Workspace directory exists."""
        with patch("cuda_healthcheck.databricks.runtime_detector.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            result = _detect_from_workspace_indicator()

            assert result["is_databricks"] is True
            assert result["detection_method"] == "workspace"

    def test_workspace_not_exists(self):
        """Test when /Workspace directory doesn't exist."""
        with patch("cuda_healthcheck.databricks.runtime_detector.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = _detect_from_workspace_indicator()

            assert result["is_databricks"] is False


class TestDetectFromIPython:
    """Tests for _detect_from_ipython function."""

    def test_ipython_databricks_config(self):
        """Test detection from IPython config."""
        mock_ipython = MagicMock()
        mock_ipython.config = {"DATABRICKS": True}

        # Mock IPython module
        mock_ipython_module = MagicMock()
        mock_ipython_module.get_ipython.return_value = mock_ipython

        with patch.dict(sys.modules, {"IPython": mock_ipython_module}):
            result = _detect_from_ipython()

            assert result["is_databricks"] is True
            assert result["detection_method"] == "ipython"

    def test_ipython_not_available(self):
        """Test when IPython is not available."""
        # Temporarily remove IPython from sys.modules
        original_ipython = sys.modules.get("IPython")

        try:
            if "IPython" in sys.modules:
                del sys.modules["IPython"]

            with patch.dict(sys.modules, {"IPython": None}):
                # This will cause ImportError in _detect_from_ipython
                result = _detect_from_ipython()

                assert result["is_databricks"] is False
        finally:
            # Restore original state
            if original_ipython is not None:
                sys.modules["IPython"] = original_ipython


class TestDetectDatabricksRuntime:
    """Tests for detect_databricks_runtime function (main function)."""

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3.x-gpu-ml-scala2.12"})
    def test_full_detection_ml_runtime_14_3(self):
        """Test full detection with ML Runtime 14.3."""
        result = detect_databricks_runtime()

        assert result["is_databricks"] is True
        assert result["runtime_version"] == 14.3
        assert result["is_ml_runtime"] is True
        assert result["is_gpu_runtime"] is True
        assert result["cuda_version"] == "12.2"
        assert result["detection_method"] == "env_var"

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "15.2.x-gpu-ml-scala2.12"})
    def test_full_detection_ml_runtime_15_2(self):
        """Test full detection with ML Runtime 15.2."""
        result = detect_databricks_runtime()

        assert result["is_databricks"] is True
        assert result["runtime_version"] == 15.2
        assert result["cuda_version"] == "12.4"

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "16.4.x-gpu-ml-scala2.12"})
    def test_full_detection_ml_runtime_16_4(self):
        """Test full detection with ML Runtime 16.4."""
        result = detect_databricks_runtime()

        assert result["is_databricks"] is True
        assert result["runtime_version"] == 16.4
        assert result["cuda_version"] == "12.6"

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "serverless-gpu-v4"})
    def test_full_detection_serverless(self):
        """Test full detection with Serverless GPU Compute."""
        result = detect_databricks_runtime()

        assert result["is_databricks"] is True
        assert result["is_serverless"] is True
        assert result["cuda_version"] == "12.6"

    @patch.dict(os.environ, {}, clear=True)
    def test_full_detection_no_databricks(self):
        """Test detection when not in Databricks."""
        with patch("cuda_healthcheck.databricks.runtime_detector.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            result = detect_databricks_runtime()

            assert result["is_databricks"] is False
            assert result["runtime_version"] is None
            assert result["detection_method"] == "unknown"


class TestGetRuntimeInfoSummary:
    """Tests for get_runtime_info_summary function."""

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3.x-gpu-ml-scala2.12"})
    def test_summary_ml_runtime(self):
        """Test summary for ML Runtime."""
        summary = get_runtime_info_summary()

        assert "Databricks ML Runtime 14.3" in summary
        assert "GPU" in summary
        assert "CUDA 12.2" in summary
        assert "env_var" in summary

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "serverless-gpu-v4"})
    def test_summary_serverless(self):
        """Test summary for Serverless GPU Compute."""
        summary = get_runtime_info_summary()

        assert "Serverless GPU Compute" in summary

    @patch.dict(os.environ, {}, clear=True)
    def test_summary_no_databricks(self):
        """Test summary when not in Databricks."""
        with patch("cuda_healthcheck.databricks.runtime_detector.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            summary = get_runtime_info_summary()

            assert "Not running in Databricks" in summary


class TestIsDatabricksEnvironment:
    """Tests for is_databricks_environment convenience function."""

    @patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3.x-gpu-ml-scala2.12"})
    def test_is_databricks_true(self):
        """Test when in Databricks environment."""
        assert is_databricks_environment() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_databricks_false(self):
        """Test when not in Databricks environment."""
        with patch("cuda_healthcheck.databricks.runtime_detector.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            assert is_databricks_environment() is False


class TestCreateResult:
    """Tests for _create_result helper function."""

    def test_create_full_result(self):
        """Test creating a full result dictionary."""
        result = _create_result(
            is_databricks=True,
            runtime_version=14.3,
            runtime_version_string="14.3.x-gpu-ml-scala2.12",
            is_ml_runtime=True,
            is_gpu_runtime=True,
            is_serverless=False,
            cuda_version="12.2",
            detection_method="env_var",
        )

        assert result["is_databricks"] is True
        assert result["runtime_version"] == 14.3
        assert result["runtime_version_string"] == "14.3.x-gpu-ml-scala2.12"
        assert result["is_ml_runtime"] is True
        assert result["is_gpu_runtime"] is True
        assert result["is_serverless"] is False
        assert result["cuda_version"] == "12.2"
        assert result["detection_method"] == "env_var"

    def test_create_minimal_result(self):
        """Test creating a minimal result dictionary."""
        result = _create_result(is_databricks=False)

        assert result["is_databricks"] is False
        assert result["runtime_version"] is None
        assert result["runtime_version_string"] is None
        assert result["is_ml_runtime"] is False
        assert result["is_gpu_runtime"] is False
        assert result["is_serverless"] is False
        assert result["cuda_version"] is None
        assert result["detection_method"] == "unknown"
