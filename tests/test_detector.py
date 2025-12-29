"""Unit tests for CUDA detector."""

from unittest.mock import Mock, mock_open, patch

import pytest

from src.cuda_detector.detector import CUDADetector, detect_cuda_environment


class TestCUDADetector:
    """Test suite for CUDADetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = CUDADetector()

    @patch("subprocess.run")
    def test_detect_nvidia_smi_success(self, mock_run):
        """Test successful nvidia-smi detection."""
        # Mock nvidia-smi output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="535.104.05, Tesla V100-SXM2-16GB, 16384, 7.0, 0\n",
            stderr="",
        )

        result = self.detector.detect_nvidia_smi()

        assert result["success"] is True
        assert len(result["gpus"]) == 1
        assert result["gpus"][0].name == "Tesla V100-SXM2-16GB"
        assert result["gpus"][0].compute_capability == "7.0"

    @patch("subprocess.run")
    def test_detect_nvidia_smi_not_found(self, mock_run):
        """Test nvidia-smi not found."""
        mock_run.side_effect = FileNotFoundError()

        result = self.detector.detect_nvidia_smi()

        assert result["success"] is False
        assert "nvidia-smi" in result["error"] and "not found" in result["error"]

    @patch("src.cuda_detector.detector.check_command_available")
    @patch("subprocess.run")
    def test_detect_nvcc_version(self, mock_run, mock_check_command):
        """Test nvcc version detection."""
        mock_check_command.return_value = True
        mock_run.return_value = Mock(
            returncode=0, stdout="Cuda compilation tools, release 12.4, V12.4.131"
        )

        version = self.detector.detect_nvcc_version()

        assert version == "12.4"

    @patch("subprocess.run")
    def test_detect_nvcc_not_found(self, mock_run):
        """Test nvcc not found."""
        mock_run.side_effect = FileNotFoundError()

        version = self.detector.detect_nvcc_version()

        assert version is None

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"cuda": {"version": "12.4.0"}}',
    )
    @patch("pathlib.Path.exists")
    def test_detect_cuda_runtime_json(self, mock_exists, mock_file):
        """Test CUDA runtime detection from version.json."""
        mock_exists.return_value = True

        version = self.detector.detect_cuda_runtime()

        assert version == "12.4.0"

    def test_detect_pytorch_not_installed(self):
        """Test PyTorch detection when not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            lib_info = self.detector.detect_pytorch()

            assert lib_info.name == "pytorch"
            assert lib_info.version == "Not installed"
            assert lib_info.is_compatible is False

    def test_detect_tensorflow_not_installed(self):
        """Test TensorFlow detection when not installed."""
        with patch.dict("sys.modules", {"tensorflow": None}):
            lib_info = self.detector.detect_tensorflow()

            assert lib_info.name == "tensorflow"
            assert lib_info.version == "Not installed"
            assert lib_info.is_compatible is False

    def test_detect_cudf_not_installed(self):
        """Test cuDF detection when not installed."""
        with patch.dict("sys.modules", {"cudf": None}):
            lib_info = self.detector.detect_cudf()

            assert lib_info.name == "cudf"
            assert lib_info.version == "Not installed"
            assert lib_info.is_compatible is False

    @patch.object(CUDADetector, "detect_nvidia_smi")
    @patch.object(CUDADetector, "detect_cuda_runtime")
    @patch.object(CUDADetector, "detect_nvcc_version")
    @patch.object(CUDADetector, "detect_all_libraries")
    def test_detect_environment(self, mock_libs, mock_nvcc, mock_runtime, mock_nvidia):
        """Test complete environment detection."""
        # Set up mocks
        mock_nvidia.return_value = {"success": True, "gpus": [], "cuda_version": "12.4"}
        mock_runtime.return_value = "12.4.0"
        mock_nvcc.return_value = "12.4"
        mock_libs.return_value = []

        env = self.detector.detect_environment()

        assert env.cuda_runtime_version == "12.4.0"
        assert env.cuda_driver_version == "12.4"
        assert env.nvcc_version == "12.4"
        assert isinstance(env.gpus, list)
        assert isinstance(env.libraries, list)


def test_detect_cuda_environment():
    """Test convenience function."""
    with patch("src.cuda_detector.detector.CUDADetector") as mock_detector:
        mock_instance = Mock()
        mock_detector.return_value = mock_instance

        # Call detect_cuda_environment
        detect_cuda_environment()

        mock_instance.detect_environment.assert_called_once()
        mock_instance.to_dict.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
