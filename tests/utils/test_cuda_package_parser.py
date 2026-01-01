"""
Unit tests for CUDA package parser.
"""

from cuda_healthcheck.utils.cuda_package_parser import (
    _extract_major_minor,
    _extract_version,
    _parse_nvidia_package,
    _parse_torch_version,
    check_cuopt_nvjitlink_compatibility,
    check_pytorch_cuda_branch_compatibility,
    format_cuda_packages_report,
    parse_cuda_packages,
)


class TestParseTorchVersion:
    """Test PyTorch version parsing."""

    def test_torch_with_cuda_branch(self):
        """Test parsing torch with CUDA branch."""
        result = _parse_torch_version("torch==2.4.1+cu124")
        assert result == {"version": "2.4.1", "cuda_branch": "cu124"}

    def test_torch_with_cu121(self):
        """Test parsing torch with cu121."""
        result = _parse_torch_version("torch==2.3.0+cu121")
        assert result == {"version": "2.3.0", "cuda_branch": "cu121"}

    def test_torch_cpu_only(self):
        """Test parsing CPU-only torch."""
        result = _parse_torch_version("torch==2.4.1")
        assert result == {"version": "2.4.1", "cuda_branch": None}

    def test_non_torch_package(self):
        """Test non-torch package returns None."""
        result = _parse_torch_version("numpy==1.24.3")
        assert result is None


class TestExtractVersion:
    """Test version extraction."""

    def test_extract_standard_version(self):
        """Test extracting standard version."""
        assert _extract_version("nvidia-cublas-cu12==12.4.5.8") == "12.4.5.8"

    def test_extract_short_version(self):
        """Test extracting short version."""
        assert _extract_version("some-package==1.2.3") == "1.2.3"

    def test_invalid_format(self):
        """Test invalid format returns None."""
        assert _extract_version("invalid-line") is None


class TestExtractMajorMinor:
    """Test major.minor extraction."""

    def test_four_part_version(self):
        """Test extracting from 4-part version."""
        assert _extract_major_minor("12.4.5.8") == "12.4"

    def test_three_part_version(self):
        """Test extracting from 3-part version."""
        assert _extract_major_minor("12.4.127") == "12.4"

    def test_two_part_version(self):
        """Test extracting from 2-part version."""
        assert _extract_major_minor("12.4") == "12.4"

    def test_single_part_version(self):
        """Test single-part version returns None."""
        assert _extract_major_minor("12") is None

    def test_empty_version(self):
        """Test empty version returns None."""
        assert _extract_major_minor("") is None


class TestParseNvidiaPackage:
    """Test NVIDIA package parsing."""

    def test_parse_cuda_runtime(self):
        """Test parsing CUDA runtime package."""
        result = _parse_nvidia_package("nvidia-cuda-runtime-cu12==12.6.77")
        assert result == {"name": "nvidia-cuda-runtime-cu12", "version": "12.6.77"}

    def test_parse_cudnn(self):
        """Test parsing cuDNN package."""
        result = _parse_nvidia_package("nvidia-cudnn-cu12==9.1.0.70")
        assert result == {"name": "nvidia-cudnn-cu12", "version": "9.1.0.70"}

    def test_non_nvidia_package(self):
        """Test non-nvidia package returns None."""
        result = _parse_nvidia_package("torch==2.4.1")
        assert result is None


class TestParseCudaPackages:
    """Test full CUDA packages parsing."""

    def test_databricks_ml_runtime_16_4(self):
        """Test parsing Databricks ML Runtime 16.4 pip freeze."""
        pip_output = """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.3.61
"""
        result = parse_cuda_packages(pip_output)

        assert result["torch"] == "2.4.1"
        assert result["torch_cuda_branch"] == "cu124"
        assert result["cublas"]["version"] == "12.4.5.8"
        assert result["cublas"]["major_minor"] == "12.4"
        assert result["nvjitlink"]["version"] == "12.4.127"
        assert result["nvjitlink"]["major_minor"] == "12.4"
        assert "nvidia-cuda-runtime-cu12" in result["other_nvidia"]
        assert result["other_nvidia"]["nvidia-cuda-runtime-cu12"] == "12.6.77"
        assert "nvidia-cudnn-cu12" in result["other_nvidia"]

    def test_cuopt_environment(self):
        """Test parsing environment with CuOPT."""
        pip_output = """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
cuopt-server-cu12==25.12.0
"""
        result = parse_cuda_packages(pip_output)

        assert result["torch"] == "2.4.1"
        assert result["torch_cuda_branch"] == "cu124"
        assert result["nvjitlink"]["major_minor"] == "12.4"
        assert "cuopt-server-cu12" not in result["other_nvidia"]  # Not nvidia-*

    def test_cpu_only_environment(self):
        """Test parsing CPU-only environment."""
        pip_output = """
torch==2.4.1
numpy==1.24.3
pandas==2.0.3
"""
        result = parse_cuda_packages(pip_output)

        assert result["torch"] == "2.4.1"
        assert result["torch_cuda_branch"] is None
        assert result["cublas"]["version"] is None
        assert result["nvjitlink"]["version"] is None
        assert len(result["other_nvidia"]) == 0

    def test_empty_output(self):
        """Test parsing empty output."""
        result = parse_cuda_packages("")

        assert result["torch"] is None
        assert result["torch_cuda_branch"] is None
        assert result["cublas"]["version"] is None
        assert result["nvjitlink"]["version"] is None

    def test_with_comments(self):
        """Test parsing with comment lines."""
        pip_output = """
# This is a comment
torch==2.4.1+cu124
# Another comment
nvidia-nvjitlink-cu12==12.4.127
"""
        result = parse_cuda_packages(pip_output)

        assert result["torch"] == "2.4.1"
        assert result["nvjitlink"]["version"] == "12.4.127"


class TestFormatCudaPackagesReport:
    """Test report formatting."""

    def test_format_full_report(self):
        """Test formatting full report."""
        packages = {
            "torch": "2.4.1",
            "torch_cuda_branch": "cu124",
            "cublas": {"version": "12.4.5.8", "major_minor": "12.4"},
            "nvjitlink": {"version": "12.4.127", "major_minor": "12.4"},
            "other_nvidia": {
                "nvidia-cuda-runtime-cu12": "12.6.77",
                "nvidia-cudnn-cu12": "9.1.0.70",
            },
        }

        report = format_cuda_packages_report(packages)

        assert "CUDA Packages Report" in report
        assert "PyTorch: 2.4.1 (cu124)" in report
        assert "cuBLAS: 12.4.5.8 (12.4)" in report
        assert "nvJitLink: 12.4.127 (12.4)" in report
        assert "nvidia-cuda-runtime-cu12: 12.6.77" in report

    def test_format_cpu_only(self):
        """Test formatting CPU-only report."""
        packages = {
            "torch": "2.4.1",
            "torch_cuda_branch": None,
            "cublas": {"version": None, "major_minor": None},
            "nvjitlink": {"version": None, "major_minor": None},
            "other_nvidia": {},
        }

        report = format_cuda_packages_report(packages)

        assert "PyTorch: 2.4.1 (CPU-only)" in report
        assert "cuBLAS: Not installed" in report
        assert "nvJitLink: Not installed" in report


class TestCheckCuoptNvjitlinkCompatibility:
    """Test CuOPT nvJitLink compatibility checking."""

    def test_compatible_version(self):
        """Test compatible nvJitLink version."""
        packages = {
            "nvjitlink": {"version": "12.9.79", "major_minor": "12.9"},
        }

        result = check_cuopt_nvjitlink_compatibility(packages)

        assert result["is_compatible"] is True
        assert result["nvjitlink_version"] == "12.9.79"
        assert result["error_message"] is None

    def test_incompatible_version_12_4(self):
        """Test incompatible nvJitLink 12.4 (Databricks)."""
        packages = {
            "nvjitlink": {"version": "12.4.127", "major_minor": "12.4"},
        }

        result = check_cuopt_nvjitlink_compatibility(packages)

        assert result["is_compatible"] is False
        assert result["nvjitlink_version"] == "12.4.127"
        assert "incompatible with CuOPT 25.12+" in result["error_message"]
        assert "PLATFORM CONSTRAINT" in result["error_message"]

    def test_missing_nvjitlink(self):
        """Test missing nvJitLink."""
        packages = {
            "nvjitlink": {"version": None, "major_minor": None},
        }

        result = check_cuopt_nvjitlink_compatibility(packages)

        assert result["is_compatible"] is False
        assert result["error_message"] == "nvJitLink not installed"


class TestCheckPytorchCudaBranchCompatibility:
    """Test PyTorch CUDA branch compatibility checking."""

    def test_compatible_cu124_with_12_4(self):
        """Test compatible cu124 with CUDA 12.4."""
        packages = {
            "torch_cuda_branch": "cu124",
        }

        result = check_pytorch_cuda_branch_compatibility(packages, "12.4")

        assert result["is_compatible"] is True
        assert result["error_message"] is None

    def test_compatible_cu121_with_12_1(self):
        """Test compatible cu121 with CUDA 12.1."""
        packages = {
            "torch_cuda_branch": "cu121",
        }

        result = check_pytorch_cuda_branch_compatibility(packages, "12.1")

        assert result["is_compatible"] is True

    def test_incompatible_cu121_with_12_4(self):
        """Test incompatible cu121 with CUDA 12.4."""
        packages = {
            "torch_cuda_branch": "cu121",
        }

        result = check_pytorch_cuda_branch_compatibility(packages, "12.4")

        assert result["is_compatible"] is False
        assert "does not match expected CUDA 12.4" in result["error_message"]

    def test_missing_cuda_branch(self):
        """Test missing CUDA branch (CPU-only)."""
        packages = {
            "torch_cuda_branch": None,
        }

        result = check_pytorch_cuda_branch_compatibility(packages, "12.4")

        assert result["is_compatible"] is False
        assert "CPU-only" in result["error_message"]


class TestRealWorldScenarios:
    """Test real-world scenarios."""

    def test_databricks_cuopt_incompatibility(self):
        """Test detecting the Databricks CuOPT incompatibility."""
        pip_output = """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.6.77
cuopt-server-cu12==25.12.0
"""
        packages = parse_cuda_packages(pip_output)
        compat = check_cuopt_nvjitlink_compatibility(packages)

        assert not compat["is_compatible"]
        assert "12.4.127" in compat["nvjitlink_version"]
        assert "incompatible" in compat["error_message"].lower()

    def test_successful_environment(self):
        """Test a successful, compatible environment."""
        pip_output = """
torch==2.5.0+cu124
nvidia-cublas-cu12==12.6.3.3
nvidia-nvjitlink-cu12==12.9.79
nvidia-cuda-runtime-cu12==12.6.77
"""
        packages = parse_cuda_packages(pip_output)
        nvjitlink_compat = check_cuopt_nvjitlink_compatibility(packages)
        torch_compat = check_pytorch_cuda_branch_compatibility(packages, "12.4")

        assert nvjitlink_compat["is_compatible"]
        assert torch_compat["is_compatible"]

    def test_edge_case_cu118(self):
        """Test older CUDA branch cu118."""
        pip_output = """
torch==2.0.1+cu118
nvidia-nvjitlink-cu12==11.8.89
"""
        packages = parse_cuda_packages(pip_output)

        assert packages["torch"] == "2.0.1"
        assert packages["torch_cuda_branch"] == "cu118"
        assert packages["nvjitlink"]["version"] == "11.8.89"
        assert packages["nvjitlink"]["major_minor"] == "11.8"
