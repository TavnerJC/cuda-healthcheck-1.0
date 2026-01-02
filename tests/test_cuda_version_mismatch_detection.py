"""
Comprehensive unit tests for CUDA version mismatch detection.

Tests cover nvJitLink mismatches, mixed CUDA versions, driver incompatibilities,
and feature-based requirements validation.
"""

from unittest.mock import MagicMock, patch

import pytest

from cuda_healthcheck.utils import (
    check_cublas_nvjitlink_version_match,
    detect_mixed_cuda_versions,
    parse_cuda_packages,
    validate_cuda_library_versions,
    validate_torch_branch_compatibility,
)

# ============================================================================
# FIXTURES - Mock Data
# ============================================================================


@pytest.fixture
def pip_freeze_cublas_12_1_nvjitlink_12_4():
    """Mismatch: cuBLAS 12.1.x but nvJitLink 12.4.x."""
    return """
torch==2.4.1+cu121
nvidia-cublas-cu12==12.1.3.1
nvidia-nvjitlink-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.1.105
numpy==1.26.4
"""


@pytest.fixture
def pip_freeze_cublas_12_4_nvjitlink_12_1():
    """Mismatch: cuBLAS 12.4.x but nvJitLink 12.1.x."""
    return """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.4.127
numpy==1.26.4
"""


@pytest.fixture
def pip_freeze_missing_nvjitlink():
    """cuBLAS present but nvJitLink missing."""
    return """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-runtime-cu12==12.4.127
numpy==1.26.4
"""


@pytest.fixture
def pip_freeze_mixed_cu11_cu12():
    """Both CUDA 11 and CUDA 12 packages present."""
    return """
torch==2.4.1+cu124
nvidia-cublas-cu11==11.10.3.66
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
cupy-cuda11x==13.0.0
numpy==1.26.4
"""


@pytest.fixture
def pip_freeze_valid_cu124():
    """Valid configuration: matching cuBLAS/nvJitLink cu124."""
    return """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
numpy==1.26.4
"""


@pytest.fixture
def pip_freeze_valid_cu120():
    """Valid configuration: matching cuBLAS/nvJitLink cu120."""
    return """
torch==2.4.1+cu120
nvidia-cublas-cu12==12.1.3.1
nvidia-nvjitlink-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
numpy==1.26.4
"""


@pytest.fixture
def pip_freeze_valid_cu121():
    """Valid configuration: matching cuBLAS/nvJitLink cu121."""
    return """
torch==2.4.1+cu121
nvidia-cublas-cu12==12.1.3.1
nvidia-nvjitlink-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
numpy==1.26.4
"""


@pytest.fixture
def pip_freeze_only_cu11():
    """Only CUDA 11 packages, no CUDA 12."""
    return """
torch==2.1.0+cu118
nvidia-cublas-cu11==11.10.3.66
cupy-cuda11x==13.0.0
numpy==1.26.4
"""


# ============================================================================
# TEST CASE 1: nvJitLink Version Mismatch
# ============================================================================


class TestNvJitLinkVersionMismatch:
    """Test detection of cuBLAS/nvJitLink major.minor version mismatches."""

    def test_cublas_12_1_nvjitlink_12_4_mismatch(self):
        """Test mismatch: cuBLAS 12.1.x but nvJitLink 12.4.x."""
        result = check_cublas_nvjitlink_version_match("12.1.3.1", "12.4.127")

        assert result["is_mismatch"] is True
        assert result["severity"] == "BLOCKER"
        assert "12.1" in result["error_message"]
        assert "12.4" in result["error_message"]
        assert "pip install" in result["fix_command"]
        assert "nvidia-nvjitlink-cu12==12.1.*" in result["fix_command"]

    def test_cublas_12_4_nvjitlink_12_1_mismatch(self):
        """Test mismatch: cuBLAS 12.4.x but nvJitLink 12.1.x."""
        result = check_cublas_nvjitlink_version_match("12.4.5.8", "12.1.105")

        assert result["is_mismatch"] is True
        assert result["severity"] == "BLOCKER"
        assert "12.4" in result["error_message"]
        assert "12.1" in result["error_message"]
        assert "pip install" in result["fix_command"]
        assert "nvidia-nvjitlink-cu12==12.4.*" in result["fix_command"]

    def test_matching_versions_12_1(self):
        """Test matching versions: both 12.1.x."""
        result = check_cublas_nvjitlink_version_match("12.1.3.1", "12.1.105")

        assert result["is_mismatch"] is False
        assert result["severity"] == "OK"
        assert result["error_message"] is None
        assert result["fix_command"] is None

    def test_matching_versions_12_4(self):
        """Test matching versions: both 12.4.x."""
        result = check_cublas_nvjitlink_version_match("12.4.5.8", "12.4.127")

        assert result["is_mismatch"] is False
        assert result["severity"] == "OK"
        assert result["error_message"] is None
        assert result["fix_command"] is None

    def test_integrated_mismatch_detection(self, pip_freeze_cublas_12_1_nvjitlink_12_4):
        """Test integrated detection from pip freeze output."""
        packages = parse_cuda_packages(pip_freeze_cublas_12_1_nvjitlink_12_4)
        cublas_version = packages["cublas"]["version"]
        nvjitlink_version = packages["nvjitlink"]["version"]

        result = check_cublas_nvjitlink_version_match(cublas_version, nvjitlink_version)

        assert result["is_mismatch"] is True
        assert result["severity"] == "BLOCKER"


# ============================================================================
# TEST CASE 2: Missing nvJitLink
# ============================================================================


class TestMissingNvJitLink:
    """Test detection when cuBLAS is present but nvJitLink is missing."""

    def test_missing_nvjitlink_with_cublas(self, pip_freeze_missing_nvjitlink):
        """Test detection of missing nvJitLink when cuBLAS is present."""
        packages = parse_cuda_packages(pip_freeze_missing_nvjitlink)

        assert packages["cublas"]["version"] is not None
        assert packages["nvjitlink"]["version"] is None

    def test_missing_nvjitlink_blocker(self):
        """Test that missing nvJitLink is flagged as BLOCKER."""
        # Simulate missing nvJitLink
        result = check_cublas_nvjitlink_version_match("12.4.5.8", None)

        # Should handle None gracefully
        assert result["is_mismatch"] is True
        assert result["severity"] == "BLOCKER"

    def test_comprehensive_validation_missing_nvjitlink(self, pip_freeze_missing_nvjitlink):
        """Test comprehensive validation catches missing nvJitLink."""
        packages = parse_cuda_packages(pip_freeze_missing_nvjitlink)
        validation = validate_cuda_library_versions(packages)

        # Should have blocker for missing nvJitLink
        blockers = validation["blockers"]
        assert len(blockers) > 0

        # Check if any blocker mentions nvJitLink
        nvjitlink_blocker = any("nvjitlink" in str(b).lower() for b in blockers)
        assert nvjitlink_blocker


# ============================================================================
# TEST CASE 3: Mixed cu11/cu12
# ============================================================================


class TestMixedCu11Cu12:
    """Test detection of mixed CUDA 11 and CUDA 12 packages."""

    def test_mixed_versions_detected(self, pip_freeze_mixed_cu11_cu12):
        """Test that mixed CUDA 11/12 packages are detected."""
        result = detect_mixed_cuda_versions(pip_freeze_mixed_cu11_cu12)

        assert result["has_cu11"] is True
        assert result["has_cu12"] is True
        assert result["severity"] == "BLOCKER"
        assert len(result["cu11_packages"]) > 0
        assert len(result["cu12_packages"]) > 0

    def test_mixed_versions_blocker_message(self, pip_freeze_mixed_cu11_cu12):
        """Test that blocker message is clear."""
        result = detect_mixed_cuda_versions(pip_freeze_mixed_cu11_cu12)

        assert "conflict" in result["error_message"].lower()
        assert "CUDA 11" in result["error_message"]
        assert "CUDA 12" in result["error_message"]

    def test_mixed_versions_fix_command(self, pip_freeze_mixed_cu11_cu12):
        """Test that fix command includes clean reinstall steps."""
        result = detect_mixed_cuda_versions(pip_freeze_mixed_cu11_cu12)

        assert "pip uninstall" in result["fix_command"]
        assert "pip cache purge" in result["fix_command"]
        assert "pip install" in result["fix_command"]

    def test_only_cu12_no_blocker(self, pip_freeze_valid_cu124):
        """Test that only CUDA 12 packages don't trigger blocker."""
        result = detect_mixed_cuda_versions(pip_freeze_valid_cu124)

        assert result["has_cu11"] is False
        assert result["has_cu12"] is True
        assert result["severity"] is None

    def test_only_cu11_no_blocker(self, pip_freeze_only_cu11):
        """Test that only CUDA 11 packages don't trigger blocker."""
        result = detect_mixed_cuda_versions(pip_freeze_only_cu11)

        assert result["has_cu11"] is True
        assert result["has_cu12"] is False
        assert result["severity"] is None

    def test_mixed_cu11_cu12_package_lists(self, pip_freeze_mixed_cu11_cu12):
        """Test that package lists are correctly populated."""
        result = detect_mixed_cuda_versions(pip_freeze_mixed_cu11_cu12)

        # Check cu11 packages
        cu11_packages = result["cu11_packages"]
        assert any("cublas-cu11" in pkg for pkg in cu11_packages)
        assert any("cupy-cuda11x" in pkg for pkg in cu11_packages)

        # Check cu12 packages
        cu12_packages = result["cu12_packages"]
        # torch+cu124 might be captured in package names or in whole lines
        assert any("cu124" in pkg or "cu12" in pkg for pkg in cu12_packages)
        assert any("cublas-cu12" in pkg for pkg in cu12_packages)


# ============================================================================
# TEST CASE 4: Driver Incompatibility
# ============================================================================


class TestDriverIncompatibility:
    """Test detection of driver/PyTorch CUDA branch incompatibilities."""

    def test_runtime_14_3_cu124_blocker(self):
        """Test Runtime 14.3 + cu124 is a BLOCKER."""
        result = validate_torch_branch_compatibility(
            runtime_version=14.3, torch_cuda_branch="cu124"
        )

        assert result["is_compatible"] is False
        assert result["severity"] == "BLOCKER"
        assert "14.3" in result["issue"]
        assert "cu124" in result["issue"]

    def test_runtime_14_3_cu124_fix_options(self):
        """Test that Runtime 14.3 + cu124 provides two fix options."""
        result = validate_torch_branch_compatibility(
            runtime_version=14.3, torch_cuda_branch="cu124"
        )

        assert len(result["fix_options"]) == 2
        # Option 1: Downgrade to cu120
        assert any("cu120" in opt for opt in result["fix_options"])
        # Option 2: Upgrade runtime
        assert any("15.2" in opt or "runtime" in opt.lower() for opt in result["fix_options"])

    def test_runtime_14_3_cu120_compatible(self):
        """Test Runtime 14.3 + cu120 is compatible."""
        result = validate_torch_branch_compatibility(
            runtime_version=14.3, torch_cuda_branch="cu120"
        )

        assert result["is_compatible"] is True
        assert result["severity"] is None

    def test_runtime_15_2_cu124_compatible(self):
        """Test Runtime 15.2 + cu124 is compatible."""
        result = validate_torch_branch_compatibility(
            runtime_version=15.2, torch_cuda_branch="cu124"
        )

        assert result["is_compatible"] is True
        assert result["severity"] is None

    def test_runtime_15_1_cu124_compatible(self):
        """Test Runtime 15.1 + cu124 is compatible."""
        result = validate_torch_branch_compatibility(
            runtime_version=15.1, torch_cuda_branch="cu124"
        )

        assert result["is_compatible"] is True
        assert result["severity"] is None

    def test_runtime_16_4_cu124_compatible(self):
        """Test Runtime 16.4 + cu124 is compatible."""
        result = validate_torch_branch_compatibility(
            runtime_version=16.4, torch_cuda_branch="cu124"
        )

        assert result["is_compatible"] is True
        assert result["severity"] is None

    def test_unknown_runtime_no_validation(self):
        """Test that unknown runtime doesn't cause errors."""
        result = validate_torch_branch_compatibility(
            runtime_version=13.0, torch_cuda_branch="cu124"
        )

        # Should handle gracefully
        assert "is_compatible" in result


# ============================================================================
# TEST CASE 5: Valid Configuration
# ============================================================================


class TestValidConfiguration:
    """Test that valid configurations pass all checks."""

    def test_runtime_15_2_cu124_all_checks_pass(self, pip_freeze_valid_cu124):
        """Test Runtime 15.2 + cu124 with matching libraries passes all checks."""
        # Parse packages
        packages = parse_cuda_packages(pip_freeze_valid_cu124)

        # Check cuBLAS/nvJitLink match
        cublas_nvjitlink = check_cublas_nvjitlink_version_match(
            packages["cublas"]["version"], packages["nvjitlink"]["version"]
        )
        assert cublas_nvjitlink["is_mismatch"] is False
        assert cublas_nvjitlink["severity"] == "OK"

        # Check no mixed CUDA versions
        mixed_cuda = detect_mixed_cuda_versions(pip_freeze_valid_cu124)
        assert mixed_cuda["severity"] is None

        # Check PyTorch branch compatibility
        torch_compat = validate_torch_branch_compatibility(
            runtime_version=15.2, torch_cuda_branch="cu124"
        )
        assert torch_compat["is_compatible"] is True
        assert torch_compat["severity"] is None

    def test_runtime_14_3_cu120_all_checks_pass(self, pip_freeze_valid_cu120):
        """Test Runtime 14.3 + cu120 with matching libraries passes all checks."""
        packages = parse_cuda_packages(pip_freeze_valid_cu120)

        # Check cuBLAS/nvJitLink match
        cublas_nvjitlink = check_cublas_nvjitlink_version_match(
            packages["cublas"]["version"], packages["nvjitlink"]["version"]
        )
        assert cublas_nvjitlink["is_mismatch"] is False

        # Check no mixed CUDA versions
        mixed_cuda = detect_mixed_cuda_versions(pip_freeze_valid_cu120)
        assert mixed_cuda["severity"] is None

        # Check PyTorch branch compatibility
        torch_compat = validate_torch_branch_compatibility(
            runtime_version=14.3, torch_cuda_branch="cu120"
        )
        assert torch_compat["is_compatible"] is True

    def test_comprehensive_validation_all_pass(self, pip_freeze_valid_cu124):
        """Test comprehensive validation passes with valid config."""
        packages = parse_cuda_packages(pip_freeze_valid_cu124)
        validation = validate_cuda_library_versions(packages)

        # Should have no blockers
        blockers = validation["blockers"]
        assert len(blockers) == 0

        # Overall status should be OK
        assert validation["all_compatible"] is True


# ============================================================================
# TEST CASE 6: Feature-Based Requirements
# ============================================================================


class TestFeatureBasedRequirements:
    """Test validation based on enabled DataDesigner features."""

    @pytest.fixture
    def mock_features_local_inference_enabled(self):
        """Mock features with local_llm_inference enabled."""
        from cuda_healthcheck.nemo.datadesigner_detector import (
            DataDesignerFeature,
            FeatureRequirements,
        )

        return {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                detection_method="environment",
                requirements=FeatureRequirements(
                    feature_name="local_llm_inference",
                    requires_torch=True,
                    requires_cuda=True,
                    compatible_cuda_branches=["cu120", "cu121", "cu124"],
                    min_gpu_memory_gb=40.0,
                    description="Local LLM inference",
                ),
                validation_status="PENDING",
                validation_message="",
            )
        }

    @pytest.fixture
    def mock_features_cloud_inference_only(self):
        """Mock features with only cloud_llm_inference enabled."""
        from cuda_healthcheck.nemo.datadesigner_detector import (
            DataDesignerFeature,
            FeatureRequirements,
        )

        return {
            "cloud_llm_inference": DataDesignerFeature(
                feature_name="cloud_llm_inference",
                is_enabled=True,
                detection_method="environment",
                requirements=FeatureRequirements(
                    feature_name="cloud_llm_inference",
                    requires_torch=False,
                    requires_cuda=False,
                    compatible_cuda_branches=[],
                    min_gpu_memory_gb=None,
                    description="Cloud API inference",
                ),
                validation_status="PENDING",
                validation_message="",
            )
        }

    def test_local_inference_cu120_valid(
        self, mock_features_local_inference_enabled, pip_freeze_valid_cu120
    ):
        """Test local_llm_inference with cu120 is valid."""
        packages = parse_cuda_packages(pip_freeze_valid_cu120)

        # Check that cu120 is in compatible branches
        requirements = mock_features_local_inference_enabled["local_llm_inference"].requirements
        assert "cu120" in requirements.compatible_cuda_branches

        # Validate torch branch compatibility
        torch_compat = validate_torch_branch_compatibility(
            runtime_version=14.3, torch_cuda_branch=packages["torch_cuda_branch"]
        )
        assert torch_compat["is_compatible"] is True

    def test_local_inference_cu121_valid(
        self, mock_features_local_inference_enabled, pip_freeze_valid_cu121
    ):
        """Test local_llm_inference with cu121 is valid."""
        packages = parse_cuda_packages(pip_freeze_valid_cu121)

        requirements = mock_features_local_inference_enabled["local_llm_inference"].requirements
        assert "cu121" in requirements.compatible_cuda_branches

        # cu121 is compatible with Runtime 14.3
        torch_compat = validate_torch_branch_compatibility(
            runtime_version=14.3, torch_cuda_branch=packages["torch_cuda_branch"]
        )
        assert torch_compat["is_compatible"] is True

    def test_local_inference_cu124_valid_runtime_15_2(
        self, mock_features_local_inference_enabled, pip_freeze_valid_cu124
    ):
        """Test local_llm_inference with cu124 on Runtime 15.2 is valid."""
        packages = parse_cuda_packages(pip_freeze_valid_cu124)

        requirements = mock_features_local_inference_enabled["local_llm_inference"].requirements
        assert "cu124" in requirements.compatible_cuda_branches

        # cu124 is compatible with Runtime 15.2
        torch_compat = validate_torch_branch_compatibility(
            runtime_version=15.2, torch_cuda_branch=packages["torch_cuda_branch"]
        )
        assert torch_compat["is_compatible"] is True

    def test_local_inference_cu124_invalid_runtime_14_3(
        self, mock_features_local_inference_enabled, pip_freeze_valid_cu124
    ):
        """Test local_llm_inference with cu124 on Runtime 14.3 is BLOCKER."""
        packages = parse_cuda_packages(pip_freeze_valid_cu124)

        # cu124 is NOT compatible with Runtime 14.3
        torch_compat = validate_torch_branch_compatibility(
            runtime_version=14.3, torch_cuda_branch=packages["torch_cuda_branch"]
        )
        assert torch_compat["is_compatible"] is False
        assert torch_compat["severity"] == "BLOCKER"

    def test_cloud_inference_no_cuda_required(self, mock_features_cloud_inference_only):
        """Test cloud_llm_inference doesn't require CUDA."""
        requirements = mock_features_cloud_inference_only["cloud_llm_inference"].requirements
        assert requirements.requires_torch is False
        assert requirements.requires_cuda is False
        assert len(requirements.compatible_cuda_branches) == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegratedValidation:
    """Test end-to-end validation scenarios."""

    def test_scenario_runtime_14_3_cu124_mismatch(self, pip_freeze_valid_cu124):
        """
        Scenario: Runtime 14.3 + cu124 (BLOCKER scenario).
        Expected: PyTorch branch incompatibility blocker.
        """
        packages = parse_cuda_packages(pip_freeze_valid_cu124)

        # Test PyTorch branch compatibility separately
        torch_compat = validate_torch_branch_compatibility(
            runtime_version=14.3, torch_cuda_branch=packages["torch_cuda_branch"]
        )

        # Should be incompatible
        assert torch_compat["is_compatible"] is False
        assert torch_compat["severity"] == "BLOCKER"
        assert "14.3" in torch_compat["issue"]
        assert "cu124" in torch_compat["issue"]

    def test_scenario_mixed_cuda_and_mismatch(self, pip_freeze_mixed_cu11_cu12):
        """
        Scenario: Mixed cu11/cu12 (BLOCKER scenario).
        Expected: Mixed CUDA versions blocker.
        """
        mixed_result = detect_mixed_cuda_versions(pip_freeze_mixed_cu11_cu12)

        assert mixed_result["severity"] == "BLOCKER"
        assert mixed_result["has_cu11"] is True
        assert mixed_result["has_cu12"] is True

    def test_scenario_missing_nvjitlink(self, pip_freeze_missing_nvjitlink):
        """
        Scenario: cuBLAS present but nvJitLink missing (BLOCKER).
        Expected: nvJitLink installation required.
        """
        packages = parse_cuda_packages(pip_freeze_missing_nvjitlink)

        assert packages["cublas"]["version"] is not None
        assert packages["nvjitlink"]["version"] is None

        # Validate that this is caught as a blocker
        validation = validate_cuda_library_versions(packages)
        blockers = validation["blockers"]
        assert len(blockers) > 0

    def test_scenario_all_valid_runtime_15_2(self, pip_freeze_valid_cu124):
        """
        Scenario: Runtime 15.2 + cu124 + matching libraries (PASS).
        Expected: All checks pass, no blockers.
        """
        packages = parse_cuda_packages(pip_freeze_valid_cu124)
        validation = validate_cuda_library_versions(packages)

        # No blockers
        blockers = validation["blockers"]
        assert len(blockers) == 0

        # Overall status OK
        assert validation["all_compatible"] is True

        # Also check torch branch compatibility
        torch_compat = validate_torch_branch_compatibility(
            runtime_version=15.2, torch_cuda_branch=packages["torch_cuda_branch"]
        )
        assert torch_compat["is_compatible"] is True

    def test_scenario_all_valid_runtime_14_3_cu120(self, pip_freeze_valid_cu120):
        """
        Scenario: Runtime 14.3 + cu120 + matching libraries (PASS).
        Expected: All checks pass, no blockers.
        """
        packages = parse_cuda_packages(pip_freeze_valid_cu120)
        validation = validate_cuda_library_versions(packages)

        # No blockers
        blockers = validation["blockers"]
        assert len(blockers) == 0

        # Overall status OK
        assert validation["all_compatible"] is True

        # Also check torch branch compatibility
        torch_compat = validate_torch_branch_compatibility(
            runtime_version=14.3, torch_cuda_branch=packages["torch_cuda_branch"]
        )
        assert torch_compat["is_compatible"] is True


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_pip_freeze(self):
        """Test handling of empty pip freeze output."""
        packages = parse_cuda_packages("")

        assert packages["torch"] is None
        assert packages["torch_cuda_branch"] is None
        assert packages["cublas"]["version"] is None
        assert packages["nvjitlink"]["version"] is None

    def test_no_torch_installed(self):
        """Test handling when torch is not installed."""
        pip_output = """
numpy==1.26.4
pandas==2.0.0
"""
        packages = parse_cuda_packages(pip_output)

        assert packages["torch"] is None
        assert packages["torch_cuda_branch"] is None

    def test_torch_without_cuda_branch(self):
        """Test handling of torch without CUDA branch (CPU-only)."""
        pip_output = """
torch==2.4.1
numpy==1.26.4
"""
        packages = parse_cuda_packages(pip_output)

        assert packages["torch"] == "2.4.1"
        assert packages["torch_cuda_branch"] is None

    def test_invalid_version_strings(self):
        """Test handling of malformed version strings."""
        # Should not crash, should handle gracefully
        result = check_cublas_nvjitlink_version_match("invalid", "also-invalid")

        # Should return a result, not crash
        assert "is_mismatch" in result

    def test_none_runtime_version(self):
        """Test handling of None runtime version."""
        result = validate_torch_branch_compatibility(
            runtime_version=None, torch_cuda_branch="cu124"
        )

        # Should handle gracefully
        assert "is_compatible" in result

    def test_none_torch_branch(self):
        """Test handling of None torch branch."""
        result = validate_torch_branch_compatibility(runtime_version=14.3, torch_cuda_branch=None)

        # Should handle gracefully - when torch is not installed or has no CUDA branch
        assert "is_compatible" in result
        # Since torch isn't installed/detected, it should be compatible (no conflict)
        assert result["is_compatible"] is True
