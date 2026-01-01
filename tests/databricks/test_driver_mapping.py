"""
Unit tests for driver version mapping and compatibility checks.
"""

import pytest

from cuda_healthcheck.databricks.runtime_detector import (
    RUNTIME_DRIVER_MAPPING,
    check_driver_compatibility,
    get_driver_version_for_runtime,
)


class TestRuntimeDriverMapping:
    """Tests for RUNTIME_DRIVER_MAPPING constant."""

    def test_mapping_structure(self):
        """Test that mapping has correct structure."""
        assert len(RUNTIME_DRIVER_MAPPING) > 0

        for runtime, info in RUNTIME_DRIVER_MAPPING.items():
            assert isinstance(runtime, float)
            assert "driver_min" in info
            assert "driver_max" in info
            assert "cuda_version" in info
            assert isinstance(info["driver_min"], int)
            assert isinstance(info["driver_max"], int)
            assert isinstance(info["cuda_version"], str)

    def test_immutable_runtimes_present(self):
        """Test that known immutable runtimes are in mapping."""
        immutable_runtimes = {14.3, 15.1, 15.2}

        for runtime in immutable_runtimes:
            assert (
                runtime in RUNTIME_DRIVER_MAPPING
            ), f"Immutable runtime {runtime} missing from mapping"


class TestGetDriverVersionForRuntime:
    """Tests for get_driver_version_for_runtime function."""

    def test_runtime_14_3_immutable(self):
        """Test Runtime 14.3 (immutable driver 535.x)."""
        result = get_driver_version_for_runtime(14.3)

        assert result["driver_min_version"] == 535
        assert result["driver_max_version"] == 545
        assert result["cuda_version"] == "12.2"
        assert result["is_immutable"] is True

    def test_runtime_15_1_immutable(self):
        """Test Runtime 15.1 (immutable driver 550.x)."""
        result = get_driver_version_for_runtime(15.1)

        assert result["driver_min_version"] == 550
        assert result["driver_max_version"] == 560
        assert result["cuda_version"] == "12.4"
        assert result["is_immutable"] is True

    def test_runtime_15_2_immutable(self):
        """Test Runtime 15.2+ (immutable driver 550.x)."""
        result = get_driver_version_for_runtime(15.2)

        assert result["driver_min_version"] == 550
        assert result["driver_max_version"] == 560
        assert result["cuda_version"] == "12.4"
        assert result["is_immutable"] is True

    def test_runtime_16_4(self):
        """Test Runtime 16.4."""
        result = get_driver_version_for_runtime(16.4)

        assert result["driver_min_version"] == 560
        assert result["driver_max_version"] == 570
        assert result["cuda_version"] == "12.6"
        assert result["is_immutable"] is False

    def test_runtime_15_3(self):
        """Test Runtime 15.3 (not immutable)."""
        result = get_driver_version_for_runtime(15.3)

        assert result["driver_min_version"] == 550
        assert result["driver_max_version"] == 560
        assert result["cuda_version"] == "12.4"
        assert result["is_immutable"] is False

    def test_unknown_runtime_raises_value_error(self):
        """Test that unknown runtime raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_driver_version_for_runtime(99.9)

        assert "Unknown Databricks runtime version: 99.9" in str(exc_info.value)
        assert "Known versions" in str(exc_info.value)

    def test_all_mapped_runtimes(self):
        """Test all runtimes in mapping."""
        for runtime_version in RUNTIME_DRIVER_MAPPING.keys():
            result = get_driver_version_for_runtime(runtime_version)

            # All results should have these keys
            assert "driver_min_version" in result
            assert "driver_max_version" in result
            assert "cuda_version" in result
            assert "is_immutable" in result

            # Driver max should be greater than min
            assert result["driver_max_version"] > result["driver_min_version"]


class TestCheckDriverCompatibility:
    """Tests for check_driver_compatibility function."""

    def test_compatible_runtime_14_3_driver_535(self):
        """Test compatible: Runtime 14.3 with driver 535."""
        result = check_driver_compatibility(14.3, 535)

        assert result["is_compatible"] is True
        assert result["expected_driver_min"] == 535
        assert result["expected_driver_max"] == 545
        assert result["detected_driver"] == 535
        assert result["cuda_version"] == "12.2"
        assert result["is_immutable"] is True
        assert result["error_message"] is None

    def test_compatible_runtime_14_3_driver_540(self):
        """Test compatible: Runtime 14.3 with driver 540."""
        result = check_driver_compatibility(14.3, 540)

        assert result["is_compatible"] is True
        assert result["detected_driver"] == 540
        assert result["is_immutable"] is True
        assert result["error_message"] is None

    def test_incompatible_runtime_14_3_driver_550(self):
        """Test incompatible: Runtime 14.3 with driver 550."""
        result = check_driver_compatibility(14.3, 550)

        assert result["is_compatible"] is False
        assert result["expected_driver_min"] == 535
        assert result["expected_driver_max"] == 545
        assert result["detected_driver"] == 550
        assert result["cuda_version"] == "12.2"
        assert result["is_immutable"] is True
        assert result["error_message"] is not None
        assert "CRITICAL" in result["error_message"]
        assert "IMMUTABLE" in result["error_message"]

    def test_incompatible_runtime_14_3_driver_520(self):
        """Test incompatible: Runtime 14.3 with older driver 520."""
        result = check_driver_compatibility(14.3, 520)

        assert result["is_compatible"] is False
        assert result["detected_driver"] == 520
        assert result["is_immutable"] is True
        assert result["error_message"] is not None
        assert "CRITICAL" in result["error_message"]

    def test_compatible_runtime_15_1_driver_550(self):
        """Test compatible: Runtime 15.1 with driver 550."""
        result = check_driver_compatibility(15.1, 550)

        assert result["is_compatible"] is True
        assert result["expected_driver_min"] == 550
        assert result["expected_driver_max"] == 560
        assert result["detected_driver"] == 550
        assert result["cuda_version"] == "12.4"
        assert result["is_immutable"] is True
        assert result["error_message"] is None

    def test_compatible_runtime_15_2_driver_550(self):
        """Test compatible: Runtime 15.2 with driver 550."""
        result = check_driver_compatibility(15.2, 550)

        assert result["is_compatible"] is True
        assert result["expected_driver_min"] == 550
        assert result["expected_driver_max"] == 560
        assert result["detected_driver"] == 550
        assert result["cuda_version"] == "12.4"
        assert result["is_immutable"] is True
        assert result["error_message"] is None

    def test_incompatible_runtime_15_2_driver_535(self):
        """Test incompatible: Runtime 15.2 with older driver 535."""
        result = check_driver_compatibility(15.2, 535)

        assert result["is_compatible"] is False
        assert result["detected_driver"] == 535
        assert result["is_immutable"] is True
        assert result["error_message"] is not None
        assert "CRITICAL" in result["error_message"]

    def test_compatible_runtime_16_4_driver_560(self):
        """Test compatible: Runtime 16.4 with driver 560."""
        result = check_driver_compatibility(16.4, 560)

        assert result["is_compatible"] is True
        assert result["expected_driver_min"] == 560
        assert result["expected_driver_max"] == 570
        assert result["detected_driver"] == 560
        assert result["cuda_version"] == "12.6"
        assert result["is_immutable"] is False
        assert result["error_message"] is None

    def test_incompatible_non_immutable_runtime(self):
        """Test incompatible driver for non-immutable runtime."""
        result = check_driver_compatibility(15.3, 535)

        assert result["is_compatible"] is False
        assert result["is_immutable"] is False
        assert result["error_message"] is not None
        # Should NOT contain CRITICAL/IMMUTABLE for non-immutable runtimes
        assert "CRITICAL" not in result["error_message"]
        assert "IMMUTABLE" not in result["error_message"]

    def test_unknown_runtime_version(self):
        """Test with unknown runtime version."""
        result = check_driver_compatibility(99.9, 550)

        assert result["is_compatible"] is False
        assert result["expected_driver_min"] is None
        assert result["expected_driver_max"] is None
        assert result["detected_driver"] == 550
        assert result["cuda_version"] is None
        assert result["is_immutable"] is False
        assert result["error_message"] is not None
        assert "Unknown Databricks runtime version" in result["error_message"]

    def test_edge_case_driver_equals_min(self):
        """Test edge case: driver equals minimum."""
        result = check_driver_compatibility(14.3, 535)

        assert result["is_compatible"] is True

    def test_edge_case_driver_equals_max(self):
        """Test edge case: driver equals maximum (exclusive)."""
        result = check_driver_compatibility(14.3, 545)

        # Max is exclusive, so should be incompatible
        assert result["is_compatible"] is False

    def test_error_message_format_immutable(self):
        """Test error message format for immutable runtime."""
        result = check_driver_compatibility(14.3, 550)

        error = result["error_message"]
        assert "Driver 550" in error
        assert "Runtime 14.3" in error
        assert "requires 535-545" in error
        assert "CRITICAL" in error
        assert "IMMUTABLE" in error
        assert "PyTorch/CUDA" in error

    def test_error_message_format_non_immutable(self):
        """Test error message format for non-immutable runtime."""
        result = check_driver_compatibility(16.4, 550)

        error = result["error_message"]
        assert "Driver 550" in error
        assert "Runtime 16.4" in error
        assert "requires 560-570" in error
        # Should be simpler message without CRITICAL/IMMUTABLE
        assert "CRITICAL" not in error
        assert "IMMUTABLE" not in error
