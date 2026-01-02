"""Unit tests for CUDA availability diagnostics."""

import sys
from unittest.mock import MagicMock, patch

from cuda_healthcheck.nemo.datadesigner_detector import (
    FEATURE_DEFINITIONS,
    DataDesignerFeature,
    diagnose_cuda_availability,
)


class TestDiagnoseCudaAvailability:
    """Test CUDA availability diagnostics function."""

    def test_no_features_require_cuda(self):
        """Test when no enabled features require CUDA."""
        features = {
            "cloud_llm_inference": DataDesignerFeature(
                feature_name="cloud_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["cloud_llm_inference"],
            ),
            "sampler_generation": DataDesignerFeature(
                feature_name="sampler_generation",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["sampler_generation"],
            ),
        }

        result = diagnose_cuda_availability(features)

        assert result["feature_requires_cuda"] is False
        assert result["severity"] == "SKIPPED"
        assert result["diagnostics"]["issue"] == "No enabled features require CUDA"

    def test_cuda_available_and_required(self):
        """Test when CUDA is required and available."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = diagnose_cuda_availability(features)

        assert result["feature_requires_cuda"] is True
        assert result["cuda_available"] is True
        assert result["gpu_device"] == "NVIDIA A100"
        assert result["severity"] == "OK"
        assert result["diagnostics"]["issue"] == "CUDA is available"

    def test_torch_not_installed(self):
        """Test when PyTorch is not installed."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        with patch.dict(sys.modules, {"torch": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
                result = diagnose_cuda_availability(features)

        assert result["feature_requires_cuda"] is True
        assert result["cuda_available"] is False
        assert result["severity"] == "BLOCKER"
        assert result["diagnostics"]["root_cause"] == "torch_not_installed"
        assert "pip install torch" in result["fix_command"]

    def test_driver_too_old_for_cuda_branch(self):
        """Test driver version too old for CUDA branch."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = diagnose_cuda_availability(
                features,
                runtime_version=14.3,
                torch_cuda_branch="cu124",
                driver_version=535,
            )

        assert result["feature_requires_cuda"] is True
        assert result["cuda_available"] is False
        assert result["severity"] == "BLOCKER"
        assert result["diagnostics"]["root_cause"] == "driver_too_old"
        assert "535" in result["diagnostics"]["issue"]
        assert "cu124" in result["diagnostics"]["issue"]
        assert result["diagnostics"]["expected_driver_min"] == 550
        assert result["diagnostics"]["is_driver_compatible"] is False

    def test_driver_too_old_provides_fix_options(self):
        """Test that driver too old provides appropriate fix options."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = diagnose_cuda_availability(
                features,
                runtime_version=14.3,
                torch_cuda_branch="cu124",
                driver_version=535,
            )

        assert len(result["fix_options"]) == 2
        assert "Downgrade PyTorch to cu120" in result["fix_options"][0]
        assert "Upgrade Databricks runtime to 15.2+" in result["fix_options"][1]
        assert "cu120" in result["fix_command"]

    def test_cuda_branch_cu120_compatible_with_driver_535(self):
        """Test that cu120 is compatible with driver 535."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = diagnose_cuda_availability(
                features,
                runtime_version=14.3,
                torch_cuda_branch="cu120",
                driver_version=535,
            )

        # cu120 requires driver 525+, so 535 is OK
        # If CUDA still unavailable, it's a different issue
        assert result["feature_requires_cuda"] is True
        assert result["cuda_available"] is False
        assert result["severity"] == "BLOCKER"
        # Should NOT be "driver_too_old" for cu120 with driver 535
        if result["diagnostics"]["root_cause"] == "driver_too_old":
            # This would be an error in our logic
            assert False, "cu120 should be compatible with driver 535"

    def test_cuda_branch_cu121_compatible_with_driver_550(self):
        """Test that cu121 is compatible with driver 550."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = diagnose_cuda_availability(
                features,
                runtime_version=15.2,
                torch_cuda_branch="cu121",
                driver_version=550,
            )

        # cu121 requires driver 525+, so 550 is OK
        assert result["cuda_available"] is True
        assert result["severity"] == "OK"

    def test_no_gpu_device_detected(self):
        """Test when no GPU device is detected."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.version.cuda = "12.1"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = diagnose_cuda_availability(features)

        assert result["feature_requires_cuda"] is True
        assert result["cuda_available"] is False
        assert result["severity"] == "BLOCKER"
        # Should detect CUDA libraries missing or no GPU device
        assert len(result["fix_options"]) > 0
        # One of the fix options should mention GPU, cluster, or CUDA libraries
        fix_text = " ".join(result["fix_options"]).lower()
        assert any(keyword in fix_text for keyword in ["gpu", "cluster", "cuda libraries", "cloud"])

    def test_torch_no_cuda_support(self):
        """Test when PyTorch was built without CUDA support."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        # Simulate torch without CUDA support
        delattr(mock_torch, "version")

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = diagnose_cuda_availability(features)

        assert result["feature_requires_cuda"] is True
        assert result["cuda_available"] is False
        assert result["severity"] == "BLOCKER"

    def test_runtime_14_3_driver_mapping(self):
        """Test Runtime 14.3 driver mapping integration."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        # Mock the get_driver_version_for_runtime function
        mock_get_driver = MagicMock(
            return_value={
                "driver_min_version": 535,
                "driver_max_version": 535,
                "cuda_version": "12.0",
                "is_immutable": True,
            }
        )

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch(
                "cuda_healthcheck.databricks.get_driver_version_for_runtime",
                mock_get_driver,
            ):
                result = diagnose_cuda_availability(
                    features,
                    runtime_version=14.3,
                    torch_cuda_branch="cu124",
                    driver_version=535,
                )

        # Runtime mapping should be called
        assert result["diagnostics"]["expected_driver_min"] in [535, 550]
        # cu124 requires driver 550+, so this should be incompatible
        assert result["diagnostics"]["root_cause"] == "driver_too_old"

    def test_multiple_features_one_requires_cuda(self):
        """Test with multiple features where only one requires CUDA."""
        features = {
            "cloud_llm_inference": DataDesignerFeature(
                feature_name="cloud_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["cloud_llm_inference"],
            ),
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            ),
            "sampler_generation": DataDesignerFeature(
                feature_name="sampler_generation",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["sampler_generation"],
            ),
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = diagnose_cuda_availability(features)

        # Should require CUDA because local_llm_inference is enabled
        assert result["feature_requires_cuda"] is True
        assert result["severity"] == "BLOCKER"

    def test_all_features_disabled(self):
        """Test when all features are disabled."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=False,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        result = diagnose_cuda_availability(features)

        assert result["feature_requires_cuda"] is False
        assert result["severity"] == "SKIPPED"

    def test_diagnostics_structure_complete(self):
        """Test that diagnostics structure is complete."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = diagnose_cuda_availability(
                features,
                runtime_version=14.3,
                torch_cuda_branch="cu124",
                driver_version=535,
            )

        # Check all required keys exist
        assert "feature_requires_cuda" in result
        assert "cuda_available" in result
        assert "gpu_device" in result
        assert "diagnostics" in result
        assert "severity" in result
        assert "fix_command" in result
        assert "fix_options" in result

        # Check diagnostics sub-structure
        diag = result["diagnostics"]
        assert "driver_version" in diag
        assert "torch_cuda_branch" in diag
        assert "runtime_version" in diag
        assert "expected_driver_min" in diag
        assert "is_driver_compatible" in diag
        assert "issue" in diag
        assert "root_cause" in diag
