"""Unit tests for recommendation generator."""

from cuda_healthcheck.utils.recommendations import (
    format_recommendations_for_notebook,
    generate_recommendations,
)


class TestGenerateRecommendations:
    """Test recommendation generation."""

    def test_no_blockers(self):
        """Test with no blockers."""
        result = generate_recommendations([])
        assert "No Blockers Detected" in result
        assert "✅" in result

    def test_driver_too_old_runtime_14_3(self):
        """Test driver too old on Runtime 14.3."""
        blockers = [
            {
                "issue": "Driver 535 (too old) for cu124 (requires 550+)",
                "root_cause": "driver_too_old",
                "fix_options": [
                    "Downgrade PyTorch to cu120",
                    "Upgrade Databricks runtime to 15.2+",
                ],
            }
        ]
        result = generate_recommendations(blockers, runtime_version=14.3)

        assert "GPU Driver Too Old" in result
        assert "IMMUTABLE" in result
        assert "14.3" in result
        assert "Downgrade" in result
        assert "cu120" in result

    def test_torch_not_installed(self):
        """Test PyTorch not installed."""
        blockers = [
            {
                "issue": "PyTorch is required but not installed",
                "root_cause": "torch_not_installed",
                "fix_command": (
                    "pip install torch --index-url " "https://download.pytorch.org/whl/cu121"
                ),
            }
        ]
        result = generate_recommendations(blockers)

        assert "PyTorch Not Installed" in result
        assert "pip install" in result
        assert "torch" in result

    def test_mixed_cuda_versions(self):
        """Test mixed CUDA 11 and 12 packages."""
        blockers = [
            {
                "issue": "Both CUDA 11 and CUDA 12 packages detected",
                "root_cause": "mixed_cuda_versions",
                "fix_options": [
                    "pip uninstall torch",
                    "pip cache purge",
                    "pip install torch --index-url https://download.pytorch.org/whl/cu124",
                ],
            }
        ]
        result = generate_recommendations(blockers)

        assert "Mixed CUDA" in result
        assert "conflict" in result.lower()
        assert "pip uninstall" in result

    def test_nvjitlink_mismatch(self):
        """Test cuBLAS/nvJitLink version mismatch."""
        blockers = [
            {
                "issue": "cuBLAS 12.4.x requires nvJitLink 12.4.x, found 12.1.x",
                "root_cause": "nvjitlink_mismatch",
                "fix_command": "pip install --upgrade nvidia-nvjitlink-cu12==12.4.*",
            }
        ]
        result = generate_recommendations(blockers)

        assert "CUDA Library Version Mismatch" in result
        assert "cuBLAS" in result or "nvJitLink" in result
        assert "pip install" in result

    def test_no_gpu_device(self):
        """Test no GPU detected."""
        blockers = [
            {
                "issue": "No GPU device detected",
                "root_cause": "no_gpu_device",
                "fix_options": [
                    "Switch to GPU cluster in Databricks",
                    "Use cloud_llm_inference instead",
                ],
            }
        ]
        result = generate_recommendations(blockers)

        assert "No GPU Detected" in result
        assert "cluster" in result.lower()

    def test_torch_no_cuda_support(self):
        """Test PyTorch built without CUDA support."""
        blockers = [
            {
                "issue": "PyTorch was not built with CUDA support",
                "root_cause": "torch_no_cuda_support",
                "fix_command": (
                    "pip uninstall torch && pip install torch "
                    "--index-url https://download.pytorch.org/whl/cu121"
                ),
            }
        ]
        result = generate_recommendations(blockers)

        assert "PyTorch Missing CUDA Support" in result
        assert "CPU-only" in result
        assert "pip install" in result

    def test_torch_branch_incompatible(self):
        """Test PyTorch CUDA branch incompatible."""
        blockers = [
            {
                "issue": "PyTorch cu124 incompatible with Runtime 14.3",
                "root_cause": "torch_branch_incompatible",
                "fix_options": [
                    "Downgrade to cu120",
                    "Upgrade Databricks runtime to 15.2+",
                ],
            }
        ]
        result = generate_recommendations(blockers, runtime_version=14.3)

        assert "PyTorch CUDA Branch Incompatible" in result
        assert "14.3" in result
        assert "cu124" in result or "cu120" in result

    def test_multiple_blockers(self):
        """Test multiple blockers at once."""
        blockers = [
            {
                "issue": "Driver too old",
                "root_cause": "driver_too_old",
                "fix_options": ["Upgrade driver"],
            },
            {
                "issue": "PyTorch not installed",
                "root_cause": "torch_not_installed",
                "fix_command": "pip install torch",
            },
        ]
        result = generate_recommendations(blockers)

        assert "Issue #1" in result
        assert "Issue #2" in result
        assert "GPU Driver Too Old" in result
        assert "PyTorch Not Installed" in result

    def test_unknown_root_cause(self):
        """Test blocker with unknown root cause."""
        blockers = [{"issue": "Some unknown error", "root_cause": "unknown_error_type"}]
        result = generate_recommendations(blockers)

        assert "Configuration Issue" in result
        assert "Some unknown error" in result

    def test_feature_specific_blocker(self):
        """Test blocker with feature context."""
        blockers = [
            {
                "issue": "CUDA required but not available",
                "root_cause": "",
                "feature": "local_llm_inference",
            }
        ]
        result = generate_recommendations(blockers)

        assert "local_llm_inference" in result

    def test_includes_general_tips(self):
        """Test that general tips are included."""
        blockers = [{"issue": "Test error", "root_cause": "test"}]
        result = generate_recommendations(blockers)

        assert "General Tips" in result
        assert "dbutils.library.restartPython()" in result
        assert "--no-cache-dir" in result


class TestFormatRecommendationsForNotebook:
    """Test notebook-specific formatting."""

    def test_no_blockers_notebook_format(self):
        """Test no blockers with notebook formatting."""
        result = format_recommendations_for_notebook([])
        assert "NO BLOCKERS DETECTED" in result
        assert "=" * 80 in result

    def test_driver_too_old_notebook_format(self):
        """Test driver too old with notebook formatting."""
        blockers = [
            {
                "issue": "Driver 535 too old for cu124",
                "root_cause": "driver_too_old",
                "fix_options": ["Downgrade to cu120", "Upgrade runtime"],
            }
        ]
        result = format_recommendations_for_notebook(blockers, runtime_version=14.3)

        assert "ACTION REQUIRED" in result
        assert "driver is too old" in result
        assert "IMMUTABLE" in result
        assert "14.3" in result
        assert "1." in result
        assert "2." in result

    def test_includes_restart_reminder(self):
        """Test that restart reminder is included."""
        blockers = [{"issue": "Test", "root_cause": "test"}]
        result = format_recommendations_for_notebook(blockers)

        assert "dbutils.library.restartPython()" in result

    def test_technical_details_hidden_by_default(self):
        """Test that technical details are hidden by default."""
        blockers = [
            {
                "issue": "Very technical error message with symbols",
                "root_cause": "driver_too_old",
            }
        ]
        result = format_recommendations_for_notebook(blockers, show_technical_details=False)

        # Should show simplified message
        assert "driver is too old" in result
        # Should NOT show full technical message
        assert "symbols" not in result

    def test_technical_details_shown_when_requested(self):
        """Test that technical details are shown when requested."""
        blockers = [
            {
                "issue": "Very technical error message with symbols",
                "root_cause": "driver_too_old",
            }
        ]
        result = format_recommendations_for_notebook(blockers, show_technical_details=True)

        # Should show both simplified and technical
        assert "driver is too old" in result
        assert "symbols" in result
        assert "Technical:" in result

    def test_multiple_blockers_numbered(self):
        """Test that multiple blockers are numbered correctly."""
        blockers = [
            {"issue": "Error 1", "root_cause": "driver_too_old"},
            {"issue": "Error 2", "root_cause": "torch_not_installed"},
            {"issue": "Error 3", "root_cause": "mixed_cuda_versions"},
        ]
        result = format_recommendations_for_notebook(blockers)

        assert "Issue #1" in result
        assert "Issue #2" in result
        assert "Issue #3" in result

    def test_runtime_15_2_message(self):
        """Test Runtime 15.2+ message."""
        blockers = [
            {
                "issue": "Driver issue",
                "root_cause": "driver_too_old",
                "fix_options": ["Fix 1"],
            }
        ]
        result = format_recommendations_for_notebook(blockers, runtime_version=15.2)

        # Runtime 15.2 should NOT have immutable message
        assert "IMMUTABLE" not in result

    def test_clean_fix_options(self):
        """Test that fix options are cleaned properly."""
        blockers = [
            {
                "issue": "Test",
                "root_cause": "test",
                "fix_options": [
                    "Option 1: pip install torch",
                    "Option 2: Upgrade runtime",
                ],
            }
        ]
        result = format_recommendations_for_notebook(blockers)

        # "Option 1:" should be removed
        assert "pip install torch" in result
        assert "Upgrade runtime" in result
        # But "Option 1:" should NOT appear
        lines = result.split("\n")
        fix_lines = [l for l in lines if "pip install" in l or "Upgrade" in l]
        for line in fix_lines:
            assert not line.strip().startswith("Option")
