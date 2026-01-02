"""Unit tests for NeMo DataDesigner feature detection."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from cuda_healthcheck.nemo.datadesigner_detector import (
    FEATURE_DEFINITIONS,
    DataDesignerFeature,
    FeatureRequirements,
    detect_enabled_features,
    detect_from_config_file,
    detect_from_environment_vars,
    detect_from_installed_packages,
    detect_from_notebook_cells,
    get_feature_validation_report,
    validate_feature_requirements,
)


class TestFeatureRequirements:
    """Test FeatureRequirements dataclass."""

    def test_create_no_cuda_feature(self):
        """Test creating a feature with no CUDA requirement."""
        req = FeatureRequirements(
            feature_name="test_feature",
            requires_torch=False,
            requires_cuda=False,
            description="Test feature",
        )
        assert req.feature_name == "test_feature"
        assert not req.requires_torch
        assert not req.requires_cuda

    def test_create_cuda_feature(self):
        """Test creating a feature with CUDA requirement."""
        req = FeatureRequirements(
            feature_name="gpu_feature",
            requires_torch=False,
            requires_cuda=True,
            compatible_cuda_branches=["cu121", "cu124"],
        )
        # Should auto-set requires_torch to True
        assert req.requires_torch is True
        assert req.requires_cuda is True

    def test_feature_definitions_complete(self):
        """Test that all expected feature definitions exist."""
        expected_features = [
            "cloud_llm_inference",
            "local_llm_inference",
            "sampler_generation",
            "seed_processing",
        ]
        for feature in expected_features:
            assert feature in FEATURE_DEFINITIONS
            assert isinstance(FEATURE_DEFINITIONS[feature], FeatureRequirements)


class TestDetectFromConfigFile:
    """Test config file detection."""

    def test_cloud_inference_detected(self):
        """Test detection of cloud LLM inference."""
        config = {"inference": {"mode": "cloud", "model": "gpt-4"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = Path(f.name)

        try:
            features = detect_from_config_file(temp_path)
            assert "cloud_llm_inference" in features
            assert "local_llm_inference" not in features
        finally:
            temp_path.unlink()

    def test_local_inference_detected(self):
        """Test detection of local LLM inference."""
        config = {"inference": {"mode": "local", "model": "llama-3.3-70b"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = Path(f.name)

        try:
            features = detect_from_config_file(temp_path)
            assert "local_llm_inference" in features
            assert "cloud_llm_inference" not in features
        finally:
            temp_path.unlink()

    def test_samplers_detected(self):
        """Test detection of sampler generation."""
        config = {"samplers": {"enabled": ["category", "person", "uniform"]}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = Path(f.name)

        try:
            features = detect_from_config_file(temp_path)
            assert "sampler_generation" in features
        finally:
            temp_path.unlink()

    def test_seed_processing_detected(self):
        """Test detection of seed processing."""
        config = {"seed_data": {"enabled": True, "path": "/data/seeds"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = Path(f.name)

        try:
            features = detect_from_config_file(temp_path)
            assert "seed_processing" in features
        finally:
            temp_path.unlink()

    def test_all_features_detected(self):
        """Test detection of all features at once."""
        config = {
            "inference": {"mode": "local"},
            "samplers": {"enabled": ["category"]},
            "seed_data": {"enabled": True},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = Path(f.name)

        try:
            features = detect_from_config_file(temp_path)
            assert "local_llm_inference" in features
            assert "sampler_generation" in features
            assert "seed_processing" in features
        finally:
            temp_path.unlink()

    def test_missing_config_file(self):
        """Test behavior when config file doesn't exist."""
        features = detect_from_config_file(Path("/nonexistent/config.json"))
        assert len(features) == 0

    def test_invalid_json(self):
        """Test behavior with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            temp_path = Path(f.name)

        try:
            features = detect_from_config_file(temp_path)
            assert len(features) == 0  # Should handle gracefully
        finally:
            temp_path.unlink()


class TestDetectFromEnvironmentVars:
    """Test environment variable detection."""

    def test_cloud_inference_from_env(self):
        """Test detection of cloud inference from env var."""
        with patch.dict(os.environ, {"DATADESIGNER_INFERENCE_MODE": "cloud"}):
            features = detect_from_environment_vars()
            assert "cloud_llm_inference" in features

    def test_local_inference_from_env(self):
        """Test detection of local inference from env var."""
        with patch.dict(os.environ, {"DATADESIGNER_INFERENCE_MODE": "local"}):
            features = detect_from_environment_vars()
            assert "local_llm_inference" in features

    def test_samplers_from_env(self):
        """Test detection of samplers from env var."""
        with patch.dict(os.environ, {"DATADESIGNER_ENABLE_SAMPLERS": "true"}):
            features = detect_from_environment_vars()
            assert "sampler_generation" in features

    def test_seed_processing_from_env(self):
        """Test detection of seed processing from env var."""
        with patch.dict(os.environ, {"DATADESIGNER_ENABLE_SEED_PROCESSING": "true"}):
            features = detect_from_environment_vars()
            assert "seed_processing" in features

    def test_all_features_from_env(self):
        """Test detection of all features from env vars."""
        env = {
            "DATADESIGNER_INFERENCE_MODE": "local",
            "DATADESIGNER_ENABLE_SAMPLERS": "true",
            "DATADESIGNER_ENABLE_SEED_PROCESSING": "true",
        }
        with patch.dict(os.environ, env):
            features = detect_from_environment_vars()
            assert "local_llm_inference" in features
            assert "sampler_generation" in features
            assert "seed_processing" in features

    def test_case_insensitive(self):
        """Test that env var detection is case insensitive."""
        with patch.dict(os.environ, {"DATADESIGNER_INFERENCE_MODE": "CLOUD"}):
            features = detect_from_environment_vars()
            assert "cloud_llm_inference" in features


class TestDetectFromInstalledPackages:
    """Test installed package detection."""

    def test_cloud_package_detected(self):
        """Test detection of cloud inference package."""
        mock_spec = MagicMock()
        with patch("importlib.util.find_spec") as mock_find:
            mock_find.side_effect = lambda name: (
                mock_spec if name == "nemo.datadesigner.cloud" else None
            )
            features = detect_from_installed_packages()
            assert "cloud_llm_inference" in features

    def test_local_package_detected(self):
        """Test detection of local inference package."""
        mock_spec = MagicMock()
        with patch("importlib.util.find_spec") as mock_find:
            mock_find.side_effect = lambda name: (
                mock_spec if name == "nemo.datadesigner.local" else None
            )
            features = detect_from_installed_packages()
            assert "local_llm_inference" in features

    def test_samplers_package_detected(self):
        """Test detection of samplers package."""
        mock_spec = MagicMock()
        with patch("importlib.util.find_spec") as mock_find:
            mock_find.side_effect = lambda name: (
                mock_spec if name == "nemo.datadesigner.samplers" else None
            )
            features = detect_from_installed_packages()
            assert "sampler_generation" in features

    def test_seed_package_detected(self):
        """Test detection of seed processing package."""
        mock_spec = MagicMock()
        with patch("importlib.util.find_spec") as mock_find:
            mock_find.side_effect = lambda name: (
                mock_spec if name == "nemo.datadesigner.seed" else None
            )
            features = detect_from_installed_packages()
            assert "seed_processing" in features

    def test_no_packages_installed(self):
        """Test when no DataDesigner packages are installed."""
        with patch("importlib.util.find_spec", return_value=None):
            features = detect_from_installed_packages()
            assert len(features) == 0


class TestDetectFromNotebookCells:
    """Test notebook cell detection."""

    def test_cloud_inference_detected(self):
        """Test detection of cloud inference from notebook."""
        notebook_content = """
# Databricks notebook source
from nemo.datadesigner.cloud import CloudLLM
llm = CloudLLM(api_key="...")
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(notebook_content)
            temp_path = Path(f.name)

        try:
            features = detect_from_notebook_cells(temp_path)
            assert "cloud_llm_inference" in features
        finally:
            temp_path.unlink()

    def test_local_inference_detected(self):
        """Test detection of local inference from notebook."""
        notebook_content = """
# Databricks notebook source
from nemo.datadesigner.local import LocalLLM
llm = LocalLLM(model="llama-3.3-70b")
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(notebook_content)
            temp_path = Path(f.name)

        try:
            features = detect_from_notebook_cells(temp_path)
            assert "local_llm_inference" in features
        finally:
            temp_path.unlink()

    def test_samplers_detected(self):
        """Test detection of samplers from notebook."""
        notebook_content = """
# Databricks notebook source
from nemo.datadesigner.samplers import CategorySampler, PersonSampler
sampler = CategorySampler()
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(notebook_content)
            temp_path = Path(f.name)

        try:
            features = detect_from_notebook_cells(temp_path)
            assert "sampler_generation" in features
        finally:
            temp_path.unlink()

    def test_seed_processing_detected(self):
        """Test detection of seed processing from notebook."""
        notebook_content = """
# Databricks notebook source
from nemo.datadesigner.seed import SeedDataLoader
loader = SeedDataLoader()
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(notebook_content)
            temp_path = Path(f.name)

        try:
            features = detect_from_notebook_cells(temp_path)
            assert "seed_processing" in features
        finally:
            temp_path.unlink()

    def test_missing_notebook(self):
        """Test behavior when notebook doesn't exist."""
        features = detect_from_notebook_cells(Path("/nonexistent/notebook.py"))
        assert len(features) == 0


class TestDetectEnabledFeatures:
    """Test comprehensive feature detection."""

    def test_detect_from_config_only(self):
        """Test detection using config file only."""
        config = {"inference": {"mode": "cloud"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = Path(f.name)

        try:
            features = detect_enabled_features(
                config_paths=[temp_path],
                check_env_vars=False,
                check_packages=False,
            )
            assert features["cloud_llm_inference"].is_enabled
            assert not features["local_llm_inference"].is_enabled
            assert "config:" in features["cloud_llm_inference"].detection_method
        finally:
            temp_path.unlink()

    def test_detect_from_env_only(self):
        """Test detection using environment variables only."""
        with patch.dict(os.environ, {"DATADESIGNER_INFERENCE_MODE": "local"}):
            features = detect_enabled_features(check_env_vars=True, check_packages=False)
            assert features["local_llm_inference"].is_enabled
            assert features["local_llm_inference"].detection_method == "environment_variable"

    def test_multiple_detection_methods(self):
        """Test detection using multiple methods."""
        config = {"inference": {"mode": "cloud"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = Path(f.name)

        try:
            with patch.dict(os.environ, {"DATADESIGNER_ENABLE_SAMPLERS": "true"}):
                features = detect_enabled_features(
                    config_paths=[temp_path],
                    check_env_vars=True,
                    check_packages=False,
                )
                assert features["cloud_llm_inference"].is_enabled
                assert features["sampler_generation"].is_enabled
        finally:
            temp_path.unlink()

    def test_no_features_detected(self):
        """Test when no features are detected."""
        features = detect_enabled_features(check_env_vars=False, check_packages=False)
        for feature in features.values():
            assert not feature.is_enabled


class TestValidateFeatureRequirements:
    """Test feature requirement validation."""

    def test_cloud_inference_no_requirements(self):
        """Test that cloud inference has no blockers."""
        feature = DataDesignerFeature(
            feature_name="cloud_llm_inference",
            is_enabled=True,
            requirements=FEATURE_DEFINITIONS["cloud_llm_inference"],
        )
        validated = validate_feature_requirements(feature, torch_version=None, cuda_available=False)
        assert validated.validation_status == "OK"

    def test_local_inference_missing_torch(self):
        """Test local inference with missing PyTorch."""
        feature = DataDesignerFeature(
            feature_name="local_llm_inference",
            is_enabled=True,
            requirements=FEATURE_DEFINITIONS["local_llm_inference"],
        )
        validated = validate_feature_requirements(feature, torch_version=None, cuda_available=False)
        assert validated.validation_status == "BLOCKER"
        assert "PyTorch" in validated.validation_message
        assert len(validated.fix_commands) > 0

    def test_local_inference_missing_cuda(self):
        """Test local inference with missing CUDA."""
        feature = DataDesignerFeature(
            feature_name="local_llm_inference",
            is_enabled=True,
            requirements=FEATURE_DEFINITIONS["local_llm_inference"],
        )
        validated = validate_feature_requirements(
            feature, torch_version="2.4.1", cuda_available=False
        )
        assert validated.validation_status == "BLOCKER"
        assert "CUDA" in validated.validation_message

    def test_local_inference_incompatible_cuda_branch(self):
        """Test local inference with incompatible CUDA branch."""
        feature = DataDesignerFeature(
            feature_name="local_llm_inference",
            is_enabled=True,
            requirements=FEATURE_DEFINITIONS["local_llm_inference"],
        )
        validated = validate_feature_requirements(
            feature,
            torch_version="2.4.1",
            torch_cuda_branch="cu120",
            cuda_available=True,
        )
        assert validated.validation_status == "BLOCKER"
        assert "cu120" in validated.validation_message
        assert "cu121" in validated.validation_message or "cu124" in validated.validation_message

    def test_local_inference_all_requirements_met(self):
        """Test local inference with all requirements met."""
        feature = DataDesignerFeature(
            feature_name="local_llm_inference",
            is_enabled=True,
            requirements=FEATURE_DEFINITIONS["local_llm_inference"],
        )
        validated = validate_feature_requirements(
            feature,
            torch_version="2.4.1",
            torch_cuda_branch="cu124",
            cuda_available=True,
            gpu_memory_gb=80.0,
        )
        assert validated.validation_status == "OK"

    def test_gpu_memory_warning(self):
        """Test GPU memory warning."""
        feature = DataDesignerFeature(
            feature_name="local_llm_inference",
            is_enabled=True,
            requirements=FEATURE_DEFINITIONS["local_llm_inference"],
        )
        validated = validate_feature_requirements(
            feature,
            torch_version="2.4.1",
            torch_cuda_branch="cu124",
            cuda_available=True,
            gpu_memory_gb=20.0,  # Below minimum
        )
        assert validated.validation_status == "WARNING"
        assert "GPU memory" in validated.validation_message

    def test_disabled_feature_skipped(self):
        """Test that disabled features are skipped."""
        feature = DataDesignerFeature(
            feature_name="local_llm_inference",
            is_enabled=False,
            requirements=FEATURE_DEFINITIONS["local_llm_inference"],
        )
        validated = validate_feature_requirements(feature, torch_version=None, cuda_available=False)
        assert validated.validation_status == "SKIPPED"


class TestGetFeatureValidationReport:
    """Test comprehensive validation report generation."""

    def test_report_with_no_blockers(self):
        """Test report generation with no blockers."""
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
        report = get_feature_validation_report(features, torch_version=None, cuda_available=False)
        assert report["summary"]["blockers"] == 0
        assert report["summary"]["enabled_features"] == 2
        assert len(report["blockers"]) == 0

    def test_report_with_blockers(self):
        """Test report generation with blockers."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }
        report = get_feature_validation_report(features, torch_version=None, cuda_available=False)
        assert report["summary"]["blockers"] > 0
        assert len(report["blockers"]) > 0
        assert report["blockers"][0]["feature"] == "local_llm_inference"

    def test_report_with_warnings(self):
        """Test report generation with warnings."""
        features = {
            "local_llm_inference": DataDesignerFeature(
                feature_name="local_llm_inference",
                is_enabled=True,
                requirements=FEATURE_DEFINITIONS["local_llm_inference"],
            )
        }
        report = get_feature_validation_report(
            features,
            torch_version="2.4.1",
            torch_cuda_branch="cu124",
            cuda_available=True,
            gpu_memory_gb=20.0,  # Below minimum
        )
        assert report["summary"]["warnings"] > 0
        assert len(report["warnings"]) > 0

    def test_report_environment_info(self):
        """Test that report includes environment info."""
        features = {}
        report = get_feature_validation_report(
            features,
            torch_version="2.4.1",
            torch_cuda_branch="cu124",
            cuda_available=True,
            gpu_memory_gb=80.0,
        )
        assert report["environment"]["torch_version"] == "2.4.1"
        assert report["environment"]["torch_cuda_branch"] == "cu124"
        assert report["environment"]["cuda_available"] is True
        assert report["environment"]["gpu_memory_gb"] == 80.0
