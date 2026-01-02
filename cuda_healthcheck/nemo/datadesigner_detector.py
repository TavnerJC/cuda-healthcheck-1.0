"""
NeMo DataDesigner feature detection and CUDA requirement validation.

This module automatically detects which DataDesigner features are enabled
and validates that the environment meets the necessary CUDA requirements.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FeatureRequirements:
    """Requirements for a specific DataDesigner feature."""

    feature_name: str
    requires_torch: bool
    requires_cuda: bool
    compatible_cuda_branches: Optional[List[str]] = None  # e.g., ["cu121", "cu124"]
    min_gpu_memory_gb: Optional[float] = None
    description: str = ""

    def __post_init__(self):
        """Validate requirements."""
        if self.requires_cuda and not self.requires_torch:
            # If CUDA is required, torch is implicitly required
            self.requires_torch = True


@dataclass
class DataDesignerFeature:
    """Detected DataDesigner feature with validation status."""

    feature_name: str
    is_enabled: bool
    requirements: FeatureRequirements
    validation_status: str = "PENDING"  # PENDING, OK, BLOCKER, WARNING
    validation_message: Optional[str] = None
    fix_commands: List[str] = field(default_factory=list)
    detection_method: str = "unknown"


# Feature definitions with their CUDA requirements
FEATURE_DEFINITIONS = {
    "cloud_llm_inference": FeatureRequirements(
        feature_name="cloud_llm_inference",
        requires_torch=False,
        requires_cuda=False,
        description="API-based LLM inference using build.nvidia.com or similar",
    ),
    "local_llm_inference": FeatureRequirements(
        feature_name="local_llm_inference",
        requires_torch=True,
        requires_cuda=True,
        compatible_cuda_branches=["cu121", "cu124"],
        min_gpu_memory_gb=40.0,  # Llama 3.3 70B needs significant memory
        description="GPU-based local LLM inference (e.g., Llama 3.3 70B)",
    ),
    "sampler_generation": FeatureRequirements(
        feature_name="sampler_generation",
        requires_torch=False,
        requires_cuda=False,
        description="Pure Python data samplers (category, person, uniform)",
    ),
    "seed_processing": FeatureRequirements(
        feature_name="seed_processing",
        requires_torch=False,
        requires_cuda=False,
        description="Data loading and seed processing operations",
    ),
}


def detect_from_config_file(config_path: Path) -> Set[str]:
    """
    Detect enabled features from a DataDesigner config file.

    Args:
        config_path: Path to config file (JSON or YAML)

    Returns:
        Set of enabled feature names

    Example config structure:
        {
            "inference": {
                "mode": "local",  # or "cloud"
                "model": "llama-3.3-70b"
            },
            "samplers": {
                "enabled": ["category", "person"]
            },
            "seed_data": {
                "enabled": true
            }
        }
    """
    enabled_features = set()

    if not config_path.exists():
        logger.debug(f"Config file not found: {config_path}")
        return enabled_features

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Detect cloud_llm_inference
        if config.get("inference", {}).get("mode") == "cloud":
            enabled_features.add("cloud_llm_inference")
            logger.info("Detected cloud_llm_inference from config")

        # Detect local_llm_inference
        if config.get("inference", {}).get("mode") == "local":
            enabled_features.add("local_llm_inference")
            logger.info("Detected local_llm_inference from config")

        # Detect sampler_generation
        if config.get("samplers", {}).get("enabled"):
            enabled_features.add("sampler_generation")
            logger.info("Detected sampler_generation from config")

        # Detect seed_processing
        if config.get("seed_data", {}).get("enabled"):
            enabled_features.add("seed_processing")
            logger.info("Detected seed_processing from config")

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse config file {config_path}: {e}")
    except Exception as e:
        logger.warning(f"Error reading config file {config_path}: {e}")

    return enabled_features


def detect_from_environment_vars() -> Set[str]:
    """
    Detect enabled features from environment variables.

    Looks for variables like:
        - DATADESIGNER_INFERENCE_MODE=local/cloud
        - DATADESIGNER_ENABLE_SAMPLERS=true
        - DATADESIGNER_ENABLE_SEED_PROCESSING=true

    Returns:
        Set of enabled feature names
    """
    enabled_features = set()

    inference_mode = os.getenv("DATADESIGNER_INFERENCE_MODE", "").lower()
    if inference_mode == "cloud":
        enabled_features.add("cloud_llm_inference")
        logger.info("Detected cloud_llm_inference from env var")
    elif inference_mode == "local":
        enabled_features.add("local_llm_inference")
        logger.info("Detected local_llm_inference from env var")

    if os.getenv("DATADESIGNER_ENABLE_SAMPLERS", "").lower() == "true":
        enabled_features.add("sampler_generation")
        logger.info("Detected sampler_generation from env var")

    if os.getenv("DATADESIGNER_ENABLE_SEED_PROCESSING", "").lower() == "true":
        enabled_features.add("seed_processing")
        logger.info("Detected seed_processing from env var")

    return enabled_features


def detect_from_installed_packages() -> Set[str]:
    """
    Detect enabled features from installed Python packages.

    Checks for packages like:
        - nemo-datadesigner-cloud
        - nemo-datadesigner-local
        - nemo-datadesigner-samplers

    Returns:
        Set of enabled feature names
    """
    enabled_features = set()

    try:
        import importlib.util

        # Check for cloud inference package
        if importlib.util.find_spec("nemo.datadesigner.cloud"):
            enabled_features.add("cloud_llm_inference")
            logger.info("Detected cloud_llm_inference from installed package")

        # Check for local inference package
        if importlib.util.find_spec("nemo.datadesigner.local"):
            enabled_features.add("local_llm_inference")
            logger.info("Detected local_llm_inference from installed package")

        # Check for samplers package
        if importlib.util.find_spec("nemo.datadesigner.samplers"):
            enabled_features.add("sampler_generation")
            logger.info("Detected sampler_generation from installed package")

        # Check for seed processing package
        if importlib.util.find_spec("nemo.datadesigner.seed"):
            enabled_features.add("seed_processing")
            logger.info("Detected seed_processing from installed package")

    except Exception as e:
        logger.debug(f"Error detecting packages: {e}")

    return enabled_features


def detect_from_notebook_cells(notebook_path: Optional[Path] = None) -> Set[str]:
    """
    Detect enabled features by scanning Databricks notebook cells.

    Looks for import patterns and API usage like:
        - from nemo.datadesigner.cloud import CloudLLM
        - from nemo.datadesigner.local import LocalLLM
        - sampler = CategorySampler()
        - seed_loader = SeedDataLoader()

    Args:
        notebook_path: Path to notebook file (optional)

    Returns:
        Set of enabled feature names
    """
    enabled_features = set()

    if notebook_path and notebook_path.exists():
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "nemo.datadesigner.cloud" in content or "CloudLLM" in content:
                enabled_features.add("cloud_llm_inference")
                logger.info("Detected cloud_llm_inference from notebook")

            if "nemo.datadesigner.local" in content or "LocalLLM" in content:
                enabled_features.add("local_llm_inference")
                logger.info("Detected local_llm_inference from notebook")

            if "Sampler" in content or "CategorySampler" in content or "PersonSampler" in content:
                enabled_features.add("sampler_generation")
                logger.info("Detected sampler_generation from notebook")

            if "SeedDataLoader" in content or "seed_data" in content:
                enabled_features.add("seed_processing")
                logger.info("Detected seed_processing from notebook")

        except Exception as e:
            logger.debug(f"Error reading notebook {notebook_path}: {e}")

    return enabled_features


def detect_enabled_features(
    config_paths: Optional[List[Path]] = None,
    notebook_path: Optional[Path] = None,
    check_env_vars: bool = True,
    check_packages: bool = True,
) -> Dict[str, DataDesignerFeature]:
    """
    Auto-detect all enabled DataDesigner features.

    Uses multiple detection methods:
    1. Config files (JSON/YAML)
    2. Environment variables
    3. Installed packages
    4. Notebook cell analysis

    Args:
        config_paths: List of config file paths to check
        notebook_path: Path to Databricks notebook
        check_env_vars: Whether to check environment variables
        check_packages: Whether to check installed packages

    Returns:
        Dictionary mapping feature_name to DataDesignerFeature

    Example:
        >>> features = detect_enabled_features(
        ...     config_paths=[Path("datadesigner_config.json")],
        ...     check_env_vars=True
        ... )
        >>> print(features["local_llm_inference"].is_enabled)
        True
    """
    all_enabled = set()
    detection_methods = {}

    # Method 1: Config files
    if config_paths:
        for config_path in config_paths:
            detected = detect_from_config_file(config_path)
            all_enabled.update(detected)
            for feature in detected:
                detection_methods[feature] = f"config:{config_path.name}"

    # Method 2: Environment variables
    if check_env_vars:
        detected = detect_from_environment_vars()
        all_enabled.update(detected)
        for feature in detected:
            if feature not in detection_methods:
                detection_methods[feature] = "environment_variable"

    # Method 3: Installed packages
    if check_packages:
        detected = detect_from_installed_packages()
        all_enabled.update(detected)
        for feature in detected:
            if feature not in detection_methods:
                detection_methods[feature] = "installed_package"

    # Method 4: Notebook analysis
    if notebook_path:
        detected = detect_from_notebook_cells(notebook_path)
        all_enabled.update(detected)
        for feature in detected:
            if feature not in detection_methods:
                detection_methods[feature] = f"notebook:{notebook_path.name}"

    # Build feature objects
    features = {}
    for feature_name, requirements in FEATURE_DEFINITIONS.items():
        is_enabled = feature_name in all_enabled
        features[feature_name] = DataDesignerFeature(
            feature_name=feature_name,
            is_enabled=is_enabled,
            requirements=requirements,
            detection_method=detection_methods.get(feature_name, "not_detected"),
        )

    logger.info(f"Detected {len(all_enabled)} enabled features: {all_enabled}")
    return features


def validate_feature_requirements(
    feature: DataDesignerFeature,
    torch_version: Optional[str] = None,
    torch_cuda_branch: Optional[str] = None,
    cuda_available: bool = False,
    gpu_memory_gb: Optional[float] = None,
) -> DataDesignerFeature:
    """
    Validate that a feature's requirements are met.

    Args:
        feature: DataDesignerFeature to validate
        torch_version: Installed PyTorch version (e.g., "2.4.1")
        torch_cuda_branch: PyTorch CUDA branch (e.g., "cu121", "cu124")
        cuda_available: Whether CUDA is available
        gpu_memory_gb: Available GPU memory in GB

    Returns:
        Updated DataDesignerFeature with validation status

    Example:
        >>> feature = DataDesignerFeature(
        ...     feature_name="local_llm_inference",
        ...     is_enabled=True,
        ...     requirements=FEATURE_DEFINITIONS["local_llm_inference"]
        ... )
        >>> validated = validate_feature_requirements(
        ...     feature, torch_version="2.4.1", torch_cuda_branch="cu124",
        ...     cuda_available=True, gpu_memory_gb=80.0
        ... )
        >>> print(validated.validation_status)
        'OK'
    """
    if not feature.is_enabled:
        feature.validation_status = "SKIPPED"
        feature.validation_message = "Feature not enabled, skipping validation"
        return feature

    req = feature.requirements
    blockers = []
    warnings = []
    fix_commands = []

    # Check PyTorch requirement
    if req.requires_torch and not torch_version:
        blockers.append("PyTorch is required but not installed")
        fix_commands.append(
            "pip install torch --index-url " "https://download.pytorch.org/whl/cu121"
        )

    # Check CUDA requirement
    if req.requires_cuda and not cuda_available:
        blockers.append("CUDA is required but not available")
        fix_commands.append("Ensure you are running on a GPU-enabled cluster")

    # Check CUDA branch compatibility
    if (
        req.requires_cuda
        and req.compatible_cuda_branches
        and torch_cuda_branch
        and torch_cuda_branch not in req.compatible_cuda_branches
    ):
        compatible = ", ".join(req.compatible_cuda_branches)
        blockers.append(
            f"PyTorch CUDA branch {torch_cuda_branch} is not compatible. " f"Required: {compatible}"
        )
        fix_commands.append(
            f"pip install torch --index-url "
            f"https://download.pytorch.org/whl/{req.compatible_cuda_branches[0]}"
        )

    # Check GPU memory
    if req.min_gpu_memory_gb and gpu_memory_gb:
        if gpu_memory_gb < req.min_gpu_memory_gb:
            warnings.append(
                f"GPU memory ({gpu_memory_gb:.1f} GB) is below recommended "
                f"minimum ({req.min_gpu_memory_gb:.1f} GB). "
                f"Performance may be degraded or OOM errors may occur."
            )

    # Set validation status
    if blockers:
        feature.validation_status = "BLOCKER"
        feature.validation_message = "; ".join(blockers)
    elif warnings:
        feature.validation_status = "WARNING"
        feature.validation_message = "; ".join(warnings)
    else:
        feature.validation_status = "OK"
        feature.validation_message = "All requirements met"

    feature.fix_commands = fix_commands
    return feature


def get_feature_validation_report(
    features: Dict[str, DataDesignerFeature],
    torch_version: Optional[str] = None,
    torch_cuda_branch: Optional[str] = None,
    cuda_available: bool = False,
    gpu_memory_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validate all features and generate a comprehensive report.

    Args:
        features: Dictionary of detected features
        torch_version: Installed PyTorch version
        torch_cuda_branch: PyTorch CUDA branch
        cuda_available: Whether CUDA is available
        gpu_memory_gb: Available GPU memory in GB

    Returns:
        Dictionary with validation report

    Example:
        >>> features = detect_enabled_features()
        >>> report = get_feature_validation_report(
        ...     features, torch_version="2.4.1", torch_cuda_branch="cu124",
        ...     cuda_available=True, gpu_memory_gb=80.0
        ... )
        >>> print(report["summary"]["blockers"])
        0
        >>> print(report["summary"]["enabled_features"])
        2
    """
    validated_features = {}
    blockers = []
    warnings = []

    for feature_name, feature in features.items():
        validated = validate_feature_requirements(
            feature=feature,
            torch_version=torch_version,
            torch_cuda_branch=torch_cuda_branch,
            cuda_available=cuda_available,
            gpu_memory_gb=gpu_memory_gb,
        )
        validated_features[feature_name] = validated

        if validated.is_enabled:
            if validated.validation_status == "BLOCKER":
                blockers.append(
                    {
                        "feature": feature_name,
                        "message": validated.validation_message,
                        "fix_commands": validated.fix_commands,
                    }
                )
            elif validated.validation_status == "WARNING":
                warnings.append({"feature": feature_name, "message": validated.validation_message})

    enabled_count = sum(1 for f in validated_features.values() if f.is_enabled)

    return {
        "features": validated_features,
        "summary": {
            "total_features": len(validated_features),
            "enabled_features": enabled_count,
            "blockers": len(blockers),
            "warnings": len(warnings),
        },
        "blockers": blockers,
        "warnings": warnings,
        "environment": {
            "torch_version": torch_version,
            "torch_cuda_branch": torch_cuda_branch,
            "cuda_available": cuda_available,
            "gpu_memory_gb": gpu_memory_gb,
        },
    }


def diagnose_cuda_availability(
    features_enabled: Dict[str, DataDesignerFeature],
    runtime_version: Optional[float] = None,
    torch_cuda_branch: Optional[str] = None,
    driver_version: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Test torch.cuda availability with feature-aware diagnostics.

    This function intelligently diagnoses CUDA availability issues based on
    which DataDesigner features are enabled. If no features require CUDA,
    it skips the check. If features require CUDA but it's unavailable, it
    provides root cause analysis and fix commands.

    Args:
        features_enabled: Dictionary of detected DataDesigner features
        runtime_version: Databricks runtime version (e.g., 14.3, 15.2)
        torch_cuda_branch: PyTorch CUDA branch (e.g., "cu120", "cu124")
        driver_version: NVIDIA driver version (e.g., 535, 550)

    Returns:
        Dictionary with CUDA availability status and diagnostics

    Example (CUDA required but unavailable):
        >>> features = detect_enabled_features()
        >>> result = diagnose_cuda_availability(
        ...     features, runtime_version=14.3, torch_cuda_branch="cu124"
        ... )
        >>> print(result['severity'])
        'BLOCKER'
        >>> print(result['diagnostics']['issue'])
        'Driver 535 (too old) for cu124 (requires 550+)'

    Example (CUDA not required):
        >>> features = detect_enabled_features()  # cloud_llm_inference only
        >>> result = diagnose_cuda_availability(features)
        >>> print(result['feature_requires_cuda'])
        False
        >>> print(result['severity'])
        None
    """
    # Check if any enabled feature requires CUDA
    requires_cuda = any(
        f.is_enabled and f.requirements.requires_cuda for f in features_enabled.values()
    )

    result = {
        "feature_requires_cuda": requires_cuda,
        "cuda_available": False,
        "gpu_device": None,
        "diagnostics": {
            "driver_version": driver_version,
            "torch_cuda_branch": torch_cuda_branch,
            "runtime_version": runtime_version,
            "expected_driver_min": None,
            "is_driver_compatible": None,
            "issue": None,
            "root_cause": None,
        },
        "severity": None,
        "fix_command": None,
        "fix_options": [],
    }

    # If no features require CUDA, skip the check
    if not requires_cuda:
        result["severity"] = "SKIPPED"
        result["diagnostics"]["issue"] = "No enabled features require CUDA"
        logger.info("CUDA check skipped - no features require CUDA")
        return result

    # Import driver mapping utilities
    try:
        from cuda_healthcheck.databricks import get_driver_version_for_runtime
    except ImportError:
        logger.warning("Could not import driver mapping utilities")
        get_driver_version_for_runtime = None

    # Try to import torch and check CUDA availability
    torch_import_failed = False
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()

        if result["cuda_available"]:
            # Get GPU device info
            try:
                result["gpu_device"] = torch.cuda.get_device_name(0)
                result["severity"] = "OK"
                result["diagnostics"]["issue"] = "CUDA is available"
                logger.info(f"CUDA is available: {result['gpu_device']}")
                return result
            except Exception as e:
                logger.debug(f"Could not get GPU device name: {e}")
                result["gpu_device"] = "Unknown GPU"

    except ImportError:
        torch_import_failed = True
        result["diagnostics"]["root_cause"] = "torch_not_installed"
        result["diagnostics"]["issue"] = "PyTorch is not installed"
        result["severity"] = "BLOCKER"
        result["fix_command"] = (
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        result["fix_options"] = [
            "pip install torch --index-url https://download.pytorch.org/whl/cu121",
            "pip install torch --index-url https://download.pytorch.org/whl/cu124",
        ]
        logger.error("PyTorch is not installed")
        return result

    # If CUDA is not available, diagnose why
    if not result["cuda_available"]:
        result["severity"] = "BLOCKER"

        # Diagnosis 1: Check driver version compatibility with CUDA branch
        if driver_version and torch_cuda_branch and runtime_version:
            # Map runtime to expected driver
            if get_driver_version_for_runtime:
                try:
                    driver_info = get_driver_version_for_runtime(runtime_version)
                    expected_min = driver_info["driver_min_version"]
                    expected_max = driver_info["driver_max_version"]
                    result["diagnostics"]["expected_driver_min"] = expected_min

                    # Check if driver is compatible
                    is_compatible = expected_min <= driver_version <= expected_max

                    result["diagnostics"]["is_driver_compatible"] = is_compatible

                    if not is_compatible:
                        # Driver is out of expected range
                        result["diagnostics"]["root_cause"] = "driver_version_mismatch"
                        result["diagnostics"]["issue"] = (
                            f"Driver {driver_version} incompatible with Runtime "
                            f"{runtime_version} (expected {expected_min}-{expected_max})"
                        )

                except Exception as e:
                    logger.debug(f"Could not get driver info: {e}")

            # Check CUDA branch compatibility with driver
            cuda_branch_to_min_driver = {
                "cu118": 450,
                "cu120": 525,
                "cu121": 525,
                "cu124": 550,
            }

            if torch_cuda_branch in cuda_branch_to_min_driver:
                min_driver_needed = cuda_branch_to_min_driver[torch_cuda_branch]
                result["diagnostics"]["expected_driver_min"] = min_driver_needed

                if driver_version < min_driver_needed:
                    result["diagnostics"]["root_cause"] = "driver_too_old"
                    result["diagnostics"]["issue"] = (
                        f"Driver {driver_version} (too old) for {torch_cuda_branch} "
                        f"(requires {min_driver_needed}+)"
                    )
                    result["diagnostics"]["is_driver_compatible"] = False

                    # Provide fix options
                    if runtime_version and runtime_version < 15.2:
                        # Runtime 14.3 has immutable driver 535
                        result["fix_options"] = [
                            (
                                f"Option 1: Downgrade PyTorch to cu120: "
                                f"pip install torch --index-url "
                                f"https://download.pytorch.org/whl/cu120"
                            ),
                            (
                                f"Option 2: Upgrade Databricks runtime to 15.2+ "
                                f"(supports CUDA 12.4 and Driver 550)"
                            ),
                        ]
                        result["fix_command"] = result["fix_options"][0]
                    else:
                        result["fix_options"] = [
                            (
                                f"Upgrade NVIDIA driver to {min_driver_needed}+ "
                                f"(contact Databricks support)"
                            )
                        ]
                        result["fix_command"] = result["fix_options"][0]

        # Diagnosis 2: Check if torch was compiled with CUDA support
        if not result["diagnostics"]["root_cause"]:
            try:
                import torch

                if not torch.cuda.is_available():
                    # Check if torch was built with CUDA
                    if not hasattr(torch, "version") or not hasattr(torch.version, "cuda"):
                        result["diagnostics"]["root_cause"] = "torch_no_cuda_support"
                        result["diagnostics"]["issue"] = "PyTorch was not built with CUDA support"
                        result["fix_command"] = (
                            "pip uninstall torch && pip install torch "
                            "--index-url https://download.pytorch.org/whl/cu121"
                        )
                        result["fix_options"] = [
                            (
                                "Reinstall PyTorch with CUDA support: "
                                "pip install torch --index-url "
                                "https://download.pytorch.org/whl/cu121"
                            )
                        ]
                    else:
                        result["diagnostics"]["root_cause"] = "cuda_libraries_missing"
                        result["diagnostics"][
                            "issue"
                        ] = "CUDA libraries are missing or incompatible"
                        result["fix_command"] = (
                            "pip install --upgrade nvidia-cuda-runtime-cu12 "
                            "nvidia-cublas-cu12 nvidia-cudnn-cu12"
                        )
                        result["fix_options"] = [
                            "Reinstall CUDA libraries: "
                            "pip install --upgrade nvidia-cuda-runtime-cu12 "
                            "nvidia-cublas-cu12",
                            "Check /usr/local/cuda symlink exists",
                            "Contact Databricks support if on managed cluster",
                        ]
            except Exception as e:
                logger.debug(f"Error during torch diagnostics: {e}")

        # Diagnosis 3: Check for GPU device
        if not result["diagnostics"]["root_cause"]:
            result["diagnostics"]["root_cause"] = "no_gpu_device"
            result["diagnostics"][
                "issue"
            ] = "No GPU device detected - ensure running on GPU-enabled cluster"
            result["fix_command"] = "Switch to a GPU-enabled Databricks cluster (A10G, A100, T4)"
            result["fix_options"] = [
                "Option 1: Switch to GPU cluster in Databricks",
                "Option 2: Use cloud_llm_inference instead (no GPU required)",
            ]

    return result
