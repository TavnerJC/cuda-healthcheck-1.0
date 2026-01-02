"""
NeMo DataDesigner feature detection and CUDA requirement validation.

This module provides functionality to detect which NeMo DataDesigner features
are enabled in the environment and validate that the necessary CUDA/PyTorch
requirements are met for those features.
"""

from .datadesigner_detector import (
    DataDesignerFeature,
    FeatureRequirements,
    detect_enabled_features,
    diagnose_cuda_availability,
    get_feature_validation_report,
    validate_feature_requirements,
)

__all__ = [
    "DataDesignerFeature",
    "FeatureRequirements",
    "detect_enabled_features",
    "validate_feature_requirements",
    "get_feature_validation_report",
    "diagnose_cuda_availability",
]
