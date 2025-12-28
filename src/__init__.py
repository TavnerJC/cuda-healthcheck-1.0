"""
CUDA Healthcheck Tool for Databricks.

Main package for detecting CUDA version incompatibilities and library issues
on Databricks GPU-enabled clusters.

Example:
    ```python
    # Simple healthcheck
    from src import run_complete_healthcheck
    result = run_complete_healthcheck()
    print(f"Status: {result['status']}")
    ```

Example:
    ```python
    # Detailed orchestration
    from src import HealthcheckOrchestrator
    orchestrator = HealthcheckOrchestrator()
    report = orchestrator.generate_report()
    orchestrator.print_report_summary()
    ```

Example:
    ```python
    # Databricks integration
    from src.databricks import DatabricksHealthchecker
    checker = DatabricksHealthchecker()
    result = checker.run_healthcheck()
    checker.display_results()
    ```
"""

__version__ = "1.0.0"
__author__ = "NVIDIA - CUDA Healthcheck Team"

# Core detection
from .cuda_detector import CUDADetector, detect_cuda_environment

# Data and breaking changes
from .data import (
    BreakingChange,
    BreakingChangesDatabase,
    get_breaking_changes,
    score_compatibility,
)

# Healthcheck orchestration
from .healthcheck import (
    HealthcheckOrchestrator,
    HealthcheckReport,
    run_complete_healthcheck,
)

# Databricks integration (optional - may not be available in all environments)
try:
    from .databricks import (
        DatabricksConnector,
        DatabricksHealthchecker,
        get_healthchecker,
        is_databricks_environment,
    )

    HAS_DATABRICKS = True
except ImportError:
    HAS_DATABRICKS = False
    DatabricksHealthchecker = None  # type: ignore[assignment,misc]
    DatabricksConnector = None  # type: ignore[assignment,misc]
    get_healthchecker = None  # type: ignore[assignment]
    is_databricks_environment = None  # type: ignore[assignment]

# Utilities
from .utils import get_logger, retry_on_failure

__all__ = [
    # Core detection
    "CUDADetector",
    "detect_cuda_environment",
    # Breaking changes
    "BreakingChange",
    "BreakingChangesDatabase",
    "score_compatibility",
    "get_breaking_changes",
    # Healthcheck
    "HealthcheckOrchestrator",
    "HealthcheckReport",
    "run_complete_healthcheck",
    # Databricks (if available)
    "DatabricksHealthchecker",
    "DatabricksConnector",
    "get_healthchecker",
    "is_databricks_environment",
    # Utilities
    "get_logger",
    "retry_on_failure",
    # Flags
    "HAS_DATABRICKS",
]
