# 📚 CUDA Healthcheck Tool - API Reference

**Complete API documentation for developers**

Version: 1.0.0  
Last Updated: December 28, 2024

---

## Table of Contents

1. [Core Detection](#core-detection)
2. [Healthcheck Orchestration](#healthcheck-orchestration)
3. [Databricks Integration](#databricks-integration)
4. [Breaking Changes Database](#breaking-changes-database)
5. [Utility Functions](#utility-functions)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling](#error-handling)

---

## Core Detection

### `CUDADetector`

Main class for detecting CUDA environment details.

#### Constructor

```python
from src.cuda_detector import CUDADetector

detector = CUDADetector()
```

**No parameters required.** Automatically initializes with common CUDA paths.

#### Methods

##### `detect_nvidia_smi() -> Dict[str, Any]`

Detects GPU information using `nvidia-smi`.

**Returns:**
```python
{
    "success": bool,
    "driver_version": str,  # e.g., "535.104.05"
    "cuda_version": str,    # e.g., "12.4"
    "gpus": [
        {
            "name": str,
            "index": int,
            "memory_total_mb": int,
            "compute_capability": str,
            "driver_version": str,
            "cuda_version": str
        }
    ]
}
```

**Example:**
```python
detector = CUDADetector()
info = detector.detect_nvidia_smi()
print(f"CUDA Version: {info['cuda_version']}")
print(f"GPUs Found: {len(info['gpus'])}")
```

##### `detect_cuda_runtime() -> Optional[str]`

Detects CUDA runtime version from system files.

**Returns:** CUDA version string (e.g., "12.4") or `None`

**Example:**
```python
runtime_version = detector.detect_cuda_runtime()
if runtime_version:
    print(f"CUDA Runtime: {runtime_version}")
```

##### `detect_nvcc_version() -> Optional[str]`

Detects NVCC compiler version.

**Returns:** NVCC version string or `None`

##### `detect_pytorch() -> LibraryInfo`

Detects PyTorch installation and CUDA compatibility.

**Returns:**
```python
LibraryInfo(
    name="pytorch",
    version=str,
    cuda_version=str,
    is_compatible=bool,
    warnings=[str, ...]
)
```

##### `detect_tensorflow() -> LibraryInfo`

Detects TensorFlow installation and GPU support.

##### `detect_cudf() -> LibraryInfo`

Detects cuDF/RAPIDS installation.

##### `detect_environment() -> CUDAEnvironment`

**Main detection method** - performs complete environment scan.

**Returns:**
```python
CUDAEnvironment(
    cuda_runtime_version=str,
    cuda_driver_version=str,
    nvcc_version=str,
    gpus=[GPUInfo, ...],
    libraries=[LibraryInfo, ...],
    breaking_changes=[dict, ...],
    timestamp=str
)
```

**Example:**
```python
environment = detector.detect_environment()
print(f"Driver: {environment.cuda_driver_version}")
print(f"Runtime: {environment.cuda_runtime_version}")
print(f"GPUs: {len(environment.gpus)}")
print(f"Libraries: {len(environment.libraries)}")
```

### Convenience Function

```python
from src.cuda_detector import detect_cuda_environment

# Quick detection
result = detect_cuda_environment()
# Returns Dict[str, Any] representation
```

---

## Healthcheck Orchestration

### `HealthcheckOrchestrator`

Coordinates complete healthcheck workflow with analysis.

#### Constructor

```python
from src.healthcheck import HealthcheckOrchestrator

orchestrator = HealthcheckOrchestrator()
```

#### Methods

##### `generate_report() -> HealthcheckReport`

**Primary method** - Generates complete healthcheck report.

**Returns:**
```python
HealthcheckReport(
    healthcheck_id=str,
    timestamp=str,
    cuda_environment=dict,
    compatibility_analysis={
        "compatibility_score": int,  # 0-100
        "critical_issues": int,
        "warning_issues": int,
        "info_issues": int,
        "breaking_changes": {...},
        "recommendation": str
    },
    status=str,  # "healthy", "warning", "critical"
    recommendations=[str, ...]
)
```

**Example:**
```python
orchestrator = HealthcheckOrchestrator()
report = orchestrator.generate_report()

print(f"Status: {report.status}")
print(f"Score: {report.compatibility_analysis['compatibility_score']}/100")
print("\nRecommendations:")
for rec in report.recommendations:
    print(f"  • {rec}")
```

##### `check_compatibility(local_version: str, cluster_version: str) -> Dict[str, Any]`

Checks compatibility between two CUDA versions.

**Parameters:**
- `local_version`: Local CUDA version (e.g., "12.4")
- `cluster_version`: Cluster CUDA version (e.g., "13.0")

**Returns:**
```python
{
    "compatible": bool,
    "local_version": str,
    "cluster_version": str,
    "breaking_changes": [dict, ...],
    "severity": str,
    "recommendation": str
}
```

##### `save_report_json(filepath: str, report: Optional[HealthcheckReport] = None) -> None`

Saves report to JSON file.

**Example:**
```python
report = orchestrator.generate_report()
orchestrator.save_report_json("healthcheck_report.json", report)
```

##### `print_report_summary(report: Optional[HealthcheckReport] = None) -> None`

Prints formatted summary to console.

### Convenience Function

```python
from src.healthcheck import run_complete_healthcheck

# One-line healthcheck
result = run_complete_healthcheck()
# Returns dict representation of HealthcheckReport
```

---

## Databricks Integration

### `DatabricksHealthchecker`

High-level Databricks integration for notebooks.

#### Constructor

```python
from src.databricks import DatabricksHealthchecker

# Auto-detect cluster
checker = DatabricksHealthchecker()

# Specific cluster
checker = DatabricksHealthchecker(cluster_id="cluster-123")

# With custom connector
from src.databricks import DatabricksConnector
connector = DatabricksConnector()
checker = DatabricksHealthchecker(connector=connector)
```

#### Methods

##### `run_healthcheck() -> Dict[str, Any]`

Runs complete healthcheck on current/specified cluster.

**Returns:**
```python
{
    "healthcheck_id": str,
    "cluster_id": str,
    "cluster_name": str,
    "timestamp": str,
    "cuda_environment": {...},
    "compatibility_analysis": {...},
    "status": str,
    "recommendations": [...]
}
```

**Example:**
```python
# In Databricks notebook
checker = DatabricksHealthchecker()
result = checker.run_healthcheck()
print(f"Cluster {result['cluster_name']}: {result['status']}")
```

##### `display_results() -> None`

Displays formatted results in notebook (uses `displayHTML` if available).

##### `export_results_to_delta(table_path: str) -> None`

Exports results to Delta table.

**Parameters:**
- `table_path`: Full table path (e.g., "main.cuda.healthcheck_results")

**Example:**
```python
checker.run_healthcheck()
checker.export_results_to_delta("main.cuda.healthcheck_results")
```

##### `get_cluster_cuda_version() -> Optional[str]`

Quick check for cluster CUDA version.

##### `get_cluster_metadata() -> Dict[str, Any]`

Gets cluster metadata (ID, name, Spark version, etc.).

### `DatabricksConnector`

Low-level Databricks API connector.

#### Constructor

```python
from src.databricks import DatabricksConnector

# Use environment variables
connector = DatabricksConnector()

# Explicit credentials
connector = DatabricksConnector(
    workspace_url="https://your-workspace.cloud.databricks.com",
    token="dapi..."
)
```

#### Methods

##### `get_cluster_info(cluster_id: str) -> Optional[ClusterInfo]`

Gets detailed cluster information.

##### `get_spark_config(cluster_id: str) -> Dict[str, str]`

Gets Spark configuration for cluster.

##### `list_clusters(filter_gpu: bool = False) -> List[ClusterInfo]`

Lists all clusters (optionally filtered to GPU clusters).

##### `ensure_cluster_running(cluster_id: str, timeout: int = 600) -> bool`

Ensures cluster is running, starts if necessary.

##### `write_delta_table(table_path: str, data: List[Dict[str, Any]]) -> None`

Writes data to Delta table.

### Helper Functions

```python
from src.databricks import get_healthchecker, is_databricks_environment

# Check if running in Databricks
if is_databricks_environment():
    checker = get_healthchecker()
    result = checker.run_healthcheck()
```

---

## Breaking Changes Database

### `BreakingChangesDatabase`

Database of CUDA breaking changes and compatibility issues.

#### Constructor

```python
from src.data import BreakingChangesDatabase

db = BreakingChangesDatabase()
# Automatically loads built-in breaking changes
```

#### Methods

##### `get_all_changes() -> List[BreakingChange]`

Gets all breaking changes.

##### `get_changes_by_library(library: str) -> List[BreakingChange]`

Gets changes for specific library (pytorch, tensorflow, cudf, etc.).

**Example:**
```python
db = BreakingChangesDatabase()
pytorch_changes = db.get_changes_by_library("pytorch")
print(f"Found {len(pytorch_changes)} PyTorch breaking changes")
```

##### `get_changes_for_version(cuda_version: str) -> List[BreakingChange]`

Gets changes affecting specific CUDA version.

##### `get_changes_by_cuda_transition(from_version: str, to_version: str) -> List[BreakingChange]`

Gets changes when transitioning between CUDA versions.

**Example:**
```python
changes = db.get_changes_by_cuda_transition("12.4", "13.0")
for change in changes:
    print(f"{change.severity}: {change.title}")
```

##### `score_compatibility(...) -> Dict[str, Any]`

Scores compatibility and generates recommendations.

**Parameters:**
- `detected_libraries`: List of library info dicts
- `cuda_version`: Target CUDA version
- `compute_capability`: Optional GPU compute capability

**Returns:**
```python
{
    "compatibility_score": int,  # 0-100
    "critical_issues": int,
    "warning_issues": int,
    "info_issues": int,
    "breaking_changes": {...},
    "recommendation": str
}
```

### Convenience Functions

```python
from src.data import score_compatibility, get_breaking_changes

# Score compatibility
score = score_compatibility(
    detected_libraries=[{"name": "pytorch", "version": "2.1.0", ...}],
    cuda_version="13.0"
)

# Get breaking changes
all_changes = get_breaking_changes()
pytorch_changes = get_breaking_changes(library="pytorch")
```

---

## Utility Functions

### Logging

```python
from src.utils import get_logger, setup_logging

# Get logger for module
logger = get_logger(__name__)
logger.info("Starting detection...")

# Setup custom logging
setup_logging(level="DEBUG")
```

### Validation

```python
from src.utils import (
    validate_cuda_version,
    validate_cluster_id,
    validate_table_path,
    sanitize_cluster_name
)

# Validate formats
is_valid = validate_cuda_version("12.4")  # True
is_valid = validate_cluster_id("cluster-abc-123")  # True
is_valid = validate_table_path("main.schema.table")  # True

# Sanitize input
safe_name = sanitize_cluster_name("My Cluster #1!")  # "My_Cluster_1"
```

### Retry Logic

```python
from src.utils import retry_on_failure

@retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
def fetch_cluster_info(cluster_id: str):
    # Will retry up to 3 times with exponential backoff
    return api.get_cluster(cluster_id)
```

### Error Recovery

```python
from src.utils import GracefulDegradation, safe_detection

# Graceful degradation context
with GracefulDegradation("CUDA detection", fallback_value="Unknown") as gd:
    cuda_version = detect_cuda()
    gd.result = cuda_version

# Safe detection wrapper
result = safe_detection(
    detect_func,
    default_value="Unknown",
    operation_name="GPU detection"
)
```

---

## Performance Optimization

### Caching

```python
from src.utils import cached, memoize, LRUCache

# Decorator caching with TTL
@cached(cache_key_func=lambda x: f"cluster_{x}", ttl=300)
def get_cluster_info(cluster_id: str):
    return expensive_api_call(cluster_id)

# Simple memoization
@memoize
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Manual LRU cache
cache = LRUCache(max_size=128)
cache.put("key1", {"data": "value"})
result = cache.get("key1")
stats = cache.stats()
```

### Performance Timing

```python
from src.utils import PerformanceTimer, timed

# Context manager
with PerformanceTimer("CUDA detection"):
    environment = detector.detect_environment()
# Logs: "CUDA detection completed in 1.23s"

# Decorator
@timed
def expensive_operation():
    # ... code ...
    pass
```

### Batch Processing

```python
from src.utils import BatchProcessor

processor = BatchProcessor(batch_size=100)
results = processor.process(
    items=cluster_list,
    process_func=lambda batch: [scan_cluster(c) for c in batch]
)
```

---

## Error Handling

### Exception Hierarchy

```python
from src.utils.exceptions import (
    CudaHealthcheckError,         # Base exception
    CudaDetectionError,            # Detection failures
    DatabricksConnectionError,     # Connection issues
    ClusterNotFoundError,          # Cluster not found
    ClusterNotRunningError,        # Cluster state issues
    DeltaTableError,               # Delta table operations
    CompatibilityError,            # Compatibility issues
    BreakingChangeError,           # Breaking changes
    ConfigurationError             # Configuration problems
)
```

### Usage

```python
from src.utils.exceptions import CudaDetectionError

try:
    environment = detector.detect_environment()
except CudaDetectionError as e:
    logger.error(f"Detection failed: {e}")
    # Handle gracefully
```

---

## Data Classes

### `GPUInfo`

```python
@dataclass
class GPUInfo:
    name: str
    driver_version: str
    cuda_version: str
    compute_capability: str
    memory_total_mb: int
    gpu_index: int
```

### `LibraryInfo`

```python
@dataclass
class LibraryInfo:
    name: str
    version: str
    cuda_version: Optional[str]
    is_compatible: bool
    warnings: List[str]
```

### `CUDAEnvironment`

```python
@dataclass
class CUDAEnvironment:
    cuda_runtime_version: Optional[str]
    cuda_driver_version: Optional[str]
    nvcc_version: Optional[str]
    gpus: List[GPUInfo]
    libraries: List[LibraryInfo]
    breaking_changes: List[Dict[str, Any]]
    timestamp: str
```

### `BreakingChange`

```python
@dataclass
class BreakingChange:
    id: str
    title: str
    severity: str  # "CRITICAL", "WARNING", "INFO"
    affected_library: str
    cuda_version_from: str
    cuda_version_to: str
    description: str
    affected_apis: List[str]
    migration_path: str
    references: List[str]
    applies_to_compute_capabilities: Optional[List[str]]
```

---

## Environment Variables

### Required (for Databricks integration)

```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
```

### Optional

```bash
DATABRICKS_WAREHOUSE_ID=warehouse_id      # For Delta operations
CUDA_HEALTHCHECK_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
```

See [docs/ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) for complete details.

---

## Quick Examples

### Example 1: Basic Detection

```python
from src.cuda_detector import CUDADetector

detector = CUDADetector()
env = detector.detect_environment()
print(f"CUDA: {env.cuda_driver_version}")
print(f"GPUs: {len(env.gpus)}")
```

### Example 2: Complete Healthcheck

```python
from src.healthcheck import HealthcheckOrchestrator

orchestrator = HealthcheckOrchestrator()
report = orchestrator.generate_report()
orchestrator.print_report_summary(report)
```

### Example 3: Databricks Notebook

```python
from src.databricks import DatabricksHealthchecker

# Run and display
checker = DatabricksHealthchecker()
result = checker.run_healthcheck()
checker.display_results()

# Export to Delta
checker.export_results_to_delta("main.cuda.healthcheck_results")
```

### Example 4: Version Compatibility Check

```python
from src import HealthcheckOrchestrator

orchestrator = HealthcheckOrchestrator()
compatibility = orchestrator.check_compatibility("12.4", "13.0")
print(f"Compatible: {compatibility['compatible']}")
print(f"Recommendation: {compatibility['recommendation']}")
```

### Example 5: Breaking Changes Query

```python
from src.data import get_breaking_changes

# Get all PyTorch changes
changes = get_breaking_changes(library="pytorch")
for change in changes:
    if change['severity'] == 'CRITICAL':
        print(f"⚠️ {change['title']}")
```

---

## Type Hints

All functions include comprehensive type hints:

```python
def detect_environment(self) -> CUDAEnvironment:
    """Fully typed function."""
    pass

def score_compatibility(
    detected_libraries: List[Dict[str, Any]],
    cuda_version: str,
    compute_capability: Optional[str] = None,
) -> Dict[str, Any]:
    """All parameters and returns typed."""
    pass
```

Use with MyPy:
```bash
mypy src/ --ignore-missing-imports
```

---

## Testing

All functions are thoroughly tested:

```bash
# Run all tests
pytest tests/ -v

# Test specific module
pytest tests/test_detector.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Further Reading

- **[README.md](../README.md)** - Project overview and quick start
- **[ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)** - Configuration guide
- **[SETUP.md](SETUP.md)** - Installation and setup
- **[BREAKING_CHANGES.md](BREAKING_CHANGES.md)** - Breaking changes database
- **[CICD.md](CICD.md)** - CI/CD pipeline documentation

---

**Version**: 1.0.0  
**Last Updated**: December 28, 2024  
**Maintainers**: NVIDIA - CUDA Healthcheck Team



