# CUDA Healthcheck Tool for Databricks

Detect CUDA version incompatibilities between developer environments and Databricks clusters before they cause production failures.

## 🎯 Project Goal

Proactively identify CUDA version mismatches, library incompatibilities, and breaking changes that could cause production failures in Databricks GPU-enabled clusters.

## 🚀 Features

- **Comprehensive CUDA Detection**: Detects CUDA versions, GPU properties, and driver information
- **Library Compatibility Checking**: Analyzes PyTorch, TensorFlow, cuDF/RAPIDS compatibility
- **Breaking Changes Database**: Maintains knowledge base of CUDA version transitions
- **Databricks Integration**: High-level and low-level APIs for cluster operations
- **Delta Table Storage**: Stores results in Unity Catalog for historical analysis
- **Compatibility Scoring**: Provides actionable compatibility scores and recommendations
- **Production-Ready**: Full error handling, retry logic, logging, and testing infrastructure
- **Security-Conscious**: Supports secrets management and credential best practices

## 🏗️ Architecture

The tool provides multiple integration levels:

1. **Simple Function API** - `run_complete_healthcheck()` for quick checks
2. **Orchestrator Class** - `HealthcheckOrchestrator` for detailed workflows
3. **Databricks High-Level** - `DatabricksHealthchecker` for notebook integration
4. **Databricks Low-Level** - `DatabricksConnector` for API operations

## 📋 Prerequisites

- Python 3.10 or higher
- Databricks workspace with GPU-enabled clusters
- Databricks Personal Access Token (PAT)
- Access to Databricks Unity Catalog (for Delta table storage)

## 🔧 Installation

### For Databricks (Recommended)

**Quick Start - Import the Working Notebook:**

1. In Databricks, go to **Workspace** → **Import**
2. Select **URL**
3. Paste: `https://raw.githubusercontent.com/TavnerJC/cuda-healthcheck-1.0/main/notebooks/databricks_healthcheck.py`
4. Attach to a GPU cluster and run!

📘 **See [Databricks Deployment Guide](docs/DATABRICKS_DEPLOYMENT.md) for detailed instructions**

### For Local Development

#### 1. Clone the repository

```bash
git clone https://github.com/TavnerJC/cuda-healthcheck-1.0.git
cd cuda-healthcheck
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Or install from GitHub directly

```bash
pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git
```

### 3. Set up environment variables (Local Development Only)

See [docs/ENVIRONMENT_VARIABLES.md](docs/ENVIRONMENT_VARIABLES.md) for comprehensive configuration guide.

**Quick setup**:
```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-personal-access-token"
export DATABRICKS_WAREHOUSE_ID="your-warehouse-id"  # Optional, for Delta tables
export CUDA_HEALTHCHECK_LOG_LEVEL="INFO"  # Optional, default: INFO
```

Or use a `.env` file (recommended for local development):
```bash
# .env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi1234567890abcdef
DATABRICKS_WAREHOUSE_ID=warehouse_abc123
CUDA_HEALTHCHECK_LOG_LEVEL=DEBUG
```

> **Note:** In Databricks notebooks, credentials are auto-configured. No environment variables needed!

## 📖 Quick Start

### Databricks (Recommended)

**🚀 First time? See the [Visual Quick Start Guide](docs/DATABRICKS_QUICK_START.md) with step-by-step screenshots!**

**Use the provided notebook** - it's production-ready and tested!

```python
# Cell 1: Install the package
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git
# ⚠️ You'll see a red note: "Note: you may need to restart the kernel..." - This is NORMAL!
```

```python
# Cell 2: Restart Python (REQUIRED!)
dbutils.library.restartPython()
# ⏸️ Notebook will pause for ~10 seconds while Python restarts
# ✅ After restart, the package is ready to use!
```

```python
# Cell 3: Now you can import and use
from cuda_healthcheck import run_complete_healthcheck
import json

result = run_complete_healthcheck()
print(json.dumps(result, indent=2))
```

> **💡 Important:** After Cell 1 runs, you'll see a red warning note saying "you may need to restart the kernel". This is **expected and normal**! Just run Cell 2 to restart Python, then continue with Cell 3+. **Don't re-run Cell 1 after the restart.**

📘 **See [notebooks/databricks_healthcheck.py](notebooks/databricks_healthcheck.py) or [notebooks/databricks_healthcheck_serverless.py](notebooks/databricks_healthcheck_serverless.py) for complete working notebooks**

### Local Python (After Installation)

> **Note:** These examples assume you've already installed the package locally using `pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git`

> **For Databricks:** Use the provided notebooks instead - they include installation steps!

#### 1. Simple Healthcheck

```python
from cuda_healthcheck import run_complete_healthcheck
import json

# Run complete healthcheck
result = run_complete_healthcheck()
print(json.dumps(result, indent=2))
```

#### 2. Detailed Detection

```python
from cuda_healthcheck import CUDADetector

detector = CUDADetector()
environment = detector.detect_environment()

print(f"CUDA Driver: {environment.cuda_driver_version}")
print(f"CUDA Runtime: {environment.cuda_runtime_version}")
print(f"GPUs: {len(environment.gpus)}")
```
checker.export_results_to_delta("main.cuda_healthcheck.results")
```

### 3. Advanced Orchestration

```python
from src import HealthcheckOrchestrator

# Create orchestrator
orchestrator = HealthcheckOrchestrator()

# Generate full report
report = orchestrator.generate_report()

# Display summary
orchestrator.print_report_summary()

# Save to JSON
orchestrator.save_report_json("healthcheck_report.json")

# Check version compatibility
compatibility = orchestrator.check_compatibility(
    local_version="12.4",
    cluster_version="13.0"
)
```

## 📖 Detailed Usage

### Running Local CUDA Detection

Detect CUDA environment on the current machine:

```python
from src.cuda_detector import detect_cuda_environment

results = detect_cuda_environment()
print(results)
```

Or run directly:

```bash
python -m src.cuda_detector.detector
```

### Running Complete Healthcheck

Run a complete healthcheck with breaking change analysis:

```python
from src.healthcheck import run_complete_healthcheck

results = run_complete_healthcheck()
print(results["compatibility_analysis"]["recommendation"])
```

### Scanning Databricks Clusters

Scan all GPU clusters in your workspace:

```python
from src.databricks_api import scan_clusters

results = scan_clusters(save_to_delta=True)

# View summary
print(f"Scanned {results['summary']['total_clusters']} clusters")
print(f"Found {results['summary']['total_breaking_changes']} breaking changes")
```

### Querying Breaking Changes

Query the breaking changes database:

```python
from src.data import get_breaking_changes, score_compatibility

# Get all PyTorch changes
pytorch_changes = get_breaking_changes(library="pytorch")

# Score compatibility for your environment
score = score_compatibility(
    detected_libraries=[{"name": "pytorch", "version": "2.1.0", "cuda_version": "12.1"}],
    cuda_version="13.0",
    compute_capability="8.0"
)

print(f"Compatibility Score: {score['compatibility_score']}/100")
print(f"Recommendation: {score['recommendation']}")
```

## 📁 Project Structure

```
cuda-healthcheck/
├── src/
│   ├── cuda_detector/       # Core CUDA detection logic
│   │   ├── __init__.py
│   │   └── detector.py      # GPU, driver, library detection
│   ├── databricks_api/      # Databricks integration
│   │   ├── __init__.py
│   │   └── cluster_scanner.py  # Cluster scanning and job submission
│   ├── data/                # Breaking changes database
│   │   ├── __init__.py
│   │   └── breaking_changes.py  # CUDA breaking changes catalog
│   └── healthcheck/         # Orchestration
│       ├── __init__.py
│       └── orchestrator.py  # Complete healthcheck workflow
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── .cursorrules            # Project guidelines
└── README.md               # This file
```

## 🔍 What Gets Detected

### CUDA Environment
- CUDA driver version (via nvidia-smi)
- CUDA runtime version (from /usr/local/cuda)
- NVCC compiler version
- GPU models and memory
- Compute capabilities

### Libraries
- **PyTorch**: Version, CUDA availability, CUDA version
- **TensorFlow**: Version, GPU availability, build info
- **cuDF/RAPIDS**: Version and CUDA compatibility

### Breaking Changes
- PyTorch CUDA 12.x → 13.x compatibility
- TensorFlow compute capability requirements
- cuDF/RAPIDS version matching
- NVIDIA container requirements (Isaac Sim, BioNeMo, Modulus)
- CuDNN API changes
- Deprecated compute capabilities

## 📊 Output Format

Results are returned as JSON-formatted dictionaries:

```json
{
  "healthcheck_id": "healthcheck-20241227-092300",
  "timestamp": "2024-12-27T09:23:00.000Z",
  "cuda_environment": {
    "cuda_runtime_version": "12.4",
    "cuda_driver_version": "12.4",
    "nvcc_version": "12.4",
    "gpus": [...]
  },
  "libraries": [...],
  "compatibility_analysis": {
    "compatibility_score": 70,
    "critical_issues": 0,
    "warning_issues": 2,
    "breaking_changes": {...},
    "recommendation": "ACCEPTABLE: Environment is mostly compatible. Review warnings."
  },
  "status": "healthy"
}
```

## 🔄 Databricks Workflow

### 1. Local Development

Use Databricks Connect to test on actual clusters:

```bash
pip install databricks-connect
databricks-connect configure
```

### 2. Cluster Scanning

The tool creates a temporary notebook on your Databricks workspace and runs it on each GPU cluster to collect CUDA information.

### 3. Delta Table Storage

Results are stored in Unity Catalog:

```sql
-- Query results
SELECT cluster_id, cuda_version, compatibility_score, timestamp
FROM main.cuda_healthcheck.healthcheck_results
ORDER BY timestamp DESC;

-- Find clusters with critical issues
SELECT cluster_id, breaking_changes
FROM main.cuda_healthcheck.healthcheck_results
WHERE array_size(filter(breaking_changes, x -> x.severity = 'CRITICAL')) > 0;
```

## 🧪 Testing

The project includes comprehensive testing infrastructure:

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/databricks/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run parameterized tests for all CUDA versions
pytest tests/ -v -k "cuda_versions"
```

**Test Features**:
- 22+ comprehensive test fixtures
- Parameterized tests for CUDA 12.4, 12.6, 13.0
- Mock Databricks utilities (no real cluster required)
- High coverage potential (80%+)

See [tests/conftest.py](tests/conftest.py) for available fixtures.

## 📚 Documentation

### Core Documentation
- **[ENVIRONMENT_VARIABLES.md](docs/ENVIRONMENT_VARIABLES.md)** - Complete configuration guide
- **[DATABRICKS_DEPLOYMENT.md](docs/DATABRICKS_DEPLOYMENT.md)** - Databricks setup guide
- **[DATABRICKS_QUICK_START.md](docs/DATABRICKS_QUICK_START.md)** - Visual quick start guide
- **[BREAKING_CHANGES.md](docs/BREAKING_CHANGES.md)** - CUDA breaking changes database
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Recent enhancements overview

### Use Cases & Examples
- **[USE_CASE_ROUTING_OPTIMIZATION.md](docs/USE_CASE_ROUTING_OPTIMIZATION.md)** - Databricks Routing + CuOPT optimization
  - GPU selection (A10 vs L40S vs H100)
  - CUDA version decision making (12.4 vs 12.6 vs 13.0)
  - Problem size-based GPU selection guide
  - Integration with [Databricks Routing Accelerator](https://github.com/databricks-industry-solutions/routing)

- **[EXPERIMENT_CUOPT_BENCHMARK.md](docs/EXPERIMENT_CUOPT_BENCHMARK.md)** - 🧪 **Experimental Design: Benchmark CuOPT Performance**
  - Step-by-step guide to test A10 vs H100
  - Compare CUDA 12.6 vs 13.0 performance
  - 3 ready-to-run Databricks notebooks
  - Automated performance comparison and visualization
  - Validate with CUDA Healthcheck Tool before each run

## 🚦 CI/CD

![Tests](https://github.com/username/repo/workflows/Tests/badge.svg)
![Code Quality](https://github.com/username/repo/workflows/Code%20Quality/badge.svg)

The project includes comprehensive GitHub Actions workflows for:

### Automated Testing
- **Multi-version Testing**: Python 3.10, 3.11, 3.12
- **Coverage Reporting**: Codecov integration
- **Module Testing**: Individual module validation
- **Compatibility Tests**: CUDA version compatibility scenarios
- **Integration Tests**: Complete workflow validation
- **Notebook Validation**: Databricks notebook syntax checks

### Code Quality
- **Linting**: flake8, black, isort (line length 100)
- **Type Checking**: mypy static analysis
- **Security Scanning**: bandit vulnerability detection
- **Complexity Analysis**: radon metrics
- **Documentation**: Docstring validation

### Pull Request Automation
- **Quick Tests**: Fast validation (<2 min)
- **Changed Files Analysis**: Test coverage warnings
- **PR Size Checks**: Large PR warnings (>1000 lines)
- **Auto-labeling**: Module and type labels
- **Dependency Analysis**: Requirements validation

### Release & Maintenance
- **Automated Releases**: Tag-based releases with changelog
- **Nightly Builds**: Daily comprehensive testing
- **Dependabot**: Weekly dependency updates for GitHub Actions and Python packages

**Total**: 22 automated jobs across 5 workflows

See [docs/CICD.md](docs/CICD.md) for comprehensive CI/CD documentation.

## 📚 Supported CUDA Versions

- CUDA 12.4 ✅
- CUDA 12.6 ✅
- CUDA 13.0 ✅

## 🐛 Known Issues

- Databricks clusters must be in RUNNING state for scanning
- Delta table creation requires SQL warehouse permissions
- Some container-specific checks (Isaac Sim, BioNeMo) require specialized environments

## 🤝 Contributing

1. Create a feature branch
2. Make your changes following `.cursorrules` standards
3. Add tests for new functionality
4. Run linters: `black src/ tests/ --line-length=100 && flake8 src/ tests/ --max-line-length=100`
5. Update documentation as needed
6. Submit a PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🆕 Recent Enhancements (v1.0.0)

### New Modules
- **`src.databricks`** - Complete Databricks integration with high-level and low-level APIs
- **`src.utils`** - Logging, retry logic, and custom exceptions
- **`HealthcheckOrchestrator`** - Class-based healthcheck orchestration

### New Features
- Comprehensive error handling with custom exception types
- Retry logic with exponential backoff for API calls
- Centralized logging configuration
- Mock fixtures for testing without Databricks
- Environment variables documentation
- Security best practices for credential management

### Improvements
- Fixed all `__init__.py` files with clean exports
- Added type hints throughout
- Enhanced docstrings with examples
- Production-ready code quality

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for complete details.

## 📄 License

https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/LICENSE

## 📞 Support

For issues or questions:

- Create a GitHub issue
- https://github.com/TavnerJC/cuda-healthcheck-1.0/issues

## 🔗 References

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [Databricks GPU Clusters](https://docs.databricks.com/clusters/gpu.html)
- [PyTorch CUDA Support](https://pytorch.org/get-started/locally/)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [RAPIDS Documentation](https://docs.rapids.ai/)
