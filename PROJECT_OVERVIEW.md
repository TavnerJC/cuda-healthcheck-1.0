# CUDA Healthcheck Tool - Project Overview

## 🎯 What Is This?

A comprehensive tool for detecting CUDA version incompatibilities and library issues on Databricks GPU-enabled clusters **before they cause production failures**.

## 📦 What's Included

### Core Modules

#### 1. **CUDA Detector** (`src/cuda_detector/`)
- Detects CUDA driver and runtime versions
- Identifies GPU models and compute capabilities
- Checks PyTorch, TensorFlow, and cuDF installations
- Works on any machine with NVIDIA GPUs

**Key Functions:**
```python
from src.cuda_detector import detect_cuda_environment

# Get complete CUDA environment info
env = detect_cuda_environment()
print(env["cuda_driver_version"])  # "12.4"
print(env["gpus"])  # List of GPU details
print(env["libraries"])  # Installed ML libraries
```

#### 2. **Databricks Integration** (`src/databricks_api/`)
- Scans all GPU clusters in your workspace
- Runs healthcheck jobs on each cluster
- Stores results in Unity Catalog Delta tables
- Provides cluster-wide compatibility reports

**Key Functions:**
```python
from src.databricks_api import scan_clusters

# Scan all GPU clusters
results = scan_clusters(save_to_delta=True)
print(f"Scanned {results['summary']['total_clusters']} clusters")
```

#### 3. **Breaking Changes Database** (`src/data/`)
- Comprehensive database of CUDA compatibility issues
- Covers PyTorch, TensorFlow, cuDF, NVIDIA containers
- Includes migration paths and severity levels
- Scores compatibility (0-100)

**Key Functions:**
```python
from src.data import get_breaking_changes, score_compatibility

# Get all PyTorch issues
changes = get_breaking_changes(library="pytorch")

# Score your environment
score = score_compatibility(
    detected_libraries=[...],
    cuda_version="13.0",
    compute_capability="8.0"
)
print(score["compatibility_score"])  # 65
print(score["recommendation"])  # "CAUTION: Environment has compatibility concerns..."
```

#### 4. **Healthcheck Orchestrator** (`src/healthcheck/`)
- Combines detection + breaking change analysis
- Provides actionable recommendations
- Generates JSON reports for web UIs

**Key Functions:**
```python
from src.healthcheck import run_complete_healthcheck

# Complete healthcheck with recommendations
results = run_complete_healthcheck()
print(results["status"])  # "healthy" or "unhealthy"
print(results["compatibility_analysis"]["recommendation"])
```

## 🚀 Quick Start

### Installation
```bash
cd cuda-healthcheck
pip install -r requirements.txt
```

### Usage

#### Command Line
```bash
# Detect local CUDA environment
python main.py detect

# Run complete healthcheck
python main.py healthcheck

# Scan Databricks clusters
python main.py scan

# View breaking changes
python main.py breaking-changes --library pytorch
```

#### Python API
```python
# Option 1: Quick detection
from src.cuda_detector import detect_cuda_environment
env = detect_cuda_environment()

# Option 2: Complete healthcheck
from src.healthcheck import run_complete_healthcheck
results = run_complete_healthcheck()

# Option 3: Databricks scanning
from src.databricks_api import scan_clusters
cluster_results = scan_clusters()
```

## 📊 Output Format

All results are JSON-formatted dictionaries:

```json
{
  "healthcheck_id": "healthcheck-20241227-092300",
  "timestamp": "2024-12-27T09:23:00Z",
  "cuda_environment": {
    "cuda_runtime_version": "12.4",
    "cuda_driver_version": "12.4",
    "nvcc_version": "12.4",
    "gpus": [
      {
        "name": "NVIDIA A100-SXM4-40GB",
        "compute_capability": "8.0",
        "memory_total_mb": 40960
      }
    ]
  },
  "libraries": [
    {
      "name": "pytorch",
      "version": "2.1.0",
      "cuda_version": "12.1",
      "is_compatible": true,
      "warnings": []
    }
  ],
  "compatibility_analysis": {
    "compatibility_score": 85,
    "critical_issues": 0,
    "warning_issues": 1,
    "recommendation": "ACCEPTABLE: Environment is mostly compatible. Review warnings."
  }
}
```

## 🔍 What Gets Detected

### CUDA Environment
- ✅ CUDA driver version (nvidia-smi)
- ✅ CUDA runtime version (/usr/local/cuda)
- ✅ NVCC compiler version
- ✅ GPU models and memory
- ✅ Compute capabilities

### Libraries
- ✅ **PyTorch**: Version, CUDA compatibility, availability
- ✅ **TensorFlow**: Version, GPU detection, build info
- ✅ **cuDF/RAPIDS**: Version and CUDA matching

### Breaking Changes
- 🚨 PyTorch CUDA 12.x → 13.x incompatibility
- 🚨 TensorFlow compute capability requirements
- 🚨 cuDF package naming (cu12 vs cu13)
- 🚨 NVIDIA container requirements
- ⚠️ cuDNN API changes
- ⚠️ Deprecated compute capabilities

## 📁 Project Structure

```
cuda-healthcheck/
├── src/
│   ├── cuda_detector/          # Core detection
│   │   ├── __init__.py
│   │   └── detector.py         # 500+ lines: GPU, driver, library detection
│   ├── databricks_api/         # Databricks integration
│   │   ├── __init__.py
│   │   └── cluster_scanner.py  # 400+ lines: Cluster scanning
│   ├── data/                   # Breaking changes DB
│   │   ├── __init__.py
│   │   └── breaking_changes.py # 600+ lines: 12+ breaking changes
│   └── healthcheck/            # Orchestration
│       ├── __init__.py
│       └── orchestrator.py     # Complete healthcheck workflow
├── tests/
│   ├── test_detector.py        # Unit tests for detector
│   └── test_breaking_changes.py # Unit tests for DB
├── docs/
│   ├── MIGRATION_GUIDE.md      # Detailed CUDA migration guide
│   ├── BREAKING_CHANGES.md     # Quick reference
│   └── SETUP.md                # Setup instructions
├── .github/workflows/
│   └── cuda-compatibility-tests.yml  # CI/CD for CUDA 12.4, 12.6, 13.0
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── CONTRIBUTING.md             # Contribution guidelines
└── .cursorrules               # Project rules
```

## 🛠️ Technology Stack

- **Language**: Python 3.10+
- **Databricks**: Databricks SDK, Unity Catalog
- **Testing**: pytest, pytest-cov
- **CI/CD**: GitHub Actions, CodeRabbit
- **Detection**: nvidia-smi, nvcc, library introspection

## 📋 Supported CUDA Versions

- ✅ CUDA 12.4
- ✅ CUDA 12.6
- ✅ CUDA 13.0

## 🎯 Use Cases

### 1. Pre-Migration Assessment
**Before** upgrading to CUDA 13.0:
```bash
python main.py healthcheck
```
Get compatibility score and migration recommendations.

### 2. Cluster-Wide Scanning
**Monitor** all GPU clusters:
```bash
python main.py scan
```
Results saved to Delta table for historical analysis.

### 3. CI/CD Integration
**Add** to your pipeline:
```yaml
- name: Check CUDA Compatibility
  run: python main.py healthcheck
```

### 4. Library Compatibility Check
**Verify** specific libraries:
```bash
python main.py breaking-changes --library pytorch
```

## 📚 Documentation

- **README.md**: Main documentation
- **docs/SETUP.md**: Installation and configuration
- **docs/MIGRATION_GUIDE.md**: Complete CUDA migration guide
- **docs/BREAKING_CHANGES.md**: Quick reference
- **CONTRIBUTING.md**: Development guidelines

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_detector.py::TestCUDADetector::test_detect_nvidia_smi_success -v
```

## 🔄 CI/CD

GitHub Actions automatically:
- ✅ Tests on Python 3.10, 3.11
- ✅ Tests against CUDA 12.4, 12.6, 13.0
- ✅ Runs linting (black, flake8, mypy)
- ✅ Generates coverage reports
- ✅ CodeRabbit reviews PRs

## 🚀 Databricks Workflow

1. **Setup**: Configure credentials (DATABRICKS_HOST, DATABRICKS_TOKEN)
2. **Scan**: Tool finds all GPU clusters
3. **Execute**: Creates temporary notebook, runs on each cluster
4. **Collect**: Gathers CUDA info from each cluster
5. **Analyze**: Scores compatibility with breaking changes DB
6. **Store**: Saves to Unity Catalog Delta table
7. **Report**: Generates summary and recommendations

## 📊 Delta Table Schema

```sql
CREATE TABLE main.cuda_healthcheck.healthcheck_results (
    cluster_id STRING,
    cluster_name STRING,
    cuda_version STRING,
    driver_version STRING,
    gpu_count INT,
    gpu_types ARRAY<STRING>,
    libraries ARRAY<STRUCT<...>>,
    breaking_changes ARRAY<STRUCT<...>>,
    warnings ARRAY<STRING>,
    timestamp TIMESTAMP,
    status STRING
)
USING DELTA
```

Query results:
```sql
SELECT 
    cluster_name,
    cuda_version,
    array_size(filter(breaking_changes, x -> x.severity = 'CRITICAL')) as critical_issues,
    timestamp
FROM main.cuda_healthcheck.healthcheck_results
WHERE status = 'success'
ORDER BY timestamp DESC
```

## 🎓 Key Features Explained

### Compatibility Scoring Algorithm
```
Score = 100 - (critical_count × 30) - (warning_count × 10) - (info_count × 2)

90-100: GOOD - Highly compatible
70-89:  ACCEPTABLE - Mostly compatible
50-69:  CAUTION - Compatibility concerns
0-49:   HIGH RISK - Significant issues
```

### Severity Levels
- **CRITICAL** 🚨: Will cause failures, immediate action required
- **WARNING** ⚠️: May cause issues, testing recommended
- **INFO** ℹ️: Informational, no immediate action needed

### Detection Methods
1. **nvidia-smi**: Driver version, GPU models, compute capabilities
2. **nvcc --version**: CUDA compiler version
3. **/usr/local/cuda/version.***: Runtime version
4. **Library introspection**: torch.version.cuda, tf.config, cudf.__version__

## 🔗 External References

- [NVIDIA CUDA Docs](https://docs.nvidia.com/cuda/)
- [Databricks GPU Guide](https://docs.databricks.com/en/compute/gpu.html)
- [PyTorch CUDA Support](https://pytorch.org/get-started/locally/)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [RAPIDS Documentation](https://docs.rapids.ai/)

## 📞 Getting Help

1. **Documentation**: Check docs/ folder
2. **Examples**: See main.py for CLI usage
3. **Issues**: Create GitHub issue
4. **Questions**: Tag with `question` label

## 🏆 Project Stats

- **~2500 lines** of Python code
- **12+ breaking changes** documented
- **30+ test cases**
- **3 CUDA versions** supported
- **3 ML frameworks** detected (PyTorch, TensorFlow, cuDF)
- **4 NVIDIA containers** covered (Isaac Sim, BioNeMo, Modulus, base CUDA)

## ✅ What's Complete

- ✅ Full CUDA detection (GPU, driver, runtime, libraries)
- ✅ Databricks cluster scanning and job submission
- ✅ Comprehensive breaking changes database
- ✅ Compatibility scoring and recommendations
- ✅ CLI interface with multiple commands
- ✅ Python API for programmatic access
- ✅ Unit tests with mocking
- ✅ GitHub Actions CI/CD
- ✅ Complete documentation
- ✅ Migration guides
- ✅ Contributing guidelines

## 🎯 Next Steps (Future)

- [ ] Web UI with Flask/FastAPI
- [ ] Email notifications for critical issues
- [ ] Automated scheduled scans
- [ ] More ML frameworks (JAX, MXNet)
- [ ] Container vulnerability scanning
- [ ] Historical trend analysis
- [ ] Slack/Teams integration

---

**Built with ❤️ for Databricks GPU users**








