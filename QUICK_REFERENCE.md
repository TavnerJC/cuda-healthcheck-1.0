# CUDA Healthcheck - Quick Reference Card

## 🚀 Installation

```bash
pip install -r requirements.txt
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="dapi1234567890abcdef"
```

## 📝 Basic Usage

### Simple Check (5 seconds)
```python
from src import run_complete_healthcheck
result = run_complete_healthcheck()
print(f"Status: {result['status']}")
```

### Databricks Notebook (Recommended)
```python
from src.databricks import get_healthchecker

checker = get_healthchecker()
result = checker.run_healthcheck()
checker.display_results()
```

### Save to Delta Table
```python
checker.export_results_to_delta("main.cuda.healthcheck_results")
```

## 🎯 Common Tasks

### Check Version Compatibility
```python
from src import HealthcheckOrchestrator

orchestrator = HealthcheckOrchestrator()
result = orchestrator.check_compatibility("12.4", "13.0")

if not result['compatible']:
    print("⚠️ Breaking changes detected!")
```

### Get Breaking Changes for Library
```python
from src.data import get_breaking_changes

pytorch_changes = get_breaking_changes(library="pytorch")
for change in pytorch_changes:
    print(f"{change['severity']}: {change['title']}")
```

### Scan Specific Cluster
```python
from src.databricks import DatabricksHealthchecker, DatabricksConnector

connector = DatabricksConnector()
checker = DatabricksHealthchecker(
    cluster_id="your-cluster-id",
    connector=connector
)
result = checker.run_healthcheck()
```

## 🔍 Inspection

### View CUDA Environment
```python
from src.cuda_detector import CUDADetector

detector = CUDADetector()
env = detector.detect_environment()

print(f"CUDA: {env.cuda_driver_version}")
print(f"GPUs: {len(env.gpus)}")
for gpu in env.gpus:
    print(f"  - {gpu.name} (CC {gpu.compute_capability})")
```

### Check Specific Library
```python
detector = CUDADetector()
pytorch_info = detector.detect_pytorch()

print(f"PyTorch {pytorch_info.version}")
print(f"CUDA: {pytorch_info.cuda_version}")
print(f"Compatible: {pytorch_info.is_compatible}")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test Databricks integration
pytest tests/databricks/ -v

# Test with specific CUDA version
pytest tests/ -v -k "cuda_versions[12.4]"

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
DATABRICKS_HOST=https://workspace.databricks.com
DATABRICKS_TOKEN=dapi1234567890abcdef

# Optional
DATABRICKS_WAREHOUSE_ID=warehouse_id  # For Delta ops
CUDA_HEALTHCHECK_LOG_LEVEL=DEBUG     # DEBUG|INFO|WARNING|ERROR
CUDA_HEALTHCHECK_RETRY_ATTEMPTS=3    # API retry count
CUDA_HEALTHCHECK_TIMEOUT=300         # Timeout in seconds
```

### Using .env File
```bash
# Create .env file
cat > .env << EOF
DATABRICKS_HOST=https://workspace.databricks.com
DATABRICKS_TOKEN=dapi1234567890abcdef
CUDA_HEALTHCHECK_LOG_LEVEL=DEBUG
EOF

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

## 📊 Results Interpretation

### Compatibility Scores
- **90-100**: ✅ Excellent - No issues
- **70-89**: ⚠️ Good - Minor warnings
- **50-69**: ⚠️ Caution - Review issues
- **0-49**: ❌ Critical - Fix required

### Status Levels
- **healthy**: No critical issues, safe to proceed
- **warning**: Some warnings, testing recommended
- **critical**: Breaking changes detected, fix required

## 🎨 Custom Integrations

### Custom Logger
```python
from src.utils import get_logger

logger = get_logger(__name__, level="DEBUG")
logger.info("Starting healthcheck...")
```

### Custom Retry Logic
```python
from src.utils import retry_on_failure

@retry_on_failure(max_attempts=5, delay=2.0, backoff=2.0)
def my_api_call():
    return expensive_operation()
```

### Custom Exception Handling
```python
from src.utils.exceptions import CudaDetectionError

try:
    env = detector.detect_environment()
except CudaDetectionError as e:
    logger.error(f"Detection failed: {e}")
    # Fallback logic
```

## 🔗 Quick Links

- **Full README**: [README.md](README.md)
- **Environment Setup**: [docs/ENVIRONMENT_VARIABLES.md](docs/ENVIRONMENT_VARIABLES.md)
- **Implementation Details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Breaking Changes DB**: [docs/BREAKING_CHANGES.md](docs/BREAKING_CHANGES.md)
- **Test Fixtures**: [tests/conftest.py](tests/conftest.py)

## 💡 Tips

1. **Development**: Use `CUDA_HEALTHCHECK_LOG_LEVEL=DEBUG` for verbose output
2. **Production**: Store credentials in Databricks secrets, not environment variables
3. **Testing**: Use provided mock fixtures to avoid needing real Databricks cluster
4. **Performance**: Cache detection results if running multiple checks
5. **Security**: Never commit `.env` files or credentials to version control

## 🆘 Common Issues

### "credentials not provided"
→ Set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables

### "SDK not installed"
→ `pip install databricks-sdk`

### "nvidia-smi not found"
→ CUDA toolkit not installed or not in PATH

### "Cluster not running"
→ Start cluster or use `ensure_cluster_running()` method

## 📞 Support

- GitHub Issues: [Create Issue](https://github.com/your-repo/issues)
- Documentation: [docs/](docs/)
- Examples: See [examples.py](examples.py)

---

**Version**: 1.0.0 | **Last Updated**: December 2024


