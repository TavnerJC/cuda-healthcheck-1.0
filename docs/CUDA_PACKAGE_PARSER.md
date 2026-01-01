# CUDA Package Parser

## Overview

The `cuda_package_parser` module provides utilities to parse CUDA-related packages from `pip freeze` output, enabling automated detection of version incompatibilities critical for Databricks environments.

## Key Features

- **PyTorch CUDA Branch Detection**: Identifies CUDA version from PyTorch installation (e.g., `cu124`, `cu121`)
- **NVIDIA Library Versioning**: Extracts exact versions of cuBLAS, nvJitLink, and other NVIDIA libraries
- **Compatibility Checking**: Validates known incompatibilities like the CuOPT nvJitLink issue
- **Human-Readable Reports**: Formats package information for easy interpretation

## Use Cases

### 1. CuOPT nvJitLink Incompatibility Detection

**Problem**: CuOPT 25.12+ requires `nvJitLink >= 12.9.79`, but Databricks ML Runtime 16.4 provides `12.4.127` (immutable).

```python
from cuda_healthcheck.utils import (
    get_cuda_packages_from_pip,
    check_cuopt_nvjitlink_compatibility
)

# Get installed packages
packages = get_cuda_packages_from_pip()

# Check compatibility
compat = check_cuopt_nvjitlink_compatibility(packages)

if not compat['is_compatible']:
    print(f"❌ {compat['error_message']}")
    # Output: nvJitLink 12.4.127 is incompatible with CuOPT 25.12+
```

### 2. PyTorch CUDA Branch Validation

**Problem**: PyTorch built for CUDA 12.1 might not work optimally with CUDA 12.4 runtime.

```python
from cuda_healthcheck.utils import (
    parse_cuda_packages,
    check_pytorch_cuda_branch_compatibility
)

# Parse pip freeze output
pip_output = """
torch==2.4.1+cu121
nvidia-cublas-cu12==12.4.5.8
"""

packages = parse_cuda_packages(pip_output)

# Check against runtime CUDA version
compat = check_pytorch_cuda_branch_compatibility(packages, "12.4")

if not compat['is_compatible']:
    print(f"⚠️  {compat['error_message']}")
    # Output: PyTorch CUDA branch cu121 (CUDA 12.1) does not match expected CUDA 12.4
```

### 3. Comprehensive Package Report

```python
from cuda_healthcheck.utils import (
    get_cuda_packages_from_pip,
    format_cuda_packages_report
)

packages = get_cuda_packages_from_pip()
print(format_cuda_packages_report(packages))
```

**Output**:
```
CUDA Packages Report
================================================================================
PyTorch: 2.4.1 (cu124)
cuBLAS: 12.4.5.8 (12.4)
nvJitLink: 12.4.127 (12.4)

Other NVIDIA Packages (5):
  • nvidia-cuda-runtime-cu12: 12.6.77
  • nvidia-cudnn-cu12: 9.1.0.70
  • nvidia-cufft-cu12: 11.2.3.61
  • nvidia-curand-cu12: 10.3.7.77
  • nvidia-cusolver-cu12: 11.7.1.2
================================================================================
```

## API Reference

### Core Functions

#### `parse_cuda_packages(pip_freeze_output: str) -> Dict[str, Any]`

Parse CUDA-related packages from pip freeze output.

**Returns**:
```python
{
    'torch': '2.4.1',
    'torch_cuda_branch': 'cu124',
    'cublas': {
        'version': '12.4.5.8',
        'major_minor': '12.4'
    },
    'nvjitlink': {
        'version': '12.4.127',
        'major_minor': '12.4'
    },
    'other_nvidia': {
        'nvidia-cuda-runtime-cu12': '12.6.77',
        ...
    }
}
```

#### `get_cuda_packages_from_pip() -> Dict[str, Any]`

Get CUDA package information from current environment by executing `pip freeze`.

**Returns**: Same format as `parse_cuda_packages()`

#### `format_cuda_packages_report(packages: Dict[str, Any]) -> str`

Format CUDA packages information into a human-readable report.

### Compatibility Checks

#### `check_cuopt_nvjitlink_compatibility(packages: Dict[str, Any]) -> Dict[str, Any]`

Check if nvJitLink version is compatible with CuOPT 25.12+.

**Returns**:
```python
{
    'is_compatible': False,
    'nvjitlink_version': '12.4.127',
    'required_version': '12.9.79',
    'error_message': 'nvJitLink 12.4.127 is incompatible with CuOPT 25.12+...'
}
```

#### `check_pytorch_cuda_branch_compatibility(packages: Dict[str, Any], expected_cuda: str) -> Dict[str, Any]`

Check if PyTorch CUDA branch matches expected CUDA version.

**Parameters**:
- `packages`: Output from `parse_cuda_packages()`
- `expected_cuda`: Expected CUDA version (e.g., `"12.4"`, `"12.6"`)

**Returns**:
```python
{
    'is_compatible': True,
    'torch_cuda_branch': 'cu124',
    'expected_cuda': '12.4',
    'error_message': None
}
```

## Integration with CUDA Healthcheck

### In Databricks Notebooks

```python
# Cell 1: Install and import
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git

from cuda_healthcheck.utils import (
    get_cuda_packages_from_pip,
    format_cuda_packages_report,
    check_cuopt_nvjitlink_compatibility
)

# Cell 2: Analyze installed packages
packages = get_cuda_packages_from_pip()
print(format_cuda_packages_report(packages))

# Cell 3: Check CuOPT compatibility
compat = check_cuopt_nvjitlink_compatibility(packages)
if not compat['is_compatible']:
    print(f"\n⚠️  CRITICAL WARNING\n")
    print(compat['error_message'])
    print("\nℹ️  This is a PLATFORM CONSTRAINT - contact Databricks support")
```

### In Healthcheck Orchestrator

```python
from cuda_healthcheck import HealthcheckOrchestrator
from cuda_healthcheck.utils import (
    get_cuda_packages_from_pip,
    check_cuopt_nvjitlink_compatibility
)

# Run standard healthcheck
orchestrator = HealthcheckOrchestrator()
report = orchestrator.generate_report()

# Add package analysis
packages = get_cuda_packages_from_pip()
compat = check_cuopt_nvjitlink_compatibility(packages)

if not compat['is_compatible']:
    report.add_error({
        'type': 'nvjitlink_incompatibility',
        'severity': 'critical',
        'message': compat['error_message'],
        'impact': 'CuOPT 25.12+ cannot be used'
    })
```

## Regex Patterns

The parser uses the following regex patterns:

### PyTorch Version
```python
r"^torch==([0-9.]+)(?:\+cu([0-9]+))?"
```
Matches:
- `torch==2.4.1+cu124` → version: `2.4.1`, branch: `cu124`
- `torch==2.4.1` → version: `2.4.1`, branch: `None`

### NVIDIA Package
```python
r"^(nvidia-[a-zA-Z0-9_-]+)==([0-9.]+)"
```
Matches:
- `nvidia-cublas-cu12==12.4.5.8`
- `nvidia-nvjitlink-cu12==12.4.127`

### CUDA Branch Parsing
```python
r"cu(\d{1,2})(\d)"
```
Converts:
- `cu124` → CUDA `12.4`
- `cu121` → CUDA `12.1`
- `cu118` → CUDA `11.8`

## Testing

The module includes comprehensive tests (32 test cases):

```bash
pytest tests/utils/test_cuda_package_parser.py -v
```

**Test Coverage**:
- ✅ PyTorch version parsing (with/without CUDA branch)
- ✅ NVIDIA library version extraction
- ✅ Major.minor version parsing
- ✅ CuOPT nvJitLink compatibility
- ✅ PyTorch CUDA branch compatibility
- ✅ Real-world Databricks scenarios
- ✅ Edge cases (CPU-only, empty output, comments)

## Known Issues & Limitations

1. **PyTorch Nightly Builds**: Only supports stable release versioning
2. **Custom CUDA Builds**: Assumes standard NVIDIA package naming
3. **Pre-release Versions**: Does not parse alpha/beta/rc suffixes

## Future Enhancements

- [ ] Support for TensorFlow CUDA version detection
- [ ] JAX GPU package parsing
- [ ] Automatic remediation suggestions
- [ ] Integration with Databricks runtime detector for smarter validation

## Related Modules

- [`databricks.runtime_detector`](./DATABRICKS_RUNTIME_DETECTION.md): Detects Databricks runtime version
- [`databricks.driver_mapping`](./DRIVER_VERSION_MAPPING.md): Maps runtime to driver versions
- [`data.breaking_changes`](../cuda_healthcheck/data/breaking_changes.py): Breaking changes database

## See Also

- [Databricks Installation Troubleshooting](./DATABRICKS_INSTALLATION_TROUBLESHOOTING.md)
- [GitHub Issue: CuOPT nvJitLink Incompatibility](https://github.com/databricks-industry-solutions/routing/issues/11)

