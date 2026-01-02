# Feature-Aware CUDA Availability Diagnostics

## Overview

The `diagnose_cuda_availability()` function provides **intelligent, feature-aware diagnostics** for CUDA availability issues. Unlike simple availability checks, it:

1. **Skips checks if no features need CUDA** - Avoids false alarms for API-based workflows
2. **Identifies root causes** - Diagnoses exactly why CUDA is unavailable
3. **Provides actionable fixes** - Suggests specific commands and alternative approaches
4. **Integrates all layers** - Uses runtime detection, driver mapping, and CUDA package analysis

---

## Function Signature

```python
def diagnose_cuda_availability(
    features_enabled: Dict[str, DataDesignerFeature],
    runtime_version: Optional[float] = None,
    torch_cuda_branch: Optional[str] = None,
    driver_version: Optional[int] = None,
) -> Dict[str, Any]:
```

---

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `features_enabled` | `Dict[str, DataDesignerFeature]` | Dictionary of detected features from `detect_enabled_features()` |
| `runtime_version` | `Optional[float]` | Databricks runtime version (e.g., `14.3`, `15.2`) |
| `torch_cuda_branch` | `Optional[str]` | PyTorch CUDA branch (e.g., `"cu120"`, `"cu124"`) |
| `driver_version` | `Optional[int]` | NVIDIA driver version (e.g., `535`, `550`) |

---

## Return Structure

```python
{
    "feature_requires_cuda": bool,           # Whether any enabled feature needs CUDA
    "cuda_available": bool,                  # Result of torch.cuda.is_available()
    "gpu_device": str or None,               # GPU device name (e.g., "NVIDIA A100")
    "diagnostics": {
        "driver_version": int,               # NVIDIA driver version
        "torch_cuda_branch": str,            # PyTorch CUDA branch
        "runtime_version": float,            # Databricks runtime version
        "expected_driver_min": int or None,  # Minimum driver required
        "is_driver_compatible": bool or None,# Whether driver is compatible
        "issue": str or None,                # Human-readable issue description
        "root_cause": str or None,           # Root cause category
    },
    "severity": str or None,                 # OK, BLOCKER, SKIPPED
    "fix_command": str or None,              # Primary fix command
    "fix_options": List[str],                # Alternative fix options
}
```

---

## Root Cause Categories

| Root Cause | Description | Fix Strategy |
|------------|-------------|--------------|
| `torch_not_installed` | PyTorch is not installed | Install PyTorch with CUDA support |
| `driver_too_old` | Driver version too old for CUDA branch | Downgrade PyTorch or upgrade runtime |
| `driver_version_mismatch` | Driver incompatible with runtime | Verify runtime-driver pairing |
| `torch_no_cuda_support` | PyTorch built without CUDA support | Reinstall PyTorch with correct index URL |
| `cuda_libraries_missing` | CUDA libraries missing/incompatible | Reinstall CUDA libraries |
| `no_gpu_device` | No GPU detected | Switch to GPU cluster or use CPU alternatives |

---

## Usage Examples

### Example 1: Cloud-Only Workflow (No CUDA Required)

```python
from cuda_healthcheck.nemo import detect_enabled_features, diagnose_cuda_availability

# Set environment for cloud inference only
import os
os.environ["DATADESIGNER_INFERENCE_MODE"] = "cloud"

# Detect features
features = detect_enabled_features()

# Diagnose CUDA availability
result = diagnose_cuda_availability(features)

print(f"CUDA required: {result['feature_requires_cuda']}")  # False
print(f"Severity: {result['severity']}")                     # SKIPPED
print(f"Issue: {result['diagnostics']['issue']}")           # "No enabled features require CUDA"
```

**Output:**
```
CUDA required: False
Severity: SKIPPED
Issue: No enabled features require CUDA
```

---

### Example 2: CUDA Available and Working

```python
import os
os.environ["DATADESIGNER_INFERENCE_MODE"] = "local"

features = detect_enabled_features()

result = diagnose_cuda_availability(
    features,
    runtime_version=16.4,
    torch_cuda_branch="cu124",
    driver_version=550
)

print(f"CUDA required: {result['feature_requires_cuda']}")  # True
print(f"CUDA available: {result['cuda_available']}")        # True
print(f"GPU device: {result['gpu_device']}")                # "NVIDIA A100-SXM4-80GB"
print(f"Severity: {result['severity']}")                     # OK
```

**Output:**
```
CUDA required: True
CUDA available: True
GPU device: NVIDIA A100-SXM4-80GB
Severity: OK
```

---

### Example 3: Driver Too Old for CUDA Branch (Runtime 14.3 + cu124)

```python
import os
os.environ["DATADESIGNER_INFERENCE_MODE"] = "local"

features = detect_enabled_features()

result = diagnose_cuda_availability(
    features,
    runtime_version=14.3,
    torch_cuda_branch="cu124",
    driver_version=535
)

print(f"CUDA available: {result['cuda_available']}")        # False
print(f"Severity: {result['severity']}")                     # BLOCKER
print(f"Root cause: {result['diagnostics']['root_cause']}") # "driver_too_old"
print(f"Issue: {result['diagnostics']['issue']}")
# "Driver 535 (too old) for cu124 (requires 550+)"

print(f"\nFix options:")
for option in result['fix_options']:
    print(f"  - {option}")
```

**Output:**
```
CUDA available: False
Severity: BLOCKER
Root cause: driver_too_old
Issue: Driver 535 (too old) for cu124 (requires 550+)

Fix options:
  - Option 1: Downgrade PyTorch to cu120: pip install torch --index-url https://download.pytorch.org/whl/cu120
  - Option 2: Upgrade Databricks runtime to 15.2+ (supports CUDA 12.4 and Driver 550)
```

---

### Example 4: PyTorch Not Installed

```python
import os
os.environ["DATADESIGNER_INFERENCE_MODE"] = "local"

features = detect_enabled_features()

result = diagnose_cuda_availability(features)

print(f"CUDA available: {result['cuda_available']}")        # False
print(f"Severity: {result['severity']}")                     # BLOCKER
print(f"Root cause: {result['diagnostics']['root_cause']}") # "torch_not_installed"
print(f"Fix: {result['fix_command']}")
# "pip install torch --index-url https://download.pytorch.org/whl/cu121"
```

**Output:**
```
CUDA available: False
Severity: BLOCKER
Root cause: torch_not_installed
Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### Example 5: No GPU Device Detected

```python
import os
os.environ["DATADESIGNER_INFERENCE_MODE"] = "local"

features = detect_enabled_features()

result = diagnose_cuda_availability(features)

print(f"Root cause: {result['diagnostics']['root_cause']}") # "no_gpu_device"
print(f"Fix options:")
for option in result['fix_options']:
    print(f"  - {option}")
```

**Output:**
```
Root cause: no_gpu_device
Fix options:
  - Option 1: Switch to GPU cluster in Databricks
  - Option 2: Use cloud_llm_inference instead (no GPU required)
```

---

## Integration with Layers 1-3

The function integrates with existing healthcheck layers:

### Layer 1: Databricks Runtime Detection
```python
from cuda_healthcheck.databricks import detect_databricks_runtime

runtime_info = detect_databricks_runtime()
runtime_version = runtime_info['runtime_version']  # e.g., 14.3
```

### Layer 2: CUDA Package Parser
```python
from cuda_healthcheck.utils import parse_cuda_packages

packages = parse_cuda_packages()
torch_cuda_branch = packages['torch_cuda_branch']  # e.g., "cu124"
```

### Layer 3: Driver Version Extraction
```python
from cuda_healthcheck import detect_cuda_environment

env = detect_cuda_environment()
driver_version = int(env.cuda_driver_version.split('.')[0])  # e.g., 535
```

### Combined Usage
```python
from cuda_healthcheck.nemo import detect_enabled_features, diagnose_cuda_availability
from cuda_healthcheck.databricks import detect_databricks_runtime
from cuda_healthcheck.utils import parse_cuda_packages
from cuda_healthcheck import detect_cuda_environment

# Layer 1: Detect runtime
runtime_info = detect_databricks_runtime()

# Layer 2: Parse CUDA packages
packages = parse_cuda_packages()

# Layer 3: Detect environment
env = detect_cuda_environment()
driver_version = int(env.cuda_driver_version.split('.')[0]) if env.cuda_driver_version != "Not available" else None

# Layer 4: Detect features and diagnose CUDA
features = detect_enabled_features()
result = diagnose_cuda_availability(
    features,
    runtime_version=runtime_info['runtime_version'],
    torch_cuda_branch=packages['torch_cuda_branch'],
    driver_version=driver_version
)

if result['severity'] == 'BLOCKER':
    print(f"❌ {result['diagnostics']['issue']}")
    print(f"Fix: {result['fix_command']}")
elif result['severity'] == 'OK':
    print(f"✅ CUDA is available on {result['gpu_device']}")
elif result['severity'] == 'SKIPPED':
    print(f"⏭️  {result['diagnostics']['issue']}")
```

---

## Diagnostic Decision Tree

```
1. Check if any enabled feature requires CUDA
   ├─ NO → Return SKIPPED
   └─ YES → Continue

2. Try to import torch
   ├─ FAIL → Return BLOCKER (torch_not_installed)
   └─ SUCCESS → Check torch.cuda.is_available()

3. Is CUDA available?
   ├─ YES → Return OK with GPU device name
   └─ NO → Diagnose why

4. Diagnose CUDA unavailability:
   ├─ Check driver vs CUDA branch compatibility
   │  ├─ cu124 requires driver 550+
   │  ├─ cu121 requires driver 525+
   │  └─ cu120 requires driver 525+
   │
   ├─ Check if torch has CUDA support
   │  └─ Check torch.version.cuda attribute
   │
   ├─ Check CUDA libraries
   │  └─ Suggest reinstalling nvidia-cuda-runtime-cu12
   │
   └─ Assume no GPU device
      └─ Suggest GPU cluster or cloud alternative
```

---

## CUDA Branch to Driver Mapping

| CUDA Branch | Min Driver | Compatible Runtimes |
|-------------|------------|---------------------|
| `cu118` | 450 | Legacy only |
| `cu120` | 525 | 14.3+ |
| `cu121` | 525 | 14.3+ |
| `cu124` | 550 | 15.2+ |

**Critical Constraint:** Databricks Runtime 14.3 has **immutable Driver 535**, which is incompatible with cu124 (requires 550+).

---

## Testing

Run the test suite:
```bash
pytest tests/nemo/test_cuda_diagnostics.py -v
```

**Test Coverage: 13 tests**

1. ✅ No features require CUDA
2. ✅ CUDA available and required
3. ✅ PyTorch not installed
4. ✅ Driver too old for CUDA branch
5. ✅ Driver too old provides fix options
6. ✅ cu120 compatible with driver 535
7. ✅ cu121 compatible with driver 550
8. ✅ No GPU device detected
9. ✅ PyTorch built without CUDA support
10. ✅ Runtime 14.3 driver mapping integration
11. ✅ Multiple features, one requires CUDA
12. ✅ All features disabled
13. ✅ Diagnostics structure complete

---

## API Reference

### `diagnose_cuda_availability()`

**Purpose:** Intelligently diagnose CUDA availability issues based on enabled features.

**When to use:**
- After detecting DataDesigner features
- Before running GPU workloads
- When torch.cuda.is_available() returns False
- To provide user-friendly error messages

**When NOT to use:**
- When you just need a boolean check (use `torch.cuda.is_available()` directly)
- Before detecting features (will return SKIPPED)

**Best Practices:**
1. Always detect features first using `detect_enabled_features()`
2. Pass runtime/driver/branch info for best diagnostics
3. Check `severity` to determine if action is needed
4. Show `fix_options` to user for multiple solutions

---

## Common Scenarios and Solutions

### Scenario 1: Runtime 14.3 + PyTorch cu124 ❌

**Problem:** Driver 535 (immutable) cannot support cu124 (requires 550+)

**Solutions:**
1. **Downgrade PyTorch:**
   ```bash
   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu120
   ```
2. **Upgrade Runtime:**
   ```
   Switch to Databricks Runtime 15.2+ in cluster settings
   ```

---

### Scenario 2: No PyTorch Installed ❌

**Problem:** PyTorch is required for local LLM inference

**Solution:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### Scenario 3: Running on CPU Cluster ❌

**Problem:** No GPU detected, but local inference requires GPU

**Solutions:**
1. **Switch to GPU cluster** (recommended)
2. **Use cloud-based inference:**
   ```python
   os.environ["DATADESIGNER_INFERENCE_MODE"] = "cloud"
   ```

---

### Scenario 4: PyTorch Built for CPU Only ❌

**Problem:** PyTorch installed from wrong source (no CUDA support)

**Solution:**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## See Also

- [DataDesigner Feature Detection](./NEMO_DATADESIGNER_DETECTION.md)
- [Databricks Runtime Detection](./DATABRICKS_RUNTIME_DETECTION.md)
- [Driver Version Mapping](./DRIVER_VERSION_MAPPING.md)
- [CUDA Package Parser](./CUDA_PACKAGE_PARSER.md)
- [Enhanced Notebook](../notebooks/01_cuda_environment_validation_enhanced.py)

---

## Version Info

- **Added in:** v0.5.0
- **Module:** `cuda_healthcheck.nemo.datadesigner_detector`
- **Tests:** 13 comprehensive tests
- **Dependencies:** `torch` (optional), `cuda_healthcheck.databricks` (for runtime mapping)

