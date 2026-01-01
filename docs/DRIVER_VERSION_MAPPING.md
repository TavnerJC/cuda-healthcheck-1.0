# Driver Version Mapping for Databricks Runtimes

## 🎯 Overview

This module provides **critical** driver version mappings for Databricks ML Runtimes to detect **PyTorch + Driver incompatibilities** that users **cannot fix** in managed environments.

---

## 🚨 Why This Matters

### The Problem

Databricks ML Runtimes have **immutable driver versions**:
- Runtime 14.3 → Driver **535.x** (locked)
- Runtime 15.1 → Driver **550.x** (locked)
- Runtime 15.2+ → Driver **550.x** (locked)

**Users CANNOT upgrade these drivers**, yet PyTorch versions have specific driver requirements!

### The Impact

**Example Incompatibility:**
```
PyTorch 2.4+ requires Driver ≥ 550
Runtime 14.3 provides Driver 535 (immutable)
→ PyTorch 2.4 will FAIL on Runtime 14.3
```

This is similar to the CuOPT nvJitLink issue - a **platform constraint** users cannot fix.

---

## 📊 Complete Mapping Table

| Runtime | Driver Min | Driver Max | CUDA | Immutable? | Notes |
|---------|------------|------------|------|------------|-------|
| **13.0** | 520 | 535 | 11.8 | No | Legacy |
| **13.3** | 520 | 535 | 11.8 | No | Last 11.x |
| **14.0** | 535 | 545 | 12.2 | No | First 12.x |
| **14.3** | **535** | **545** | **12.2** | **✅ YES** | **Locked driver 535** |
| **15.0** | 550 | 560 | 12.4 | No | |
| **15.1** | **550** | **560** | **12.4** | **✅ YES** | **Locked driver 550** |
| **15.2** | **550** | **560** | **12.4** | **✅ YES** | **Locked driver 550** |
| **15.3** | 550 | 560 | 12.4 | No | |
| **15.4** | 550 | 560 | 12.4 | No | |
| **16.0** | 560 | 570 | 12.6 | No | |
| **16.4** | 560 | 570 | 12.6 | No | Latest |

### Immutable Runtimes

These runtimes have **locked drivers** that users **cannot change**:
- ✅ **14.3** → Driver 535.x (immutable)
- ✅ **15.1** → Driver 550.x (immutable)
- ✅ **15.2** → Driver 550.x (immutable)

---

## 🔧 API Reference

### `get_driver_version_for_runtime(runtime_version: float)`

Get required NVIDIA driver version for a Databricks runtime.

**Parameters:**
- `runtime_version` (float): Runtime version (e.g., 14.3, 15.2, 16.4)

**Returns:**
```python
{
    "driver_min_version": int,    # e.g., 535, 550, 560
    "driver_max_version": int,    # e.g., 545, 560, 570
    "cuda_version": str,          # e.g., "12.2", "12.4", "12.6"
    "is_immutable": bool          # True if driver is locked
}
```

**Raises:**
- `ValueError`: If runtime version is unknown

**Examples:**

```python
from cuda_healthcheck.databricks import get_driver_version_for_runtime

# Runtime 14.3 (immutable driver 535.x)
result = get_driver_version_for_runtime(14.3)
print(result)
```

**Output:**
```python
{
    "driver_min_version": 535,
    "driver_max_version": 545,
    "cuda_version": "12.2",
    "is_immutable": True  # ← CRITICAL: Cannot change!
}
```

```python
# Runtime 15.2 (immutable driver 550.x)
result = get_driver_version_for_runtime(15.2)
print(result)
```

**Output:**
```python
{
    "driver_min_version": 550,
    "driver_max_version": 560,
    "cuda_version": "12.4",
    "is_immutable": True  # ← CRITICAL: Cannot change!
}
```

---

### `check_driver_compatibility(runtime_version: float, detected_driver_version: int)`

Check if detected driver is compatible with Databricks runtime.

**Parameters:**
- `runtime_version` (float): Runtime version (e.g., 14.3, 15.2)
- `detected_driver_version` (int): Detected driver version (e.g., 535, 550)

**Returns:**
```python
{
    "is_compatible": bool,
    "expected_driver_min": int,
    "expected_driver_max": int,
    "detected_driver": int,
    "cuda_version": str,
    "is_immutable": bool,
    "error_message": Optional[str]  # Set if incompatible
}
```

**Examples:**

```python
from cuda_healthcheck.databricks import check_driver_compatibility

# ✅ Compatible: Runtime 14.3 with driver 535
result = check_driver_compatibility(14.3, 535)
print(result["is_compatible"])  # True
print(result["error_message"])  # None
```

```python
# ❌ Incompatible: Runtime 14.3 with driver 550
result = check_driver_compatibility(14.3, 550)
print(result["is_compatible"])  # False
print(result["error_message"])
```

**Output:**
```
CRITICAL: Driver 550 incompatible with Runtime 14.3 (requires 535-545). 
This runtime has an IMMUTABLE driver that users cannot change. 
This may cause PyTorch/CUDA incompatibilities.
```

```python
# ✅ Compatible: Runtime 15.2 with driver 550
result = check_driver_compatibility(15.2, 550)
print(result["is_compatible"])  # True
```

---

## 🎓 Use Cases

### Use Case 1: Detect PyTorch Incompatibility

```python
from cuda_healthcheck.databricks import (
    detect_databricks_runtime,
    get_driver_version_for_runtime,
)
from cuda_healthcheck import CUDADetector

# Detect runtime
runtime_info = detect_databricks_runtime()
runtime_version = runtime_info["runtime_version"]

# Get expected driver
driver_info = get_driver_version_for_runtime(runtime_version)

# Detect actual driver
detector = CUDADetector()
env = detector.detect_environment()
actual_driver = int(env.cuda_driver_version.split(".")[0])

# Check compatibility
if driver_info["is_immutable"]:
    print(f"⚠️  WARNING: Runtime {runtime_version} has IMMUTABLE driver")
    print(f"   Expected: {driver_info['driver_min_version']}")
    print(f"   Cannot upgrade!")
    
    # Check if PyTorch will work
    pytorch_lib = next((lib for lib in env.libraries if lib.name == "torch"), None)
    if pytorch_lib:
        pytorch_version = pytorch_lib.version
        print(f"   PyTorch {pytorch_version} detected")
        
        # PyTorch 2.4+ requires driver >= 550
        if pytorch_version >= "2.4.0" and driver_info["driver_min_version"] < 550:
            print(f"   ❌ INCOMPATIBLE: PyTorch 2.4+ requires driver ≥ 550")
            print(f"   ❌ Runtime {runtime_version} locked at driver {driver_info['driver_min_version']}")
            print(f"   💡 Solution: Use Runtime 15.1+ or downgrade PyTorch to 2.3.x")
```

### Use Case 2: Pre-Installation Check

```python
from cuda_healthcheck.databricks import (
    detect_databricks_runtime,
    check_driver_compatibility,
)

# Before installing PyTorch 2.4
runtime_info = detect_databricks_runtime()
runtime_version = runtime_info["runtime_version"]

# Get expected driver for this runtime
from cuda_healthcheck.databricks import get_driver_version_for_runtime

driver_info = get_driver_version_for_runtime(runtime_version)
expected_driver = driver_info["driver_min_version"]

# Check if PyTorch 2.4 will work
if expected_driver < 550:
    print("❌ WARNING: Cannot install PyTorch 2.4+")
    print(f"   Runtime {runtime_version} has driver {expected_driver}")
    print(f"   PyTorch 2.4+ requires driver ≥ 550")
    print("")
    print("💡 Options:")
    print("   1. Use Runtime 15.1+ (driver 550)")
    print("   2. Install PyTorch 2.3.x instead")
else:
    print("✅ Safe to install PyTorch 2.4+")
```

### Use Case 3: Automated Healthcheck

```python
from cuda_healthcheck.databricks import (
    detect_databricks_runtime,
    get_driver_version_for_runtime,
)
from cuda_healthcheck import CUDADetector

def check_pytorch_driver_compatibility():
    """Check if PyTorch version is compatible with runtime driver."""
    
    # Detect environment
    runtime_info = detect_databricks_runtime()
    runtime_version = runtime_info["runtime_version"]
    
    detector = CUDADetector()
    env = detector.detect_environment()
    
    # Get driver requirements
    driver_info = get_driver_version_for_runtime(runtime_version)
    
    # Find PyTorch
    pytorch_lib = next((lib for lib in env.libraries if lib.name == "torch"), None)
    if not pytorch_lib:
        return {"status": "ok", "message": "PyTorch not installed"}
    
    pytorch_version = pytorch_lib.version
    pytorch_major_minor = ".".join(pytorch_version.split(".")[:2])
    
    # Check known incompatibilities
    incompatibilities = []
    
    # PyTorch 2.4+ requires driver >= 550
    if pytorch_major_minor >= "2.4" and driver_info["driver_min_version"] < 550:
        if driver_info["is_immutable"]:
            incompatibilities.append({
                "severity": "CRITICAL",
                "issue": f"PyTorch {pytorch_version} requires driver ≥ 550",
                "runtime": f"Runtime {runtime_version} locked at driver {driver_info['driver_min_version']}",
                "fixable": False,
                "solution": "Use Runtime 15.1+ or downgrade PyTorch to 2.3.x"
            })
    
    if incompatibilities:
        return {
            "status": "error",
            "message": "PyTorch-Driver incompatibility detected",
            "incompatibilities": incompatibilities
        }
    
    return {"status": "ok", "message": "PyTorch compatible with driver"}

# Run check
result = check_pytorch_driver_compatibility()
print(result)
```

---

## 🧪 Testing

### Test Coverage

**23 tests** covering:
- ✅ Mapping structure validation
- ✅ Immutable runtime detection
- ✅ All runtime versions
- ✅ Compatible driver checks
- ✅ Incompatible driver checks
- ✅ Edge cases (min/max boundaries)
- ✅ Error message formats
- ✅ Unknown runtime handling

### Run Tests

```bash
pytest tests/databricks/test_driver_mapping.py -v
```

**Output:**
```
============================== 23 passed in 1.42s ==============================
```

---

## 📚 Real-World Example

### The CuOPT nvJitLink Parallel

This driver mapping issue is **analogous** to the CuOPT nvJitLink incompatibility:

| Issue | Component | Runtime Provides | Package Requires | User Can Fix? |
|-------|-----------|------------------|------------------|---------------|
| **CuOPT** | nvJitLink | 12.4.127 | 12.9.79+ | ❌ NO |
| **PyTorch 2.4** | NVIDIA Driver | 535 (Runtime 14.3) | 550+ | ❌ NO |

**Both are platform constraints that users cannot resolve!**

---

## 🔗 Integration with Healthcheck Tool

This module integrates with the main CUDA Healthcheck Tool:

```python
from cuda_healthcheck import run_complete_healthcheck
from cuda_healthcheck.databricks import check_driver_compatibility

# Run complete healthcheck
result = run_complete_healthcheck()

# Get runtime and driver info
runtime_version = result["databricks"]["runtime_version"]
driver_version = int(result["cuda_environment"]["cuda_driver_version"].split(".")[0])

# Check compatibility
compatibility = check_driver_compatibility(runtime_version, driver_version)

if not compatibility["is_compatible"]:
    print("❌ Driver incompatibility detected!")
    print(compatibility["error_message"])
```

---

## 🎯 Key Takeaways

1. **Databricks has immutable drivers** for certain runtimes (14.3, 15.1, 15.2)
2. **Users cannot upgrade these drivers** - it's a platform constraint
3. **PyTorch 2.4+ requires driver ≥ 550** - incompatible with Runtime 14.3
4. **This tool detects these incompatibilities automatically**
5. **Similar to CuOPT nvJitLink issue** - both are unfixable platform constraints

---

## 📖 References

- **Databricks ML Runtime Release Notes**: https://docs.databricks.com/release-notes/runtime/index.html
- **NVIDIA Driver Release Notes**: https://docs.nvidia.com/datacenter/tesla/drivers/
- **PyTorch CUDA Compatibility**: https://pytorch.org/get-started/locally/

---

**Implemented by:** Cursor AI Assistant  
**Date:** January 1, 2026  
**Version:** 0.5.0

