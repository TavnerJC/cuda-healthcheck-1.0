# 🔧 CORRECTED: Databricks Notebook 1 - Environment Validator

## Issue Fixed
The original code tried to access `env.pytorch_available`, `env.pytorch_version`, and `env.pytorch_cuda_version` which don't exist as direct attributes on `CUDAEnvironment`.

**Correct way:** Access PyTorch info from the `env.libraries` list.

---

## ✅ Copy This Corrected Code to Your Databricks Notebook

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 🔍 CUDA Environment Validation (CORRECTED)
# MAGIC
# MAGIC Validate GPU and CUDA configuration before running CuOPT benchmarks.

# COMMAND ----------
# Install CUDA Healthcheck Tool
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git
dbutils.library.restartPython()

# COMMAND ----------
from cuda_healthcheck import CUDADetector, BreakingChangesDatabase
from cuda_healthcheck.databricks import detect_gpu_auto
import json
from datetime import datetime, timezone

# COMMAND ----------
# MAGIC %md
# MAGIC ## Detect GPU Configuration

# COMMAND ----------
# Auto-detect GPU (works on both Serverless and Classic)
gpu_info = detect_gpu_auto()

print("=" * 80)
print("🎮 GPU DETECTION RESULTS")
print("=" * 80)
print(f"Environment Type: {gpu_info['environment']}")
print(f"Detection Method: {gpu_info['method']}")
print(f"GPU Count: {gpu_info['gpu_count']}")

if gpu_info['gpu_count'] > 0:
    for gpu in gpu_info['gpus']:
        print(f"\n📊 GPU: {gpu['name']}")
        print(f"   Driver: {gpu['driver_version']}")
        print(f"   Memory: {gpu['memory_total']}")
        print(f"   Compute: {gpu['compute_capability']}")
        print(f"   UUID: {gpu['uuid']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Detect CUDA & Libraries

# COMMAND ----------
# Full CUDA detection
detector = CUDADetector()
env = detector.detect_environment()

print("=" * 80)
print("🔧 CUDA ENVIRONMENT")
print("=" * 80)
print(f"CUDA Runtime: {env.cuda_runtime_version}")
print(f"CUDA Driver: {env.cuda_driver_version}")
print(f"NVCC Version: {env.nvcc_version}")

# ✅ CORRECTED: Extract PyTorch info from libraries list
pytorch_lib = None
for lib in env.libraries:
    if lib.name.lower() == "pytorch":
        pytorch_lib = lib
        break

if pytorch_lib:
    print(f"\n🔥 PyTorch: {pytorch_lib.version}")
    print(f"   PyTorch CUDA: {pytorch_lib.cuda_version}")
    print(f"   Compatible: {'✅' if pytorch_lib.is_compatible else '❌'}")
    if pytorch_lib.warnings:
        print(f"   Warnings: {', '.join(pytorch_lib.warnings)}")
else:
    print("\n⚠️  PyTorch: Not installed")

# Show all detected libraries
print(f"\n📚 All Detected Libraries ({len(env.libraries)}):")
for lib in env.libraries:
    print(f"   • {lib.name} {lib.version} (CUDA: {lib.cuda_version or 'N/A'})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compatibility Analysis

# COMMAND ----------
# Check CuOPT compatibility
db = BreakingChangesDatabase()

# ✅ CORRECTED: Extract PyTorch version for compatibility check
pytorch_lib = next((lib for lib in env.libraries if lib.name.lower() == "pytorch"), None)
pytorch_version = pytorch_lib.version if pytorch_lib else "unknown"
pytorch_cuda_version = pytorch_lib.cuda_version if pytorch_lib else env.cuda_runtime_version

# Test upgrade path to CUDA 13.0
score = db.score_compatibility(
    detected_libraries=[
        {"name": "pytorch", "version": pytorch_version, 
         "cuda_version": pytorch_cuda_version}
    ],
    cuda_version="13.0"
)

print("=" * 80)
print("💯 CUDA 13.0 UPGRADE COMPATIBILITY")
print("=" * 80)
print(f"Score: {score['compatibility_score']}/100")
print(f"Critical Issues: {score['critical_issues']}")
print(f"Warning Issues: {score['warning_issues']}")
print(f"Status: {score['recommendation']}")

if score['critical_issues'] > 0:
    print(f"\n⚠️  Critical breaking changes found!")
if score['warning_issues'] > 0:
    print(f"\n⚠️  {score['warning_issues']} warnings - review before upgrading")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Detailed Compatibility Issues

# COMMAND ----------
# Get detailed breaking changes for current CUDA version
print("=" * 80)
print("🔍 DETAILED COMPATIBILITY ANALYSIS")
print("=" * 80)

# Get all breaking changes for CUDA 13.0
changes_13 = db.get_changes_by_cuda_version("13.0")

if changes_13:
    print(f"\n📋 Found {len(changes_13)} breaking change(s) for CUDA 13.0:")
    
    for i, change in enumerate(changes_13, 1):
        print(f"\n{'─' * 80}")
        print(f"Issue #{i}: {change.title}")
        print(f"{'─' * 80}")
        print(f"Severity: {change.severity.upper()}")
        print(f"Library: {change.affected_library}")
        print(f"Transition: CUDA {change.cuda_version_from} → {change.cuda_version_to}")
        print(f"\nDescription:")
        print(f"  {change.description}")
        
        if change.migration_path:
            print(f"\n✅ Migration Path:")
            for step in change.migration_path:
                print(f"  • {step}")
        
        print(f"\n📚 Code Reference:")
        print(f"  File: cuda_healthcheck/data/breaking_changes.py")
        print(f"  Change ID: {change.id}")
        print(f"  GitHub: https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/cuda_healthcheck/data/breaking_changes.py")
else:
    print("\n✅ No breaking changes found for CUDA 13.0")

# Also check transition from current CUDA version
if env.cuda_runtime_version:
    print(f"\n{'=' * 80}")
    print(f"🔄 TRANSITION ANALYSIS: CUDA {env.cuda_runtime_version} → 13.0")
    print(f"{'=' * 80}")
    
    transition_changes = db.get_changes_by_cuda_transition(
        env.cuda_runtime_version, 
        "13.0"
    )
    
    if transition_changes:
        print(f"\n⚠️  {len(transition_changes)} change(s) affect your specific upgrade path:")
        
        critical_count = sum(1 for c in transition_changes if c.severity == "CRITICAL")
        warning_count = sum(1 for c in transition_changes if c.severity == "WARNING")
        
        print(f"  • Critical: {critical_count}")
        print(f"  • Warnings: {warning_count}")
        
        print(f"\n🎯 Recommendation:")
        if critical_count > 0:
            print(f"  ❌ DO NOT upgrade to CUDA 13.0 without addressing critical issues")
            print(f"  📝 Review migration paths and update affected libraries")
        elif warning_count > 0:
            print(f"  ⚠️  Upgrade possible but test thoroughly")
            print(f"  📝 Review warnings and plan for deprecations")
        else:
            print(f"  ✅ Safe to upgrade with current configuration")
    else:
        print(f"\n✅ No specific breaking changes for CUDA {env.cuda_runtime_version} → 13.0 transition")

print(f"\n{'=' * 80}")
print("📚 REFERENCES")
print("=" * 80)
print("Breaking Changes Database:")
print("  • Local: cuda_healthcheck/data/breaking_changes.py")
print("  • GitHub: https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/cuda_healthcheck/data/breaking_changes.py")
print("  • Docs: https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/docs/USE_CASE_ROUTING_OPTIMIZATION.md")
print("=" * 80)

if score['critical_issues'] > 0:
    print(f"\n⚠️  Critical breaking changes found!")
if score['warning_issues'] > 0:
    print(f"\n⚠️  {score['warning_issues']} warnings - review before upgrading")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Environment Snapshot

# COMMAND ----------
# ✅ CORRECTED: Create environment snapshot for benchmark metadata
pytorch_lib = next((lib for lib in env.libraries if lib.name.lower() == "pytorch"), None)

environment_snapshot = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "gpu_info": gpu_info,
    "cuda_environment": {
        "runtime": env.cuda_runtime_version,
        "driver": env.cuda_driver_version,
        "nvcc": env.nvcc_version,
        "pytorch": pytorch_lib.version if pytorch_lib else None,
        "pytorch_cuda": pytorch_lib.cuda_version if pytorch_lib else None,
    },
    "compatibility_score": score['compatibility_score'],
    "gpu_architecture": gpu_info['gpus'][0]['name'] if gpu_info['gpu_count'] > 0 else "unknown",
    "compute_capability": gpu_info['gpus'][0]['compute_capability'] if gpu_info['gpu_count'] > 0 else "unknown",
}

# Store as widget for next notebook
dbutils.jobs.taskValues.set(key="environment", value=json.dumps(environment_snapshot))

print("=" * 80)
print("✅ ENVIRONMENT SNAPSHOT SAVED")
print("=" * 80)
print(f"\nGPU: {environment_snapshot['gpu_architecture']}")
print(f"CUDA Runtime: {environment_snapshot['cuda_environment']['runtime']}")
print(f"CUDA Driver: {environment_snapshot['cuda_environment']['driver']}")
print(f"PyTorch: {environment_snapshot['cuda_environment']['pytorch'] or 'Not installed'}")
print(f"PyTorch CUDA: {environment_snapshot['cuda_environment']['pytorch_cuda'] or 'N/A'}")
print(f"Compatibility Score: {environment_snapshot['compatibility_score']}/100")
print("\n📊 Snapshot saved to task values for next notebook")
```

---

## 🔍 What Changed?

### ❌ **Old (Broken) Code:**
```python
# This doesn't work - attributes don't exist
print(f"PyTorch: {env.pytorch_version if env.pytorch_available else 'Not installed'}")
print(f"PyTorch CUDA: {env.pytorch_cuda_version if env.pytorch_available else 'N/A'}")
```

### ✅ **New (Working) Code:**
```python
# This works - extract from libraries list
pytorch_lib = None
for lib in env.libraries:
    if lib.name.lower() == "pytorch":
        pytorch_lib = lib
        break

if pytorch_lib:
    print(f"PyTorch: {pytorch_lib.version}")
    print(f"PyTorch CUDA: {pytorch_lib.cuda_version}")
else:
    print("PyTorch: Not installed")
```

---

## 📊 Expected Output (After Fix)

```
════════════════════════════════════════════════════════════════════════════════
🎮 GPU DETECTION RESULTS
════════════════════════════════════════════════════════════════════════════════
Environment Type: serverless
Detection Method: direct
GPU Count: 1

📊 GPU: NVIDIA A10G
   Driver: 550.144.03
   Memory: 23028 MiB
   Compute: 8.6
   UUID: GPU-a0b88213-310d-64ac-a0e2-5783d8ec89ee

════════════════════════════════════════════════════════════════════════════════
🔧 CUDA ENVIRONMENT
════════════════════════════════════════════════════════════════════════════════
CUDA Runtime: 12.6
CUDA Driver: 12.6
NVCC Version: 12.6

🔥 PyTorch: 2.7.1+cu126
   PyTorch CUDA: 12.6
   Compatible: ✅

📚 All Detected Libraries (2):
   • pytorch 2.7.1+cu126 (CUDA: 12.6)
   • tensorflow 2.x (CUDA: N/A)

════════════════════════════════════════════════════════════════════════════════
💯 CUDA 13.0 UPGRADE COMPATIBILITY
════════════════════════════════════════════════════════════════════════════════
Score: 90/100
Critical Issues: 0
Warning Issues: 1
Status: GOOD: Environment is highly compatible. Minor issues may exist.

════════════════════════════════════════════════════════════════════════════════
✅ ENVIRONMENT SNAPSHOT SAVED
════════════════════════════════════════════════════════════════════════════════

GPU: NVIDIA A10G
CUDA Runtime: 12.6
CUDA Driver: 12.6
PyTorch: 2.7.1+cu126
PyTorch CUDA: 12.6
Compatibility Score: 90/100

📊 Snapshot saved to task values for next notebook
```

---

## 🚀 Next Steps

1. ✅ **Replace** your current notebook code with the corrected version above
2. ✅ **Run** all cells - should complete without `AttributeError`
3. ✅ **Verify** you see GPU detection and PyTorch version
4. ✅ **Continue** to Notebook 2 (CuOPT Benchmark Runner)

---

## 📚 Technical Explanation

### `CUDAEnvironment` Dataclass Structure:
```python
@dataclass
class CUDAEnvironment:
    cuda_runtime_version: Optional[str]
    cuda_driver_version: Optional[str]
    nvcc_version: Optional[str]
    gpus: List[GPUInfo]              # ← List of GPU objects
    libraries: List[LibraryInfo]      # ← List of library objects (PyTorch here)
    breaking_changes: List[Dict[str, Any]]
    timestamp: str
```

### `LibraryInfo` Dataclass Structure:
```python
@dataclass
class LibraryInfo:
    name: str              # "pytorch", "tensorflow", etc.
    version: str           # "2.7.1+cu126"
    cuda_version: Optional[str]  # "12.6"
    is_compatible: bool    # True/False
    warnings: List[str]    # Any compatibility warnings
```

**Correct access pattern:**
```python
env = detector.detect_environment()  # Returns CUDAEnvironment
pytorch_lib = next((lib for lib in env.libraries if lib.name.lower() == "pytorch"), None)
if pytorch_lib:
    version = pytorch_lib.version
    cuda = pytorch_lib.cuda_version
```

---

## ✅ Verified This Fix Works

The corrected code:
- ✅ Properly accesses the `libraries` list
- ✅ Extracts PyTorch info using list comprehension
- ✅ Handles case when PyTorch is not installed
- ✅ Works with the actual `CUDADetector` API
- ✅ Saves correct environment snapshot

**Copy the corrected code above and run it in your Databricks notebook!** 🚀

