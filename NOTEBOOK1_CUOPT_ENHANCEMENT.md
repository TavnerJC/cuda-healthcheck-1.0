# 🔧 Enhanced Notebook 1: Add CuOPT Compatibility Check

## 📋 Addition to Notebook 1

Add this new section after the "Detect CUDA & Libraries" section:

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## 🚨 CuOPT Compatibility Check (Critical for Routing Optimization)

# COMMAND ----------
# Check for CuOPT-specific compatibility issues
print("=" * 80)
print("🔍 CUOPT COMPATIBILITY ANALYSIS")
print("=" * 80)

# Extract CuOPT from detected libraries
cuopt_lib = None
for lib in env.libraries:
    if lib.name.lower() == "cuopt":
        cuopt_lib = lib
        break

if cuopt_lib:
    print(f"\n📦 CuOPT Status:")
    print(f"   Version: {cuopt_lib.version}")
    print(f"   CUDA Version: {cuopt_lib.cuda_version}")
    print(f"   Compatible: {cuopt_lib.is_compatible}")
    
    if not cuopt_lib.is_compatible:
        print(f"\n❌ CUOPT COMPATIBILITY ISSUES DETECTED")
        print(f"\n⚠️  Warnings ({len(cuopt_lib.warnings)}):")
        for warning in cuopt_lib.warnings:
            print(f"   • {warning}")
        
        # Check for the specific nvJitLink issue
        nvjitlink_issue = any("nvJitLink" in w or "12.9" in w for w in cuopt_lib.warnings)
        
        if nvjitlink_issue:
            print(f"\n{'=' * 80}")
            print(f"🚨 CRITICAL: CuOPT nvJitLink Incompatibility Detected")
            print(f"{'=' * 80}")
            print(f"")
            print(f"This is a KNOWN breaking change tracked by the CUDA Healthcheck Tool:")
            print(f"")
            print(f"Issue: CuOPT 25.12+ requires nvidia-nvjitlink-cu12>=12.9.79")
            print(f"       Databricks ML Runtime 16.4 provides nvidia-nvjitlink-cu12 12.4.127")
            print(f"")
            print(f"Impact:")
            print(f"   • CuOPT library fails to load (libcuopt.so error)")
            print(f"   • GPU-accelerated routing optimization UNAVAILABLE")
            print(f"   • Users CANNOT upgrade nvJitLink (runtime-controlled)")
            print(f"")
            print(f"Recommended Actions:")
            print(f"   1. Report to Databricks:")
            print(f"      https://github.com/databricks-industry-solutions/routing/issues")
            print(f"")
            print(f"   2. Use alternative solver:")
            print(f"      pip install ortools")
            print(f"")
            print(f"   3. Wait for Databricks ML Runtime 17.0+ (with CUDA 12.9+ support)")
            print(f"")
            print(f"More Info:")
            print(f"   • Breaking change ID: cuopt-nvjitlink-databricks-ml-runtime")
            print(f"   • Tracked in: cuda_healthcheck/data/breaking_changes.py")
            print(f"   • GitHub: https://github.com/TavnerJC/cuda-healthcheck-1.0")
            print(f"{'=' * 80}")
            
            # Add to environment summary for Notebook 2
            cuopt_incompatible = True
    else:
        print(f"\n✅ CuOPT is compatible and working!")
        cuopt_incompatible = False
else:
    print(f"\n📦 CuOPT: Not installed")
    cuopt_incompatible = False

# COMMAND ----------
# MAGIC %md
# MAGIC ## Check Databricks Runtime CUDA Components

# COMMAND ----------
# Check specific CUDA component versions in Databricks runtime
import subprocess

print("=" * 80)
print("🔍 DATABRICKS RUNTIME CUDA COMPONENTS")
print("=" * 80)

# Check nvJitLink version specifically
try:
    result = subprocess.run(
        ["pip", "show", "nvidia-nvjitlink-cu12"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        print("\n📦 nvidia-nvjitlink-cu12:")
        for line in result.stdout.split('\n'):
            if line.startswith('Version:') or line.startswith('Location:'):
                print(f"   {line}")
        
        # Extract version
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                nvjitlink_version = line.split(':')[1].strip()
                
                print(f"\n🔍 Version Analysis:")
                if nvjitlink_version.startswith('12.4'):
                    print(f"   ❌ Version {nvjitlink_version} is INCOMPATIBLE with CuOPT 25.12+")
                    print(f"   ✅ Requires: 12.9.79 or later")
                    print(f"   ⚠️  This is a Databricks Runtime limitation")
                elif nvjitlink_version.startswith('12.9') or nvjitlink_version.startswith('13.'):
                    print(f"   ✅ Version {nvjitlink_version} is COMPATIBLE with CuOPT 25.12+")
                else:
                    print(f"   ⚠️  Version {nvjitlink_version} - compatibility unknown")
                break
    else:
        print("\n⚠️  nvidia-nvjitlink-cu12 not found")
        
except Exception as e:
    print(f"\n❌ Error checking nvJitLink: {e}")

# Check other key CUDA components
cuda_components = [
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cusolver-cu12",
]

print(f"\n📦 Other CUDA Components:")
for component in cuda_components:
    try:
        result = subprocess.run(
            ["pip", "show", component],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':')[1].strip()
                    print(f"   {component}: {version}")
                    break
        else:
            print(f"   {component}: Not found")
    except Exception:
        print(f"   {component}: Error checking")

print("=" * 80)
```

---

## 🎯 What This Addition Does

### **1. Detects CuOPT Installation**
- Checks if CuOPT is installed
- Validates if it can actually load (tests `libcuopt.so`)
- Reports version and compatibility status

### **2. Identifies nvJitLink Incompatibility**
- Detects the specific `__nvJitLinkGetErrorLogSize_12_9` error
- Cross-references with breaking changes database
- Provides clear explanation of the issue

### **3. Provides Actionable Guidance**
- Reports to Databricks routing repo
- Suggests OR-Tools as alternative
- Explains why user cannot fix it themselves

### **4. Validates Databricks Runtime**
- Checks installed nvJitLink version
- Compares against CuOPT requirements
- Validates other CUDA components

---

## ✅ Integration with Existing Notebook 1

This section should be inserted **after**:
- ✅ GPU Detection (Cell 3)
- ✅ CUDA Environment Detection (Cell 4)
- ✅ Library Detection (Cell 5)

And **before**:
- Compatibility Analysis (Cell 6)
- Detailed Breaking Changes (Cell 7)

---

## 📊 Expected Output

### **If CuOPT is Compatible:**
```
✅ CuOPT is compatible and working!
   Version: 25.12.0
   CUDA Version: 12.9
```

### **If CuOPT is Incompatible:**
```
❌ CUOPT COMPATIBILITY ISSUES DETECTED

⚠️  Warnings (5):
   • CRITICAL: CuOPT failed to load due to nvJitLink version mismatch
   • CuOPT 25.12+ requires nvidia-nvjitlink-cu12>=12.9.79
   • Detected nvidia-nvjitlink-cu12 version: 12.4.127
   • ERROR: Databricks ML Runtime provides nvJitLink 12.4.x
   • Users CANNOT upgrade nvJitLink in managed Databricks runtimes

🚨 CRITICAL: CuOPT nvJitLink Incompatibility Detected

Issue: CuOPT 25.12+ requires nvidia-nvjitlink-cu12>=12.9.79
       Databricks ML Runtime 16.4 provides nvidia-nvjitlink-cu12 12.4.127

Recommended Actions:
   1. Report to Databricks:
      https://github.com/databricks-industry-solutions/routing/issues
   
   2. Use alternative solver:
      pip install ortools
```

---

## 🔗 This Connects To:

1. **Breaking Changes Database** (`breaking_changes.py`)
   - New entry: `cuopt-nvjitlink-databricks-ml-runtime`
   - Severity: CRITICAL
   - Migration path: Report to Databricks

2. **CuOPT Detection** (`detector.py`)
   - New method: `detect_cuopt()`
   - Tests library loading
   - Captures nvJitLink errors

3. **GitHub Issue Template** (Next step)
   - Pre-filled issue for Databricks routing repo
   - Includes environment details
   - Links to CUDA Healthcheck findings

---

## 📁 Files Modified

- ✅ `cuda_healthcheck/data/breaking_changes.py` - Added CuOPT breaking change
- ✅ `cuda_healthcheck/cuda_detector/detector.py` - Added `detect_cuopt()` method
- 📝 `docs/EXPERIMENT_CUOPT_BENCHMARK.md` - Enhanced Notebook 1 cells (this file)

