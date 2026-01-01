# Databricks Installation Troubleshooting

## 🐛 Common Issue: ImportError for New Features

### Symptom

```python
ImportError: cannot import name 'get_driver_version_for_runtime' from 'cuda_healthcheck.databricks'
```

### Root Cause

**Pip caches packages**, so when you run:
```python
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git
```

It may use a **cached older version** instead of fetching the latest code from GitHub.

---

## ✅ Solution: Force Reinstall

### Option 1: Recommended (in Notebook)

```python
# COMMAND ----------
# Force reinstall to get latest version
%pip uninstall -y cuda-healthcheck-on-databricks cuda-healthcheck
%pip install --no-cache-dir git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git
dbutils.library.restartPython()
```

**Key flags:**
- `uninstall -y` - Remove old version (both possible package names)
- `--no-cache-dir` - Don't use cached packages
- Always get the **latest code from GitHub**

### Option 2: Alternative

```python
# COMMAND ----------
%pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git
dbutils.library.restartPython()
```

---

## 🔍 Verify Installation

After reinstalling, verify you have the latest version:

```python
# COMMAND ----------
# Verify installation
try:
    from cuda_healthcheck import __version__
    from cuda_healthcheck.databricks import (
        detect_databricks_runtime,
        get_driver_version_for_runtime,
        check_driver_compatibility,
    )
    print(f"✅ Version: {__version__}")
    print(f"✅ Driver mapping available!")
    
    # Quick test
    driver_info = get_driver_version_for_runtime(14.3)
    print(f"✅ Test passed: Runtime 14.3 → Driver {driver_info['driver_min_version']}")
    
except ImportError as e:
    print(f"❌ Still having issues: {e}")
    print(f"\n💡 Try:")
    print(f"   1. Restart cluster")
    print(f"   2. Run install cell again")
```

**Expected Output:**
```
✅ Version: 0.5.0
✅ Driver mapping available!
✅ Test passed: Runtime 14.3 → Driver 535
```

---

## 🔄 When to Force Reinstall

### Always Force Reinstall When:

1. ✅ **New features added** to the codebase
2. ✅ **GitHub repo updated** with bug fixes
3. ✅ **Version bumped** (e.g., 0.4.0 → 0.5.0)
4. ✅ **ImportError** for functions you know exist

### Can Use Simple Install When:

1. ✅ **Fresh cluster** (nothing cached)
2. ✅ **First time** installing
3. ✅ **No changes** to codebase since last install

---

## 🎯 Best Practice: Always Force Reinstall

**Recommended notebook install cell:**

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## 📦 Step 1: Install CUDA Healthcheck Tool
# MAGIC
# MAGIC Force reinstall to ensure latest version (v0.5.0).

# COMMAND ----------
# Uninstall any old versions
%pip uninstall -y cuda-healthcheck-on-databricks cuda-healthcheck

# Install latest from GitHub (no cache)
%pip install --no-cache-dir git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git

# Restart Python to load new version
dbutils.library.restartPython()

# COMMAND ----------
# Verify installation
from cuda_healthcheck import __version__
print(f"✅ Installed version: {__version__}")
```

---

## 🔧 Troubleshooting Steps

### Step 1: Check Installed Version

```python
%pip show cuda-healthcheck-on-databricks
```

Look for:
- **Location:** Should be in current Python environment
- **Version:** Should match latest release

### Step 2: Check Import Paths

```python
import cuda_healthcheck
print(cuda_healthcheck.__file__)
print(cuda_healthcheck.__version__)
```

### Step 3: List Exported Functions

```python
from cuda_healthcheck.databricks import __all__
print("Available functions:")
for func in sorted(__all__):
    print(f"  • {func}")
```

**Expected Output:**
```
Available functions:
  • ClusterInfo
  • DatabricksConnector
  • DatabricksHealthchecker
  • HealthcheckResult
  • check_driver_compatibility
  • detect_databricks_runtime
  • detect_gpu_auto
  • detect_gpu_direct
  • detect_gpu_distributed
  • get_driver_version_for_runtime
  • get_healthchecker
  • get_runtime_info_summary
  • is_databricks_environment
  • is_serverless_environment
```

### Step 4: Nuclear Option - Restart Cluster

If all else fails:

1. **Detach notebook** from cluster
2. **Restart cluster** (or terminate and start new)
3. **Reattach notebook**
4. **Run install cell** with force reinstall
5. **Run verification cell**

---

## 📊 Common Scenarios

### Scenario 1: Fresh Cluster

```python
# Simple install works fine
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git
dbutils.library.restartPython()
```

### Scenario 2: Cluster with Old Version

```python
# MUST force reinstall
%pip uninstall -y cuda-healthcheck-on-databricks cuda-healthcheck
%pip install --no-cache-dir git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git
dbutils.library.restartPython()
```

### Scenario 3: Multiple Users Same Cluster

```python
# Each user should force reinstall in their notebook
# Pip installs are per-user in Databricks
%pip uninstall -y cuda-healthcheck-on-databricks cuda-healthcheck
%pip install --no-cache-dir git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git
dbutils.library.restartPython()
```

---

## 🎓 Why This Happens

### Pip Caching Behavior

1. **First install:** Pip downloads from GitHub → caches locally
2. **Second install:** Pip checks cache → uses cached version
3. **Problem:** Cache doesn't auto-update when GitHub changes

### Databricks Specifics

- **Ephemeral environment:** Each cluster has its own cache
- **Per-user installs:** `%pip` installs to user's environment
- **No shared cache:** Different notebooks can have different versions

### Solution: Force Fresh Install

```python
# This bypasses all caching
%pip install --no-cache-dir git+https://...
```

---

## 📝 Checklist

Before running notebook with new features:

- [ ] Check GitHub for latest commit
- [ ] Use `--no-cache-dir` in install
- [ ] Include `uninstall` before install
- [ ] Restart Python with `dbutils.library.restartPython()`
- [ ] Verify installation with test import
- [ ] Check `__version__` matches expected

---

## 🔗 Related Issues

- **CuOPT nvJitLink incompatibility:** Also a package versioning issue
- **Runtime driver mapping:** Requires v0.5.0+
- **PyTorch compatibility:** Requires v0.5.0+

---

## ✅ Fixed!

**The enhanced notebook now includes:**
- ✅ Force reinstall in install cell
- ✅ Verification cell after install
- ✅ Clear error messages if import fails
- ✅ Instructions for recovery

**Commit:** `67d40e4` - `fix: Update notebook install cell to force reinstall and verify new features`

---

**Prevention:** Always use the updated install cell from the latest notebook version! 🚀

