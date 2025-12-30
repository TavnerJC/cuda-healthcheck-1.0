# 🐛 GitHub Issue Template: CuOPT nvJitLink Incompatibility

## 📋 Template for Databricks Routing Repository

Use this template to report the CuOPT/nvJitLink incompatibility to the Databricks routing repository:

**URL:** https://github.com/databricks-industry-solutions/routing/issues

---

## ✍️ Issue Title

```
CuOPT 25.12+ incompatible with Databricks ML Runtime 16.4 (nvJitLink version mismatch)
```

---

## 📝 Issue Body

```markdown
### 🐛 Bug Description

NVIDIA CuOPT 25.12.0 fails to load on Databricks ML Runtime 16.4 due to a `nvidia-nvjitlink-cu12` version incompatibility.

### 📊 Environment

- **Databricks Runtime:** ML Runtime 16.4 (Spark 3.5.2, Scala 2.13)
- **Cluster Type:** Classic ML Runtime with GPU workers
- **GPU:** NVIDIA A10G (g5.xlarge)
- **CuOPT Version:** 25.12.0
- **nvidia-nvjitlink-cu12 (Provided by Runtime):** 12.4.127
- **nvidia-nvjitlink-cu12 (Required by CuOPT):** >=12.9.79

### 🚨 Error Message

```
RuntimeWarning: Failed to load libcuopt library: libcuopt.so. 
Error: /local_disk0/.ephemeral_nfs/envs/pythonEnv-xxx/lib/python3.12/site-packages/libcuopt/lib64/../../nvidia/cusparse/lib/libcusparse.so.12: 
undefined symbol: __nvJitLinkGetErrorLogSize_12_9, version libnvJitLink.so.12
```

### 🔍 Root Cause

1. CuOPT 25.12.0 requires `nvidia-nvjitlink-cu12 >= 12.9.79`
2. Databricks ML Runtime 16.4 pre-installs `nvidia-nvjitlink-cu12 12.4.127`
3. Users **cannot upgrade** `nvidia-nvjitlink-cu12` because:
   - Databricks runtimes have locked CUDA component versions
   - Pip upgrade attempts either fail or are overridden by runtime
   - System paths prioritize runtime-provided libraries

### 📦 Installation Steps (That Fail)

```python
# Install CuOPT (succeeds)
%pip install --extra-index-url=https://pypi.nvidia.com cuopt-server-cu12 cuopt-sh-client

# Try to upgrade nvJitLink (fails or has no effect)
%pip install --upgrade --force-reinstall nvidia-nvjitlink-cu12>=12.9.79

dbutils.library.restartPython()

# Test CuOPT (fails with RuntimeWarning)
from cuopt import routing
data_model = routing.DataModel(10, 2)  # Error: libcuopt.so fails to load
```

### 💡 Impact

- **GPU Route Optimization notebook** (`06_gpu_route_optimization.ipynb`) is **non-functional**
- Users following the Databricks routing accelerator cannot use CuOPT
- No GPU-accelerated routing optimization available on ML Runtime 16.4

### ✅ Suggested Solutions

**Option 1: Update ML Runtime 16.4**
- Upgrade `nvidia-nvjitlink-cu12` to 12.9.79+ in ML Runtime 16.4
- Ensures compatibility with latest CuOPT releases

**Option 2: Document Compatibility**
- Update routing accelerator docs to specify:
  - CuOPT 25.12+ requires ML Runtime 17.0+ (when available)
  - ML Runtime 16.4 users should use OR-Tools alternative
  - Provide OR-Tools example code

**Option 3: Pin Older CuOPT Version**
- Identify last CuOPT version compatible with nvJitLink 12.4.x
- Update notebook to use that version for ML Runtime 16.4

### 🔗 Related Resources

- **Issue Detection Tool:** [CUDA Healthcheck for Databricks](https://github.com/TavnerJC/cuda-healthcheck-1.0)
  - This issue was detected and tracked in the breaking changes database
  - Breaking Change ID: `cuopt-nvjitlink-databricks-ml-runtime`
- **NVIDIA CuOPT:** https://github.com/NVIDIA/cuopt
- **Databricks ML Runtime Docs:** https://docs.databricks.com/en/release-notes/runtime/index.html

### 📸 Screenshots

**Error in Databricks notebook:**
[Attach screenshot of RuntimeWarning and error message]

**Installed nvJitLink version:**
```python
!pip show nvidia-nvjitlink-cu12
# Version: 12.4.127
```

**CuOPT requirements:**
```python
!pip show cuopt-server-cu12 | grep -A5 Requires
# Requires: nvidia-nvjitlink-cu12>=12.9.79
```

### 🙏 Request

Please update Databricks ML Runtime 16.4 to include `nvidia-nvjitlink-cu12 >= 12.9.79` or provide guidance on CuOPT version compatibility for current runtime.

This is blocking GPU-accelerated routing optimization for ML Runtime 16.4 users.

Thank you!
```

---

## 🎯 Auto-Generated Issue Template (Python Code)

Use this code in your Databricks notebook to generate a pre-filled GitHub issue:

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## 🐛 Generate GitHub Issue for CuOPT Incompatibility

# COMMAND ----------
import subprocess
import json

# Gather environment information
environment_info = {
    "databricks_runtime": "ML Runtime 16.4",
    "cluster_type": "Classic ML Runtime with GPU workers",
    "gpu": "NVIDIA A10G (g5.xlarge)",
    "cuopt_version": "25.12.0",
}

# Check nvJitLink version
try:
    result = subprocess.run(
        ["pip", "show", "nvidia-nvjitlink-cu12"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            environment_info['nvjitlink_installed'] = line.split(':')[1].strip()
            break
except Exception:
    environment_info['nvjitlink_installed'] = "Unknown"

# Check CuOPT requirements
try:
    result = subprocess.run(
        ["pip", "show", "cuopt-server-cu12"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    in_requires = False
    for line in result.stdout.split('\n'):
        if line.startswith('Requires:'):
            in_requires = True
        if in_requires and 'nvidia-nvjitlink' in line:
            # Extract version requirement
            import re
            match = re.search(r'nvidia-nvjitlink-cu12\s*\(?\s*>=\s*([0-9.]+)', line)
            if match:
                environment_info['nvjitlink_required'] = f">={match.group(1)}"
            break
except Exception:
    environment_info['nvjitlink_required'] = ">=12.9.79"

# Generate GitHub issue URL
issue_title = "CuOPT 25.12+ incompatible with Databricks ML Runtime 16.4 (nvJitLink version mismatch)"
issue_body = f"""### 🐛 Bug Description

NVIDIA CuOPT {environment_info['cuopt_version']} fails to load on {environment_info['databricks_runtime']} due to a `nvidia-nvjitlink-cu12` version incompatibility.

### 📊 Environment

- **Databricks Runtime:** {environment_info['databricks_runtime']}
- **Cluster Type:** {environment_info['cluster_type']}
- **GPU:** {environment_info['gpu']}
- **CuOPT Version:** {environment_info['cuopt_version']}
- **nvidia-nvjitlink-cu12 (Installed):** {environment_info['nvjitlink_installed']}
- **nvidia-nvjitlink-cu12 (Required):** {environment_info['nvjitlink_required']}

### 🚨 Error Message

```
RuntimeWarning: Failed to load libcuopt library: libcuopt.so. 
Error: undefined symbol: __nvJitLinkGetErrorLogSize_12_9, version libnvJitLink.so.12
```

### 🔍 Root Cause

CuOPT {environment_info['cuopt_version']} requires `nvidia-nvjitlink-cu12 {environment_info['nvjitlink_required']}` but {environment_info['databricks_runtime']} provides version {environment_info['nvjitlink_installed']}.

Users cannot upgrade nvJitLink because Databricks runtimes have locked CUDA component versions.

### 💡 Impact

- GPU Route Optimization notebook (`06_gpu_route_optimization.ipynb`) is non-functional
- No GPU-accelerated routing optimization available on ML Runtime 16.4

### ✅ Suggested Solution

Please update {environment_info['databricks_runtime']} to include `nvidia-nvjitlink-cu12 >= 12.9.79`.

### 🔗 Related Resources

- **Issue detected by:** [CUDA Healthcheck for Databricks](https://github.com/TavnerJC/cuda-healthcheck-1.0)
- **Breaking Change ID:** `cuopt-nvjitlink-databricks-ml-runtime`
- **NVIDIA CuOPT:** https://github.com/NVIDIA/cuopt

Thank you!
"""

# URL-encode the issue
import urllib.parse
github_url = f"https://github.com/databricks-industry-solutions/routing/issues/new?title={urllib.parse.quote(issue_title)}&body={urllib.parse.quote(issue_body)}"

print("=" * 80)
print("🐛 GITHUB ISSUE GENERATOR")
print("=" * 80)
print(f"\n📋 Issue Title:")
print(f"   {issue_title}")
print(f"\n📊 Environment Summary:")
for key, value in environment_info.items():
    print(f"   {key}: {value}")
print(f"\n🔗 GitHub Issue URL (click to open):")
print(f"   {github_url}")
print(f"\n📝 Or copy this link to your browser:")
print(f"   https://github.com/databricks-industry-solutions/routing/issues/new")
print(f"\n✅ Issue body is copied below - paste into GitHub:")
print("=" * 80)
print(issue_body)
print("=" * 80)
```

---

## 📤 How to Use This Template

### **Option 1: Manual (Copy-Paste)**
1. Go to https://github.com/databricks-industry-solutions/routing/issues/new
2. Copy the issue title and body from above
3. Paste into GitHub
4. Add your screenshots
5. Submit

### **Option 2: Automated (From Notebook)**
1. Add the Python code above to a new cell in your Databricks notebook
2. Run the cell
3. Click the generated GitHub issue URL
4. Review and submit

---

## ✅ Expected Response

The Databricks team should either:

1. **Update ML Runtime 16.4** with nvidia-nvjitlink-cu12 >= 12.9.79
2. **Release ML Runtime 17.0** with updated CUDA components
3. **Document the incompatibility** and provide OR-Tools alternative
4. **Pin older CuOPT version** that works with nvJitLink 12.4.x

---

## 🔗 Related Issues

- Check if similar issues already exist before creating
- Reference this issue in your own documentation
- Link back to the CUDA Healthcheck Tool that detected it

---

## 📊 Tracking

This template is maintained as part of the CUDA Healthcheck Tool project:
- **Repository:** https://github.com/TavnerJC/cuda-healthcheck-1.0
- **File:** `GITHUB_ISSUE_TEMPLATE_CUOPT.md`
- **Breaking Change Entry:** `cuda_healthcheck/data/breaking_changes.py`

