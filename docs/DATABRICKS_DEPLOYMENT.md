# Databricks Deployment Guide

This guide explains how to deploy and run the CUDA Healthcheck Tool on Databricks GPU clusters.

**Supports:**
- ✅ **Classic ML Runtime** clusters (driver + workers)
- ✅ **Serverless GPU Compute** (single-user, no SparkContext)

---

## 🎯 Quick Start

### Choose Your Runtime:

**Classic ML Runtime** → Use `databricks_healthcheck.py`  
**Serverless GPU Compute** → Use `databricks_healthcheck_serverless.py`

Not sure? The tool **auto-detects** and uses the right method!

### 1. Import the Notebook

**Option A: Direct URL Import**
1. In Databricks, go to **Workspace** → **Import**
2. Select **URL**
3. Paste: `https://raw.githubusercontent.com/TavnerJC/cuda-healthcheck-1.0/main/notebooks/databricks_healthcheck.py`
4. Click **Import**

**Option B: Clone the Repository**
1. In Databricks, go to **Repos** → **Add Repo**
2. Git URL: `https://github.com/TavnerJC/cuda-healthcheck-1.0`
3. Navigate to `notebooks/databricks_healthcheck.py`

### 2. Create a GPU Cluster

**Minimum Requirements:**
- **Runtime:** Databricks Runtime 13.3 LTS ML or higher
- **Instance Type:** GPU-enabled (g5.xlarge, g5.4xlarge, p3.2xlarge, etc.)
- **Python:** 3.10+

**Example Cluster Configuration:**
```
Cluster Mode: Standard
Databricks Runtime: 13.3 LTS ML (includes Apache Spark 3.4.1, GPU, Scala 2.12)
Worker Type: g5.4xlarge (1 GPU, 16 vCPUs, 64 GB RAM)
Workers: 1-4 (autoscaling)
Driver Type: i3.xlarge (no GPU needed)
```

### 3. Run the Notebook

1. Attach the notebook to your GPU cluster (or serverless compute)
2. **Run Cell 1** (`%pip install git+https://...`)
   - ⚠️ **You'll see a red note:** "Note: you may need to restart the kernel using %restart_python or dbutils.library.restartPython()"
   - ✅ **This is NORMAL and EXPECTED!** It means the package installed successfully.
3. **Run Cell 2** (`dbutils.library.restartPython()`)
   - ⏸️ The notebook will pause for ~10 seconds while Python restarts
   - ✅ After restart, all variables are cleared (expected behavior)
   - ⚠️ **Do NOT re-run Cell 1** after the restart
4. **Run Cell 3+** to perform GPU detection and analysis
5. Review the output

> **💡 Tip:** The restart is necessary because Python needs to reload its module cache to recognize the newly installed package. Without the restart, you'll get `ModuleNotFoundError: No module named 'cuda_healthcheck'`.

---

## 📊 Classic vs Serverless: Key Differences

### Classic ML Runtime Clusters

**Architecture:**
```
┌─────────────────┐
│  Driver Node    │  ← Notebooks run here
│  (CPU only)     │  ← Package installed here
└────────┬────────┘
         │
    ┌────┴────┬─────────┬─────────┐
    │         │         │         │
┌───▼────┐ ┌──▼─────┐ ┌─▼──────┐ ┌─▼──────┐
│Worker 1│ │Worker 2│ │Worker 3│ │Worker 4│ ← GPUs here!
│(GPU)   │ │(GPU)   │ │(GPU)   │ │(GPU)   │
└────────┘ └────────┘ └────────┘ └────────┘
```

**Characteristics:**
- ✅ Multiple worker nodes with GPUs
- ✅ Distributed Spark execution
- ✅ `sparkContext` access available
- ✅ Scales to many GPUs
- ⚠️ Requires Spark-based GPU detection

**Use Case:** Large-scale distributed ML training

---

### Serverless GPU Compute

**Architecture:**
```
┌─────────────────────┐
│  Single Process     │  ← Everything runs here
│  (with GPU)         │  ← Direct GPU access
│  No SparkContext    │  ← Simplified model
└─────────────────────┘
```

**Characteristics:**
- ✅ Single-user execution
- ✅ GPU directly accessible
- ✅ Faster startup
- ✅ Simpler architecture
- ❌ No `sparkContext` access
- ⚠️ Limited to single GPU per process

**Limitations:**
- Cannot access `sc = spark.sparkContext`
- Cannot use RDD operations
- No distributed execution patterns

**Use Case:** Single-user notebooks, rapid prototyping

**Learn More:** [Databricks Serverless Limitations](https://docs.databricks.com/release-notes/serverless.html#limitations)

---

## 🤖 Auto-Detection (Recommended)

The tool **automatically detects** your environment and uses the correct method:

```python
from cuda_healthcheck.databricks import detect_gpu_auto, is_serverless_environment

# Check environment
if is_serverless_environment():
    print("📍 Running on Serverless GPU Compute")
else:
    print("📍 Running on Classic ML Runtime")

# Auto-detect GPUs (works everywhere!)
gpu_info = detect_gpu_auto()

if gpu_info['success']:
    print(f"✅ Found {gpu_info.get('gpu_count', 0)} GPU(s)")
    print(f"   Method: {gpu_info['method']}")  # 'direct' or 'distributed'
    print(f"   Environment: {gpu_info['environment']}")  # 'serverless' or 'classic'
```

---

## 📊 What Gets Detected

### Cell 3: GPU Detection
- Physical GPU count and models
- CUDA driver version
- GPU memory
- Compute capability
- Number of Spark executors

### Cell 4: Breaking Changes
- PyTorch compatibility issues
- TensorFlow compatibility issues
- RAPIDS/cuDF compatibility issues
- CUDA version transition risks
- Compatibility scores (0-100)

---

## 🏗️ Architecture

### Driver vs Worker Nodes

Databricks clusters have a **driver-worker architecture**:

```
┌─────────────────┐
│  Driver Node    │  ← Notebooks run here (usually no GPU)
│  (i3.xlarge)    │  ← Package installed here via %pip
└────────┬────────┘
         │
    ┌────┴────┬─────────┬─────────┐
    │         │         │         │
┌───▼────┐ ┌──▼─────┐ ┌─▼──────┐ ┌─▼──────┐
│Worker 1│ │Worker 2│ │Worker 3│ │Worker 4│ ← GPUs are here!
│(g5.4xl)│ │(g5.4xl)│ │(g5.4xl)│ │(g5.4xl)│ ← 16 executors per worker
│1x A10G │ │1x A10G │ │1x A10G │ │1x A10G │
└────────┘ └────────┘ └────────┘ └────────┘
```

**Key Points:**
- `%pip install` only installs on the **driver**
- GPUs are on the **workers**
- We use Spark to run detection on workers
- Results are collected back to the driver

---

## 🔧 Advanced: Full Distributed Healthcheck

For complete healthcheck functionality on workers (not just GPU detection), install the package **cluster-wide**:

### Method 1: Cluster Libraries (Recommended)

1. Go to your cluster configuration
2. Click **Libraries** tab
3. Click **Install New** → **PyPI**
4. Enter: `git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git`
5. Click **Install**
6. **Restart the cluster**

Now you can run the full `DatabricksHealthchecker` on workers!

### Method 2: Init Script

Create an init script to install on cluster startup:

```bash
#!/bin/bash
pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git
```

Upload to DBFS and configure in cluster settings.

---

## 📝 Example Output

### Successful Detection

```
================================================================================
🖥️  DRIVER NODE
================================================================================
Driver: No GPU detected (expected for driver node)

================================================================================
🎮 WORKER NODES - GPU DETECTION
================================================================================
📊 Cluster Configuration:
   Spark Executors: 16
   Unique Worker Nodes: 1

📍 Worker Node 1: ip-10-0-1-234.ec2.internal
   Physical GPUs: 1
      GPU 0: NVIDIA A10G
         Driver: 535.161.07
         Memory: 23028 MiB
         Compute Capability: 8.6

================================================================================
✅ ACTUAL PHYSICAL GPUs in cluster: 1
   (Detected 16 times - once per Spark executor)
================================================================================
```

### Compatibility Analysis

```
================================================================================
🔍 CUDA BREAKING CHANGES ANALYSIS
================================================================================

📦 PyTorch Breaking Changes:
   ✅ Found 2 PyTorch breaking changes

📦 TensorFlow Breaking Changes:
   ✅ Found 2 TensorFlow breaking changes

🔄 CUDA Version Transition Analysis:
   CUDA 11.8 → 12.0: 2 breaking changes
   CUDA 12.0 → 13.0: 6 breaking changes

================================================================================
💯 COMPATIBILITY SCORING
================================================================================

📊 CUDA 12.0 Compatibility:
   Score: 100/100
   Critical: 0 | Warnings: 0
   Status: GOOD: Environment is highly compatible.

📊 CUDA 13.0 Compatibility:
   Score: 40/100
   Critical: 2 | Warnings: 0
   Status: CRITICAL: Breaking changes detected. Test before upgrading!

================================================================================
```

---

## ⚠️ Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'cuda_healthcheck'"

**Cause:** You tried to import the package before installing it, or you didn't restart Python after installation.

**Solution:**
```python
# Cell 1: Install
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git

# Cell 2: Restart (REQUIRED!)
dbutils.library.restartPython()

# Cell 3: Now import works
from cuda_healthcheck import CUDADetector
```

### Issue 2: Red warning note after %pip install

**Message you'll see:**  
> "Note: you may need to restart the kernel using %restart_python or dbutils.library.restartPython()"

**Status:** ✅ **This is COMPLETELY NORMAL!** It means the package installed successfully.

**What to do:**  
1. ✅ Celebrate - installation worked!
2. ✅ Run `dbutils.library.restartPython()` in the next cell
3. ⚠️ Do NOT re-run the install cell after restarting
4. ✅ Continue with imports in Cell 3+

**Why this happens:** Python needs to restart to recognize the newly installed package. Without the restart, you'll get `ModuleNotFoundError`.

### Issue 3: "No GPU detected on driver"

**Expected!** The driver node typically doesn't have a GPU. GPUs are on worker nodes.

### Issue 4: "Package import fails on workers"

**Solution:** Install package cluster-wide (see Advanced section above).

### Issue 5: "16 GPUs detected but only 1 physical GPU"

**Expected!** Each Spark executor reports the GPU. The code deduplicates by UUID to show actual physical GPUs.

### Issue 6: "Cell hangs with py4j messages"

**Cause:** Trying to import package on workers when it's only installed on driver.  
**Solution:** Use the provided notebook which avoids package imports on workers for basic detection.

### Issue 7: "Variables undefined after restart"

**Status:** ✅ **This is EXPECTED Python behavior!**

When you run `dbutils.library.restartPython()`, all variables are cleared. This is how Python restarts work.

**What to do:** Don't try to use variables from Cell 1 in Cell 3+. The restart clears everything.

### Issue 8: Serverless: "[JVM_ATTRIBUTE_NOT_SUPPORTED] ... 'sparkContext'"

**Cause:** Trying to use Spark/SparkContext on Serverless GPU Compute (not supported).

**Solution:** Use `databricks_healthcheck_serverless.py` notebook which uses `detect_gpu_auto()` for serverless-compatible detection.

---

## 🎯 Use Cases

### 1. Pre-Deployment Validation
Run before deploying ML models to verify CUDA compatibility.

### 2. Cluster Configuration Audit
Validate that your cluster has the expected GPU configuration.

### 3. Framework Upgrade Planning
Check compatibility scores before upgrading PyTorch, TensorFlow, or CUDA.

### 4. Breaking Changes Detection
Identify critical issues before they cause production failures.

### 5. Multi-Cluster Comparison
Run on different clusters to compare configurations.

---

## 📚 Additional Resources

- [Main README](../README.md) - Full documentation
- [API Reference](../docs/API_REFERENCE.md) - Detailed API docs
- [Local Testing](../TESTING_AND_NOTEBOOKS_SUMMARY.md) - Run tests locally
- [CI/CD](../docs/CICD.md) - GitHub Actions workflows

---

## 💡 Tips

1. **Run regularly:** Add to your cluster startup routine
2. **Before upgrades:** Always check compatibility scores
3. **Save results:** Export to Delta table for historical tracking
4. **Team sharing:** Share the notebook with your ML team
5. **Custom checks:** Extend the notebook for your specific needs

---

## 🆘 Support

- **GitHub Issues:** [Report bugs or request features](https://github.com/TavnerJC/cuda-healthcheck-1.0/issues)
- **Documentation:** Check the [main README](../README.md)
- **Examples:** See the [notebooks folder](../notebooks/)

---

**Happy GPU Healthchecking!** 🎉

