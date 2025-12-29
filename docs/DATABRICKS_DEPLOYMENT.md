# Databricks Deployment Guide

This guide explains how to deploy and run the CUDA Healthcheck Tool on Databricks GPU clusters.

## 🎯 Quick Start

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

1. Attach the notebook to your GPU cluster
2. Run all cells sequentially
3. Review the output

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

### Issue 1: "No GPU detected on driver"

**Expected!** The driver node typically doesn't have a GPU. GPUs are on worker nodes.

### Issue 2: "Package import fails on workers"

**Solution:** Install package cluster-wide (see Advanced section above).

### Issue 3: "16 GPUs detected but only 1 physical GPU"

**Expected!** Each Spark executor reports the GPU. The code deduplicates by UUID to show actual physical GPUs.

### Issue 4: "Cell hangs with py4j messages"

**Cause:** Trying to import package on workers when it's only installed on driver.  
**Solution:** Use the provided notebook which avoids package imports on workers for basic detection.

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

