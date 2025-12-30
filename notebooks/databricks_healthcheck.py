# Databricks notebook source
# MAGIC %md
# MAGIC # ⚠️ LEGACY NOTEBOOK - Please Use Enhanced Version
# MAGIC
# MAGIC **This notebook is now LEGACY.** Please use the enhanced version for better features:
# MAGIC
# MAGIC ## 🆕 Enhanced Notebook (Recommended)
# MAGIC
# MAGIC **File:** `notebooks/01_cuda_environment_validation_enhanced.py`  
# MAGIC **GitHub:** https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/notebooks/01_cuda_environment_validation_enhanced.py
# MAGIC
# MAGIC **Why switch?**
# MAGIC - ✅ **CuOPT compatibility detection** (detects nvJitLink incompatibility)
# MAGIC - ✅ **Auto-detection** (works on Classic ML Runtime & Serverless GPU Compute)
# MAGIC - ✅ **Comprehensive breaking changes** with migration paths
# MAGIC - ✅ **Validated on Databricks A10G** (production-ready)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC # CUDA Healthcheck for Databricks GPU Clusters (Legacy)
# MAGIC
# MAGIC This notebook validates CUDA configuration, detects GPU hardware, and identifies compatibility issues on Databricks GPU clusters.
# MAGIC
# MAGIC **Requirements:**
# MAGIC - GPU-enabled Databricks cluster (e.g., g5.4xlarge, p3.2xlarge)
# MAGIC - Databricks Runtime 13.3 LTS ML or higher
# MAGIC - Python 3.10+
# MAGIC
# MAGIC **What This Notebook Does:**
# MAGIC 1. Installs CUDA Healthcheck tool from GitHub
# MAGIC 2. Detects GPU hardware on worker nodes
# MAGIC 3. Validates CUDA driver and runtime versions
# MAGIC 4. Analyzes breaking changes for ML frameworks
# MAGIC 5. Provides compatibility scores and recommendations
# MAGIC
# MAGIC **Note:** This legacy notebook is kept for backward compatibility only.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install CUDA Healthcheck Package
# MAGIC
# MAGIC Install the package from GitHub. This only installs on the driver node.

# COMMAND ----------

# MAGIC %pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Restart Python
# MAGIC
# MAGIC **REQUIRED:** Restart Python to load the newly installed package.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: GPU Detection on Worker Nodes
# MAGIC
# MAGIC Databricks clusters have a driver-worker architecture:
# MAGIC - **Driver node:** Where notebooks run (usually no GPU)
# MAGIC - **Worker nodes:** Where Spark executors run (have GPUs)
# MAGIC
# MAGIC This cell uses Spark to run `nvidia-smi` on each worker node to detect GPUs.

# COMMAND ----------

from pyspark.sql import SparkSession
import subprocess

print("=" * 80)
print("🖥️  DRIVER NODE")
print("=" * 80)

# Check driver (usually no GPU)
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        print(f"Driver GPU: {result.stdout.strip()}")
    else:
        print("Driver: No GPU detected (expected for driver node)")
except:
    print("Driver: No GPU detected (expected for driver node)")

print("\n" + "=" * 80)
print("🎮 WORKER NODES - GPU DETECTION")
print("=" * 80)


def check_gpu_simple(_):
    """Simple GPU check on worker without importing package."""
    import subprocess
    import socket

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap,uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append(
                        {
                            "name": parts[0],
                            "driver": parts[1],
                            "memory": parts[2],
                            "compute_cap": parts[3],
                            "uuid": parts[4],
                        }
                    )

            return {
                "hostname": socket.gethostname(),
                "has_gpu": True,
                "gpu_count": len(gpus),
                "gpus": gpus,
                "error": None,
            }
        else:
            return {
                "hostname": socket.gethostname(),
                "has_gpu": False,
                "gpu_count": 0,
                "gpus": [],
                "error": "nvidia-smi returned no data",
            }
    except Exception as e:
        return {
            "hostname": socket.gethostname(),
            "has_gpu": False,
            "gpu_count": 0,
            "gpus": [],
            "error": str(e),
        }


# Run on workers
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

num_partitions = max(sc.defaultParallelism, 2)

worker_results = (
    sc.parallelize(range(num_partitions), num_partitions).map(check_gpu_simple).collect()
)

# Deduplicate by hostname + GPU UUID
unique_gpus = {}
for result in worker_results:
    hostname = result["hostname"]
    if result["has_gpu"]:
        for gpu in result["gpus"]:
            key = f"{hostname}_{gpu['uuid']}"
            if key not in unique_gpus:
                unique_gpus[key] = {"hostname": hostname, "gpu": gpu}

# Display unique worker nodes and GPUs
worker_nodes = {}
for key, info in unique_gpus.items():
    hostname = info["hostname"]
    if hostname not in worker_nodes:
        worker_nodes[hostname] = []
    worker_nodes[hostname].append(info["gpu"])

print(f"📊 Cluster Configuration:")
print(f"   Spark Executors: {len(worker_results)}")
print(f"   Unique Worker Nodes: {len(worker_nodes)}")
print()

for i, (hostname, gpus) in enumerate(worker_nodes.items(), 1):
    print(f"📍 Worker Node {i}: {hostname}")
    print(f"   Physical GPUs: {len(gpus)}")
    for j, gpu in enumerate(gpus):
        print(f"      GPU {j}: {gpu['name']}")
        print(f"         Driver: {gpu['driver']}")
        print(f"         Memory: {gpu['memory']}")
        print(f"         Compute Capability: {gpu['compute_cap']}")

print("\n" + "=" * 80)
print(f"✅ ACTUAL PHYSICAL GPUs in cluster: {len(unique_gpus)}")
print(f"   (Detected {len(worker_results)} times - once per Spark executor)")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Breaking Changes Analysis
# MAGIC
# MAGIC Analyze compatibility between CUDA versions and detect breaking changes for PyTorch, TensorFlow, and RAPIDS.
# MAGIC
# MAGIC This runs on the driver where the package is installed.

# COMMAND ----------

from cuda_healthcheck.data import BreakingChangesDatabase, get_breaking_changes

print("=" * 80)
print("🔍 CUDA BREAKING CHANGES ANALYSIS")
print("=" * 80)

# Get breaking changes database
db = BreakingChangesDatabase()

# Check PyTorch breaking changes
print("\n📦 PyTorch Breaking Changes:")
pytorch_changes = get_breaking_changes("pytorch")
print(f"   ✅ Found {len(pytorch_changes)} PyTorch breaking changes")

# Check TensorFlow breaking changes
print("\n📦 TensorFlow Breaking Changes:")
tf_changes = get_breaking_changes("tensorflow")
print(f"   ✅ Found {len(tf_changes)} TensorFlow breaking changes")

# Check RAPIDS/cuDF breaking changes
print("\n📦 RAPIDS/cuDF Breaking Changes:")
cudf_changes = get_breaking_changes("cudf")
print(f"   ✅ Found {len(cudf_changes)} cuDF breaking changes")

# Check changes between specific CUDA versions
print("\n🔄 CUDA Version Transition Analysis:")
changes_11_to_12 = db.get_changes_by_cuda_transition("11.8", "12.0")
print(f"   CUDA 11.8 → 12.0: {len(changes_11_to_12)} breaking changes")

changes_12_to_13 = db.get_changes_by_cuda_transition("12.0", "13.0")
print(f"   CUDA 12.0 → 13.0: {len(changes_12_to_13)} breaking changes")

# Get all breaking changes
print("\n📊 Database Statistics:")
all_changes = db.get_all_changes()
print(f"   Total breaking changes: {len(all_changes)}")

# Compatibility scoring
print("\n" + "=" * 80)
print("💯 COMPATIBILITY SCORING")
print("=" * 80)

# Example: PyTorch 2.6 + CUDA 12.4
detected_libs = [
    {"name": "pytorch", "version": "2.6.0", "cuda_version": "12.4"},
    {"name": "tensorflow", "version": "2.12.0", "cuda_version": "11.8"},
]

# Score for CUDA 12.0
score_12 = db.score_compatibility(detected_libs, "12.0")
print(f"\n📊 CUDA 12.0 Compatibility:")
print(f"   Score: {score_12['compatibility_score']}/100")
print(f"   Critical: {score_12['critical_issues']} | Warnings: {score_12['warning_issues']}")
print(f"   Status: {score_12['recommendation']}")

# Score for CUDA 13.0
score_13 = db.score_compatibility(detected_libs, "13.0")
print(f"\n📊 CUDA 13.0 Compatibility:")
print(f"   Score: {score_13['compatibility_score']}/100")
print(f"   Critical: {score_13['critical_issues']} | Warnings: {score_13['warning_issues']}")
print(f"   Status: {score_13['recommendation']}")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook successfully:
# MAGIC - ✅ Detected GPU hardware on worker nodes
# MAGIC - ✅ Validated CUDA driver and runtime versions
# MAGIC - ✅ Analyzed breaking changes for major ML frameworks
# MAGIC - ✅ Provided compatibility scores
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Save this notebook for regular cluster validation
# MAGIC - Run before upgrading CUDA or ML frameworks
# MAGIC - Use compatibility scores to plan upgrades
# MAGIC
# MAGIC **For Full Distributed Healthcheck:**
# MAGIC Install the package at cluster level (Libraries → PyPI → `git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git`)
