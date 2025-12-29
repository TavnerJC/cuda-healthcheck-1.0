# Databricks notebook source
# MAGIC %md
# MAGIC # CUDA Healthcheck for Databricks Serverless GPU Compute
# MAGIC
# MAGIC This notebook validates CUDA configuration and detects GPU hardware on **Databricks Serverless GPU Compute**.
# MAGIC
# MAGIC **For Classic ML Runtime clusters**, use `databricks_healthcheck.py` instead.
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Databricks Serverless GPU Compute enabled
# MAGIC - GPU-enabled serverless runtime
# MAGIC - Python 3.10+
# MAGIC
# MAGIC **What This Notebook Does:**
# MAGIC 1. Installs CUDA Healthcheck tool from GitHub
# MAGIC 2. Detects GPU hardware (serverless-compatible method)
# MAGIC 3. Validates CUDA driver and runtime versions
# MAGIC 4. Analyzes breaking changes for ML frameworks
# MAGIC 5. Provides compatibility scores and recommendations
# MAGIC
# MAGIC **Key Differences from Classic:**
# MAGIC - No SparkContext usage (not available on serverless)
# MAGIC - Direct GPU detection on current process
# MAGIC - Simpler, faster execution
# MAGIC - Single-user model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install CUDA Healthcheck Package
# MAGIC
# MAGIC Install the package from GitHub.

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
# MAGIC ## Step 3: Environment Detection & GPU Discovery
# MAGIC
# MAGIC Serverless GPU Compute provides direct GPU access without Spark distribution.
# MAGIC This cell uses **auto-detection** that works on both serverless and classic clusters.

# COMMAND ----------

from cuda_healthcheck.databricks import detect_gpu_auto, is_serverless_environment

print("=" * 80)
print("🌟 DATABRICKS ENVIRONMENT DETECTION")
print("=" * 80)

# Detect environment
if is_serverless_environment():
    print("📍 Environment: Serverless GPU Compute")
    print("   • Single-user execution model")
    print("   • Direct GPU access (no Spark distribution)")
    print("   • SparkContext not available (expected)")
else:
    print("📍 Environment: Classic ML Runtime")
    print("   • Driver-worker architecture")
    print("   • Distributed Spark execution")
    print("   • Note: Consider using databricks_healthcheck.py for classic clusters")

print("\n" + "=" * 80)
print("🎮 GPU DETECTION")
print("=" * 80)

# Auto-detect GPUs (works on both serverless and classic!)
gpu_info = detect_gpu_auto()

if gpu_info["success"]:
    print(f"✅ Detection Method: {gpu_info['method']}")
    print(f"✅ Environment: {gpu_info['environment']}")

    if gpu_info["method"] == "direct":
        # Serverless: Direct detection
        print(f"\n📊 GPU Information:")
        print(f"   Hostname: {gpu_info['hostname']}")
        print(f"   GPU Count: {gpu_info['gpu_count']}")

        for gpu in gpu_info["gpus"]:
            print(f"\n   GPU {gpu['gpu_index']}: {gpu['name']}")
            print(f"      Driver: {gpu['driver_version']}")
            print(f"      Memory: {gpu['memory_total']}")
            print(f"      Compute Capability: {gpu['compute_capability']}")
            print(f"      UUID: {gpu['uuid'][:20]}...")

    elif gpu_info["method"] == "distributed":
        # Classic: Distributed detection
        print(f"\n📊 Cluster Configuration:")
        print(f"   Total Executors: {gpu_info['total_executors']}")
        print(f"   Worker Nodes: {gpu_info['worker_node_count']}")
        print(f"   Physical GPUs: {gpu_info['physical_gpu_count']}")

        for hostname, gpus in gpu_info["worker_nodes"].items():
            print(f"\n   Worker: {hostname}")
            for gpu in gpus:
                print(f"      GPU: {gpu['name']}")
                print(f"         Driver: {gpu['driver']}")
                print(f"         Memory: {gpu['memory']}")
else:
    print(f"❌ GPU Detection Failed")
    print(f"   Error: {gpu_info.get('error', 'Unknown error')}")

print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: CUDA Runtime Detection
# MAGIC
# MAGIC Detect CUDA runtime version and toolkit installation.

# COMMAND ----------

from cuda_healthcheck import CUDADetector

print("=" * 80)
print("🔍 CUDA RUNTIME DETECTION")
print("=" * 80)

detector = CUDADetector()

# Detect CUDA runtime
cuda_runtime = detector.detect_cuda_runtime()
print(f"\n✅ CUDA Runtime: {cuda_runtime if cuda_runtime else 'Not detected'}")

# Detect NVCC
nvcc_version = detector.detect_nvcc_version()
print(f"✅ NVCC Version: {nvcc_version if nvcc_version else 'Not detected'}")

# Detect PyTorch
pytorch_info = detector.detect_pytorch()
print(f"\n📦 PyTorch:")
print(f"   Version: {pytorch_info.version}")
print(f"   CUDA Available: {pytorch_info.cuda_version}")
print(f"   Compatible: {pytorch_info.is_compatible}")

# Detect TensorFlow
tf_info = detector.detect_tensorflow()
print(f"\n📦 TensorFlow:")
print(f"   Version: {tf_info.version}")
print(f"   CUDA Version: {tf_info.cuda_version}")
print(f"   Compatible: {tf_info.is_compatible}")

print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Breaking Changes Analysis
# MAGIC
# MAGIC Analyze compatibility between CUDA versions and detect breaking changes for PyTorch, TensorFlow, and RAPIDS.

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
# MAGIC - ✅ Detected Serverless GPU Compute environment
# MAGIC - ✅ Validated GPU hardware (serverless-compatible method)
# MAGIC - ✅ Detected CUDA driver and runtime versions
# MAGIC - ✅ Analyzed ML framework compatibility
# MAGIC - ✅ Provided compatibility scores and recommendations
# MAGIC
# MAGIC **Key Features of Serverless:**
# MAGIC - Direct GPU access (no Spark distribution needed)
# MAGIC - Faster execution (single-process model)
# MAGIC - Simpler architecture (no driver-worker complexity)
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Save this notebook for regular validation
# MAGIC - Run before upgrading CUDA or ML frameworks
# MAGIC - Use compatibility scores to plan upgrades
# MAGIC
# MAGIC **For Classic ML Runtime:**
# MAGIC - Use `databricks_healthcheck.py` for distributed detection
# MAGIC - Leverages Spark for multi-worker GPU discovery
