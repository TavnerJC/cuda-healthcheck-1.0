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
# MAGIC
# MAGIC **⚠️ Important:** After running this cell, you'll see a **red warning note** that says:
# MAGIC > "Note: you may need to restart the kernel using %restart_python or dbutils.library.restartPython()"
# MAGIC
# MAGIC **This is NORMAL and EXPECTED!** It means the installation succeeded. Just proceed to Step 2.

# COMMAND ----------

# MAGIC %pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Restart Python
# MAGIC
# MAGIC **REQUIRED:** Restart Python to load the newly installed package.
# MAGIC
# MAGIC **What happens:**
# MAGIC - ⏸️ Notebook execution pauses (~10 seconds)
# MAGIC - 🔄 Python interpreter restarts
# MAGIC - 🧹 All variables cleared (expected behavior)
# MAGIC - ✅ Package now ready to use
# MAGIC
# MAGIC **⚠️ Do NOT re-run Step 1 after this restart!**

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
# MAGIC ## Step 6 (Optional): CUDA 12.6 Specific Compatibility Check
# MAGIC
# MAGIC If your environment uses CUDA 12.6 (common with PyTorch 2.7.1+cu126), run this cell to verify specific compatibility.

# COMMAND ----------

from cuda_healthcheck.data import BreakingChangesDatabase

print("=" * 80)
print("🔍 CUDA 12.6 SPECIFIC COMPATIBILITY CHECK")
print("=" * 80)

db = BreakingChangesDatabase()

# Detect your actual PyTorch CUDA version
from cuda_healthcheck import CUDADetector
detector = CUDADetector()
pytorch_info = detector.detect_pytorch()

detected_libs = [
    {"name": "pytorch", "version": pytorch_info.version, "cuda_version": pytorch_info.cuda_version},
]

print(f"\n📦 Detected Environment:")
print(f"   PyTorch: {pytorch_info.version}")
print(f"   CUDA: {pytorch_info.cuda_version}")

# Score for CUDA 12.6 specifically
if "12.6" in pytorch_info.cuda_version:
    print("\n📊 Testing Against CUDA 12.6:")
    score_126 = db.score_compatibility(detected_libs, "12.6")
    
    print(f"\n💯 Compatibility Score: {score_126['compatibility_score']}/100")
    print(f"   Critical Issues: {score_126['critical_issues']}")
    print(f"   Warning Issues: {score_126['warning_issues']}")
    print(f"   Recommendation: {score_126['recommendation']}")
    
    if score_126['breaking_changes']['WARNING']:
        print("\n⚠️  NOTE: If you see a warning about 'CUDA 12.4 binaries on 12.6':")
        print("   This is overly cautious - you have PyTorch built FOR 12.6 (cu126)")
        print("   Your setup is actually optimal! ✅")
else:
    print(f"\n✅ Your CUDA version: {pytorch_info.cuda_version}")
    score = db.score_compatibility(detected_libs, pytorch_info.cuda_version.split('+')[-1][:4])
    print(f"   Compatibility Score: {score['compatibility_score']}/100")

print("\n" + "=" * 80)

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
# MAGIC - ✅ Verified CUDA 12.6 support (if applicable)
# MAGIC
# MAGIC **Key Features of Serverless:**
# MAGIC - Direct GPU access (no Spark distribution needed)
# MAGIC - Faster execution (single-process model)
# MAGIC - Simpler architecture (no driver-worker complexity)
# MAGIC - Uses `detect_gpu_auto()` for environment-aware detection
# MAGIC
# MAGIC **Validated Hardware (Example from Testing):**
# MAGIC - NVIDIA A10G (23GB, Compute Capability 8.6)
# MAGIC - Driver: 550.144.03
# MAGIC - CUDA Runtime: 12.6 (via PyTorch 2.7.1+cu126)
# MAGIC - Compatibility Score: 90-100/100 ✅
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Save this notebook for regular validation
# MAGIC - Run before upgrading CUDA or ML frameworks
# MAGIC - Use compatibility scores to plan upgrades
# MAGIC - Share with your ML team
# MAGIC
# MAGIC **For Classic ML Runtime:**
# MAGIC - Use `databricks_healthcheck.py` for distributed detection
# MAGIC - Leverages Spark for multi-worker GPU discovery
# MAGIC
# MAGIC **Troubleshooting:**
# MAGIC - Red warning after %pip install? → Normal! Proceed to restart.
# MAGIC - ModuleNotFoundError? → Did you restart Python (Step 2)?
# MAGIC - Variables undefined after restart? → Expected behavior, continue to Step 3+
# MAGIC
# MAGIC **Documentation:**
# MAGIC - Full Guide: https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/docs/DATABRICKS_DEPLOYMENT.md
# MAGIC - Quick Start: https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/docs/DATABRICKS_QUICK_START.md
