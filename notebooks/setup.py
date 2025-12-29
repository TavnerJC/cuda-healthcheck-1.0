# Databricks notebook source
# MAGIC %md
# MAGIC # CUDA Healthcheck Tool - Setup and Installation
# MAGIC 
# MAGIC This notebook guides you through installing and configuring the CUDA Healthcheck Tool on your Databricks cluster.
# MAGIC 
# MAGIC **What this notebook does:**
# MAGIC 1. Checks prerequisites
# MAGIC 2. Installs required dependencies
# MAGIC 3. Configures environment variables
# MAGIC 4. Validates installation
# MAGIC 5. Runs a test healthcheck
# MAGIC 
# MAGIC **Requirements:**
# MAGIC - GPU-enabled Databricks cluster
# MAGIC - Databricks Runtime 13.3 LTS ML or higher
# MAGIC - Python 3.10+

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Check Prerequisites

# COMMAND ----------

import sys
import subprocess
import os

print("=" * 80)
print("CUDA HEALTHCHECK SETUP - Prerequisites Check")
print("=" * 80)
print()

# Check Python version
print(f"✓ Python Version: {sys.version}")
python_version = sys.version_info
if python_version.major >= 3 and python_version.minor >= 10:
    print("  Status: ✓ Compatible (3.10+)")
else:
    print("  Status: ⚠️ Warning - Python 3.10+ recommended")
print()

# Check if running on GPU cluster
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        gpu_info = result.stdout.strip().split('\n')
        print(f"✓ GPU Detected: {len(gpu_info)} GPU(s)")
        for i, gpu in enumerate(gpu_info):
            parts = gpu.split(',')
            print(f"  [{i}] {parts[0].strip()} - Driver: {parts[1].strip()}")
    else:
        print("⚠️ Warning: nvidia-smi check failed")
except Exception as e:
    print(f"⚠️ Warning: Could not detect GPU - {e}")
    print("   This might be a CPU cluster. GPU cluster required for CUDA healthcheck.")
print()

# Check Databricks environment
if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
    runtime = os.environ['DATABRICKS_RUNTIME_VERSION']
    print(f"✓ Databricks Runtime: {runtime}")
else:
    print("⚠️ Warning: Not running in Databricks environment")
print()

print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Install Dependencies
# MAGIC 
# MAGIC Install required Python packages.

# COMMAND ----------

print("Installing dependencies...")
print("=" * 80)

# Install databricks-sdk
print("📦 Installing databricks-sdk...")
%pip install databricks-sdk --quiet

# Install python-dotenv for environment variable management
print("📦 Installing python-dotenv...")
%pip install python-dotenv --quiet

print()
print("✓ All dependencies installed successfully")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Clone or Upload CUDA Healthcheck Repository
# MAGIC 
# MAGIC **Option A: Clone from Git (Recommended)**
# MAGIC 
# MAGIC If you have the repository in Git, clone it to Databricks Repos:
# MAGIC 1. Go to Repos in Databricks workspace
# MAGIC 2. Click "Add Repo"
# MAGIC 3. Enter repository URL
# MAGIC 4. Clone to `/Workspace/Repos/<username>/cuda-healthcheck`
# MAGIC 
# MAGIC **Option B: Upload Files (Alternative)**
# MAGIC 
# MAGIC Upload the `cuda-healthcheck` directory to:
# MAGIC `/Workspace/cuda-healthcheck/`

# COMMAND ----------

# Check if package is accessible
expected_paths = [
    "/Workspace/Repos/cuda-healthcheck/cuda-healthcheck",
    "/Workspace/cuda-healthcheck",
]

package_path = None
for path in expected_paths:
    try:
        files = dbutils.fs.ls(f"file://{path}")
        package_path = path
        print(f"✓ CUDA Healthcheck found at: {path}")
        break
    except:
        continue

if package_path is None:
    print("⚠️ CUDA Healthcheck package not found!")
    print()
    print("Please do one of the following:")
    print("1. Clone the repository to Databricks Repos")
    print("2. Upload the package to /Workspace/cuda-healthcheck/")
    print()
    print("Then re-run this cell.")
    dbutils.notebook.exit("Package not found")
else:
    # Add to Python path
    sys.path.insert(0, package_path)
    print(f"✓ Added to Python path: {package_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Configure Environment Variables (Optional)
# MAGIC 
# MAGIC Set environment variables for advanced features like Delta table exports.

# COMMAND ----------

# Check if environment variables are set
print("Checking environment variables...")
print("=" * 80)

# Check DATABRICKS_HOST (usually auto-set in Databricks)
databricks_host = os.getenv("DATABRICKS_HOST")
if databricks_host:
    print(f"✓ DATABRICKS_HOST: {databricks_host}")
else:
    print("⚠️ DATABRICKS_HOST not set (optional)")

# Check DATABRICKS_TOKEN (required for API operations)
databricks_token = os.getenv("DATABRICKS_TOKEN")
if databricks_token:
    print(f"✓ DATABRICKS_TOKEN: {'*' * 20} (hidden)")
else:
    print("⚠️ DATABRICKS_TOKEN not set (optional, needed for cluster scanning)")

# Check DATABRICKS_WAREHOUSE_ID (required for Delta operations)
warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
if warehouse_id:
    print(f"✓ DATABRICKS_WAREHOUSE_ID: {warehouse_id}")
else:
    print("⚠️ DATABRICKS_WAREHOUSE_ID not set (optional, needed for some Delta operations)")

# Check log level
log_level = os.getenv("CUDA_HEALTHCHECK_LOG_LEVEL", "INFO")
print(f"✓ CUDA_HEALTHCHECK_LOG_LEVEL: {log_level}")

print("=" * 80)
print()
print("💡 Tip: To set environment variables, use Databricks secrets:")
print("   token = dbutils.secrets.get(scope='cuda-healthcheck', key='token')")
print("   os.environ['DATABRICKS_TOKEN'] = token")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Validate Installation
# MAGIC 
# MAGIC Import the package and verify all components are working.

# COMMAND ----------

print("Validating installation...")
print("=" * 80)

try:
    # Test imports
    print("Testing imports...")
    
    from src import CUDADetector
    print("  ✓ CUDADetector imported")
    
    from src import HealthcheckOrchestrator
    print("  ✓ HealthcheckOrchestrator imported")
    
    from src import BreakingChangesDatabase
    print("  ✓ BreakingChangesDatabase imported")
    
    from cuda_healthcheck.databricks import DatabricksHealthchecker
    print("  ✓ DatabricksHealthchecker imported")
    
    from cuda_healthcheck.utils import get_logger
    print("  ✓ Utilities imported")
    
    print()
    print("✓ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print()
    print("Please check:")
    print("1. Package is in the correct location")
    print("2. All files are present")
    print("3. __init__.py files exist in all directories")
    dbutils.notebook.exit(f"Import failed: {e}")

print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Run Test Healthcheck
# MAGIC 
# MAGIC Run a quick healthcheck to verify everything is working.

# COMMAND ----------

print("Running test healthcheck...")
print("=" * 80)
print()

try:
    from src import run_complete_healthcheck
    
    # Run simple healthcheck
    result = run_complete_healthcheck()
    
    print("✓ Healthcheck completed successfully!")
    print()
    print(f"Healthcheck ID: {result.get('healthcheck_id')}")
    print(f"Status: {result.get('status', 'unknown').upper()}")
    print(f"Timestamp: {result.get('timestamp')}")
    print()
    
    # Show CUDA info
    cuda_env = result.get('cuda_environment', {})
    print("CUDA Environment:")
    print(f"  Runtime: {cuda_env.get('cuda_runtime_version', 'N/A')}")
    print(f"  Driver: {cuda_env.get('cuda_driver_version', 'N/A')}")
    
    gpus = cuda_env.get('gpus', [])
    print(f"  GPUs: {len(gpus)}")
    for gpu in gpus:
        print(f"    - {gpu.get('name')} (CC {gpu.get('compute_capability')})")
    
    print()
    print("=" * 80)
    print("✓ INSTALLATION SUCCESSFUL!")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Test healthcheck failed: {e}")
    print()
    import traceback
    traceback.print_exc()
    dbutils.notebook.exit(f"Test failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Installation Summary
# MAGIC 
# MAGIC Review your installation and next steps.

# COMMAND ----------

print("=" * 80)
print("INSTALLATION SUMMARY")
print("=" * 80)
print()
print("✓ Prerequisites checked")
print("✓ Dependencies installed")
print("✓ Package configured")
print("✓ Installation validated")
print("✓ Test healthcheck passed")
print()
print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Run the main healthcheck notebook:")
print("   /Workspace/Repos/cuda-healthcheck/notebooks/healthcheck_runner")
print()
print("2. Read the documentation:")
print("   - README.md - Overview and quick start")
print("   - docs/ENVIRONMENT_VARIABLES.md - Configuration guide")
print("   - docs/BREAKING_CHANGES.md - Known compatibility issues")
print("   - QUICK_REFERENCE.md - Common tasks and examples")
print()
print("3. Set up automated healthchecks:")
print("   - Create a Databricks Job to run healthcheck_runner.py")
print("   - Schedule to run after cluster creation or updates")
print("   - Configure alerts for critical issues")
print()
print("4. Query historical results:")
print("   SELECT * FROM main.cuda_healthcheck.healthcheck_results")
print("   ORDER BY timestamp DESC LIMIT 10")
print()
print("=" * 80)
print("🎉 Setup Complete!")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC 
# MAGIC ### Common Issues and Solutions
# MAGIC 
# MAGIC **Issue: "Package not found"**
# MAGIC - Solution: Ensure the repository is cloned to the correct location
# MAGIC - Check: `/Workspace/Repos/<username>/cuda-healthcheck`
# MAGIC 
# MAGIC **Issue: "nvidia-smi not found"**
# MAGIC - Solution: Ensure you're using a GPU-enabled cluster
# MAGIC - Check: Cluster configuration has GPU instance type (g5.xlarge, p3.2xlarge, etc.)
# MAGIC 
# MAGIC **Issue: "Import errors"**
# MAGIC - Solution: Verify all `__init__.py` files exist
# MAGIC - Check: Run `ls -R /Workspace/Repos/cuda-healthcheck/cuda-healthcheck/src`
# MAGIC 
# MAGIC **Issue: "Delta table errors"**
# MAGIC - Solution: Set DATABRICKS_WAREHOUSE_ID environment variable
# MAGIC - Or: Use Spark directly instead of SQL warehouse
# MAGIC 
# MAGIC ### Getting Help
# MAGIC 
# MAGIC - Documentation: `/Workspace/Repos/cuda-healthcheck/docs/`
# MAGIC - Quick Reference: `/Workspace/Repos/cuda-healthcheck/QUICK_REFERENCE.md`
# MAGIC - Implementation Guide: `/Workspace/Repos/cuda-healthcheck/IMPLEMENTATION_SUMMARY.md`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Create Delta Table Schema
# MAGIC 
# MAGIC Pre-create the Delta table with optimized schema.

# COMMAND ----------

# Create schema and table for healthcheck results
catalog = "main"
schema = "cuda_healthcheck"
table = "healthcheck_results"

# Create schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"✓ Schema created: {catalog}.{schema}")

# Create table with explicit schema
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{table} (
    healthcheck_id STRING,
    cluster_id STRING,
    cluster_name STRING,
    timestamp TIMESTAMP,
    cuda_runtime_version STRING,
    cuda_driver_version STRING,
    compatibility_score INT,
    critical_issues INT,
    warning_issues INT,
    status STRING,
    recommendation STRING
)
USING DELTA
COMMENT 'CUDA Healthcheck results for GPU clusters'
"""

spark.sql(create_table_sql)
print(f"✓ Table created: {catalog}.{schema}.{table}")
print()
print("Query with:")
print(f"  SELECT * FROM {catalog}.{schema}.{table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC **Setup Notebook Information:**
# MAGIC - Version: 1.0.0
# MAGIC - Last Updated: December 2024
# MAGIC - Estimated Time: 5-10 minutes
# MAGIC - Requires: GPU-enabled Databricks cluster
# MAGIC 
# MAGIC **Maintenance:**
# MAGIC - Re-run after package updates
# MAGIC - Re-run if changing cluster configuration
# MAGIC - Verify setup after Databricks runtime upgrades



