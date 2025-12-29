# Databricks notebook source
# MAGIC %md
# MAGIC # CUDA Healthcheck Runner
# MAGIC
# MAGIC This notebook runs a complete CUDA healthcheck on the current Databricks cluster.
# MAGIC
# MAGIC **Features:**
# MAGIC - Detects CUDA version and GPU properties
# MAGIC - Checks library compatibility (PyTorch, TensorFlow, cuDF)
# MAGIC - Analyzes breaking changes
# MAGIC - Provides compatibility scores and recommendations
# MAGIC - Exports results to Delta table
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - GPU-enabled Databricks cluster
# MAGIC - CUDA Healthcheck package installed (see Setup notebook)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Imports

# COMMAND ----------

# Import the CUDA Healthcheck package
import sys

sys.path.insert(0, "/Workspace/Repos/cuda-healthcheck/cuda-healthcheck")

from cuda_healthcheck.databricks import get_healthchecker, DatabricksHealthchecker
from src import run_complete_healthcheck
import json

print("✓ CUDA Healthcheck package loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Quick Healthcheck
# MAGIC
# MAGIC Run a simple healthcheck to verify everything is working.

# COMMAND ----------

# Quick healthcheck using simple API
result = run_complete_healthcheck()

print("=" * 80)
print("QUICK HEALTHCHECK RESULTS")
print("=" * 80)
print(f"Status: {result['status'].upper()}")
print(f"Healthcheck ID: {result['healthcheck_id']}")
print(f"Timestamp: {result['timestamp']}")
print()

# Show CUDA environment
cuda_env = result.get("cuda_environment", {})
print("CUDA Environment:")
print(f"  Runtime Version: {cuda_env.get('cuda_runtime_version', 'N/A')}")
print(f"  Driver Version: {cuda_env.get('cuda_driver_version', 'N/A')}")
print(f"  NVCC Version: {cuda_env.get('nvcc_version', 'N/A')}")
print()

# Show GPUs
gpus = cuda_env.get("gpus", [])
print(f"GPUs Detected: {len(gpus)}")
for gpu in gpus:
    print(f"  [{gpu.get('gpu_index')}] {gpu.get('name')}")
    print(f"      Compute Capability: {gpu.get('compute_capability')}")
    print(f"      Memory: {gpu.get('memory_total_mb')} MB")

print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Detailed Healthcheck with Databricks Integration
# MAGIC
# MAGIC Use the DatabricksHealthchecker for more detailed analysis and cluster metadata.

# COMMAND ----------

# Get configured healthchecker
checker = get_healthchecker()

print("Running detailed healthcheck...")
detailed_result = checker.run_healthcheck()

# Display results with nice formatting
checker.display_results(detailed_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Analyze Compatibility
# MAGIC
# MAGIC Show detailed compatibility analysis and breaking changes.

# COMMAND ----------

# Extract compatibility analysis
compat = detailed_result.get("compatibility_analysis", {})

print("=" * 80)
print("COMPATIBILITY ANALYSIS")
print("=" * 80)
print(f"Compatibility Score: {compat.get('compatibility_score', 0)}/100")
print(f"Total Issues: {compat.get('total_issues', 0)}")
print(f"  - Critical: {compat.get('critical_issues', 0)}")
print(f"  - Warnings: {compat.get('warning_issues', 0)}")
print(f"  - Info: {compat.get('info_issues', 0)}")
print()
print(f"Recommendation: {compat.get('recommendation', 'N/A')}")
print("=" * 80)

# Show breaking changes if any
breaking_changes = compat.get("breaking_changes", {})
critical_changes = breaking_changes.get("CRITICAL", [])
warning_changes = breaking_changes.get("WARNING", [])

if critical_changes:
    print()
    print("⚠️ CRITICAL BREAKING CHANGES:")
    for change in critical_changes:
        print(f"  - {change.get('title', 'Unknown')}")
        print(f"    Library: {change.get('affected_library', 'N/A')}")
        print()

if warning_changes:
    print()
    print("⚠️ WARNING-LEVEL CHANGES:")
    for change in warning_changes:
        print(f"  - {change.get('title', 'Unknown')}")
        print(f"    Library: {change.get('affected_library', 'N/A')}")
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Export Results to Delta Table
# MAGIC
# MAGIC Save the healthcheck results to a Delta table for historical tracking and analysis.

# COMMAND ----------

# Define Delta table location
catalog = "main"
schema = "cuda_healthcheck"
table = "healthcheck_results"
table_path = f"{catalog}.{schema}.{table}"

print(f"Exporting results to Delta table: {table_path}")

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# Convert result to DataFrame
from pyspark.sql import Row
from datetime import datetime

# Flatten the result for Delta table
flat_result = {
    "healthcheck_id": detailed_result.get("healthcheck_id"),
    "cluster_id": detailed_result.get("cluster_id"),
    "cluster_name": detailed_result.get("cluster_name"),
    "timestamp": datetime.fromisoformat(detailed_result.get("timestamp").replace("Z", "+00:00")),
    "cuda_runtime_version": detailed_result.get("cuda_environment", {}).get("cuda_runtime_version"),
    "cuda_driver_version": detailed_result.get("cuda_environment", {}).get("cuda_driver_version"),
    "compatibility_score": detailed_result.get("compatibility_analysis", {}).get(
        "compatibility_score"
    ),
    "critical_issues": detailed_result.get("compatibility_analysis", {}).get("critical_issues"),
    "warning_issues": detailed_result.get("compatibility_analysis", {}).get("warning_issues"),
    "status": detailed_result.get("status"),
    "recommendation": detailed_result.get("compatibility_analysis", {}).get("recommendation"),
}

# Create DataFrame
df = spark.createDataFrame([Row(**flat_result)])

# Write to Delta table
df.write.format("delta").mode("append").saveAsTable(table_path)

print(f"✓ Results exported successfully to {table_path}")
print()
print("Query results with:")
print(f"  SELECT * FROM {table_path} ORDER BY timestamp DESC LIMIT 10")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Historical Analysis
# MAGIC
# MAGIC Query historical healthcheck results to track changes over time.

# COMMAND ----------

# Query recent healthcheck results
recent_results = spark.sql(
    f"""
    SELECT 
        timestamp,
        cluster_name,
        cuda_driver_version,
        compatibility_score,
        status,
        critical_issues,
        warning_issues
    FROM {table_path}
    ORDER BY timestamp DESC
    LIMIT 10
"""
)

display(recent_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Generate Summary Report
# MAGIC
# MAGIC Create a summary report for the current cluster.

# COMMAND ----------

# Summary report
recommendations = detailed_result.get("recommendations", [])

print("=" * 80)
print("HEALTHCHECK SUMMARY REPORT")
print("=" * 80)
print(f"Cluster: {detailed_result.get('cluster_name', 'Unknown')}")
print(f"Cluster ID: {detailed_result.get('cluster_id', 'Unknown')}")
print(f"Timestamp: {detailed_result.get('timestamp')}")
print()
print(f"Overall Status: {detailed_result.get('status', 'unknown').upper()}")
print(f"Compatibility Score: {compat.get('compatibility_score', 0)}/100")
print()
print("Recommendations:")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")
print("=" * 80)

# Save report as JSON for download
report_filename = f"/tmp/healthcheck_report_{detailed_result.get('healthcheck_id')}.json"
dbutils.fs.put(f"file://{report_filename}", json.dumps(detailed_result, indent=2), overwrite=True)
print(f"\nFull report saved to: {report_filename}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Next Steps
# MAGIC
# MAGIC Based on the healthcheck results:
# MAGIC
# MAGIC **If Status is HEALTHY (Score 90-100):**
# MAGIC - ✓ Your environment is well-configured
# MAGIC - Continue monitoring for library updates
# MAGIC - Run healthcheck monthly or after major updates
# MAGIC
# MAGIC **If Status is WARNING (Score 70-89):**
# MAGIC - ⚠️ Review warnings in the compatibility analysis
# MAGIC - Test your workloads thoroughly before production
# MAGIC - Consider upgrading libraries with warnings
# MAGIC
# MAGIC **If Status is CRITICAL (Score < 70):**
# MAGIC - ❌ Address critical breaking changes immediately
# MAGIC - Review migration paths for affected libraries
# MAGIC - Do NOT deploy to production until resolved
# MAGIC - Consult the breaking changes documentation
# MAGIC
# MAGIC **Resources:**
# MAGIC - Breaking Changes Documentation: `/Workspace/Repos/cuda-healthcheck/docs/BREAKING_CHANGES.md`
# MAGIC - Migration Guide: `/Workspace/Repos/cuda-healthcheck/docs/MIGRATION_GUIDE.md`
# MAGIC - Environment Variables: `/Workspace/Repos/cuda-healthcheck/docs/ENVIRONMENT_VARIABLES.md`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: Custom Checks
# MAGIC
# MAGIC Run custom compatibility checks for specific scenarios.

# COMMAND ----------

# Example: Check compatibility between two specific CUDA versions
from src import HealthcheckOrchestrator

orchestrator = HealthcheckOrchestrator()

# Check if upgrading from 12.4 to 13.0 is safe
compat_check = orchestrator.check_compatibility("12.4", "13.0")

print("Compatibility Check: CUDA 12.4 → 13.0")
print("=" * 80)
print(f"Compatible: {compat_check['compatible']}")
print(f"Breaking Changes Found: {compat_check['total_changes']}")
print()

if not compat_check["compatible"]:
    print("⚠️ INCOMPATIBLE - Review the following:")
    critical = compat_check["breaking_changes"]["critical"]
    for change in critical:
        print(f"\n  Library: {change['affected_library']}")
        print(f"  Issue: {change['title']}")
        print(f"  Migration: {change['migration_path'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC **Notebook Information:**
# MAGIC - Version: 1.0.0
# MAGIC - Last Updated: December 2024
# MAGIC - Compatible with: CUDA 12.4, 12.6, 13.0
# MAGIC - Requires: GPU-enabled Databricks cluster
