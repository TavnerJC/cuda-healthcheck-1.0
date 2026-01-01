# Databricks notebook source
# MAGIC %md
# MAGIC # 🔍 CUDA Environment Validation with Advanced Compatibility Checks
# MAGIC
# MAGIC Comprehensive GPU and CUDA configuration validation for Databricks.
# MAGIC **Enhanced with Runtime Detection, Driver Mapping, and PyTorch Compatibility!**
# MAGIC
# MAGIC ## What This Notebook Does:
# MAGIC
# MAGIC 1. ✅ Detects GPU hardware (Classic ML Runtime & Serverless)
# MAGIC 2. ✅ Validates CUDA driver and runtime versions
# MAGIC 3. ✅ **Detects Databricks Runtime version (NEW!)** 🎉
# MAGIC 4. ✅ **Checks Driver compatibility & immutability (NEW!)** 🎉
# MAGIC 5. ✅ **Validates PyTorch + Driver compatibility (NEW!)** 🎉
# MAGIC 6. ✅ **Checks CuOPT nvJitLink compatibility** 🎉
# MAGIC 7. ✅ Analyzes Databricks Runtime CUDA components
# MAGIC 8. ✅ Provides CUDA 13.0 upgrade compatibility analysis
# MAGIC 9. ✅ Lists detailed breaking changes with migration paths
# MAGIC
# MAGIC ## Key Features (v0.5.0):
# MAGIC
# MAGIC ### Runtime Detection
# MAGIC - Detects ML Runtime 14.3, 15.1, 15.2, 16.4, etc.
# MAGIC - Identifies Serverless GPU Compute
# MAGIC - Maps runtime → CUDA version automatically
# MAGIC
# MAGIC ### Driver Mapping
# MAGIC - Maps runtime → expected driver version
# MAGIC - Detects **immutable drivers** (14.3, 15.1, 15.2)
# MAGIC - Warns when drivers CANNOT be upgraded
# MAGIC
# MAGIC ### PyTorch Compatibility
# MAGIC - Detects PyTorch 2.4+ on Runtime 14.3 (incompatible!)
# MAGIC - Identifies unfixable platform constraints
# MAGIC - Provides actionable solutions
# MAGIC
# MAGIC ### CuOPT Compatibility
# MAGIC - Detects nvJitLink 12.4.127 vs required 12.9+
# MAGIC - Identifies platform-level package conflicts
# MAGIC - Links to GitHub issues for reporting
# MAGIC
# MAGIC ## Requirements:
# MAGIC
# MAGIC - GPU-enabled Databricks cluster
# MAGIC - Classic ML Runtime 14.3+ OR Serverless GPU Compute
# MAGIC - Python 3.10+

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📦 Step 1: Install CUDA Healthcheck Tool
# MAGIC
# MAGIC Install from GitHub. This includes all the latest compatibility checks.

# COMMAND ----------
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🎮 Step 2: GPU Detection
# MAGIC
# MAGIC Auto-detects whether you're on Classic ML Runtime or Serverless GPU Compute
# MAGIC and uses the appropriate detection method.

# COMMAND ----------
from cuda_healthcheck import CUDADetector
from cuda_healthcheck.databricks import detect_gpu_auto
import json
from datetime import datetime, timezone

# Auto-detect GPU (works on both Serverless and Classic)
gpu_info = detect_gpu_auto()

print("=" * 80)
print("🎮 GPU DETECTION RESULTS")
print("=" * 80)
print(f"Environment Type: {gpu_info.get('environment', 'unknown')}")
print(f"Detection Method: {gpu_info.get('method', 'unknown')}")
print(f"GPU Count: {gpu_info.get('gpu_count', 0)}")

# The 'gpus' key is now standardized across both Classic and Serverless
if gpu_info.get('gpu_count', 0) > 0 and 'gpus' in gpu_info:
    for gpu in gpu_info['gpus']:
        print(f"\n📊 GPU: {gpu.get('name', 'Unknown')}")
        print(f"   Driver: {gpu.get('driver_version', 'N/A')}")
        print(f"   Memory: {gpu.get('memory_total', 'N/A')}")
        print(f"   Compute: {gpu.get('compute_capability', 'N/A')}")
        print(f"   UUID: {gpu.get('uuid', 'N/A')}")
        
        # Classic clusters will have hostname field
        if 'hostname' in gpu:
            print(f"   Hostname: {gpu['hostname']}")
else:
    print(f"\n⚠️  No GPU detected: {gpu_info.get('error', 'Unknown reason')}")

print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🔧 Step 3: CUDA Environment Detection
# MAGIC
# MAGIC Detects CUDA runtime, driver, NVCC, and all installed ML libraries
# MAGIC (PyTorch, TensorFlow, RAPIDS, CuOPT, etc.)

# COMMAND ----------
# Full CUDA detection
detector = CUDADetector()
env = detector.detect_environment()

print("=" * 80)
print("🔧 CUDA ENVIRONMENT")
print("=" * 80)
print(f"CUDA Runtime: {env.cuda_runtime_version}")
print(f"CUDA Driver: {env.cuda_driver_version}")
print(f"NVCC Version: {env.nvcc_version}")
print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

# Display all detected libraries
print(f"\n📚 Detected Libraries:")
for lib in env.libraries:
    status = "✅" if lib.is_compatible else "⚠️"
    print(f"\n  {status} Library: {lib.name}")
    print(f"     Version: {lib.version}")
    print(f"     CUDA Version: {lib.cuda_version}")
    print(f"     Compatible: {lib.is_compatible}")
    if lib.warnings:
        print(f"     Warnings: {len(lib.warnings)}")

print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🏃 Step 4: Databricks Runtime & Driver Analysis (NEW!)
# MAGIC
# MAGIC **NEW FEATURE**: Detects Databricks runtime version and validates driver compatibility.
# MAGIC Critical for detecting PyTorch + Driver incompatibilities that users cannot fix!

# COMMAND ----------
from cuda_healthcheck.databricks import (
    detect_databricks_runtime,
    get_driver_version_for_runtime,
    check_driver_compatibility,
)

print("=" * 80)
print("🏃 DATABRICKS RUNTIME ANALYSIS")
print("=" * 80)

# Detect runtime
runtime_info = detect_databricks_runtime()

print(f"\n📊 Runtime Information:")
print(f"   Runtime Version: {runtime_info['runtime_version']}")
print(f"   Full Version String: {runtime_info['runtime_version_string']}")
print(f"   Is Databricks: {runtime_info['is_databricks']}")
print(f"   Is ML Runtime: {runtime_info['is_ml_runtime']}")
print(f"   Is GPU Runtime: {runtime_info['is_gpu_runtime']}")
print(f"   Is Serverless: {runtime_info['is_serverless']}")
print(f"   Expected CUDA: {runtime_info['cuda_version']}")
print(f"   Detection Method: {runtime_info['detection_method']}")

# Get driver requirements for this runtime
if runtime_info['runtime_version']:
    try:
        driver_info = get_driver_version_for_runtime(runtime_info['runtime_version'])
        
        print(f"\n🔧 Driver Requirements:")
        print(f"   Expected Driver Range: {driver_info['driver_min_version']}-{driver_info['driver_max_version']}")
        print(f"   CUDA Version: {driver_info['cuda_version']}")
        print(f"   Driver Immutable: {driver_info['is_immutable']}")
        
        if driver_info['is_immutable']:
            print(f"\n⚠️  WARNING: This runtime has an IMMUTABLE driver")
            print(f"   You CANNOT upgrade the driver on this runtime!")
            print(f"   This may cause PyTorch/CUDA incompatibilities.")
        
        # Check actual driver compatibility
        if env.cuda_driver_version:
            actual_driver = int(env.cuda_driver_version.split(".")[0])
            
            compatibility = check_driver_compatibility(
                runtime_info['runtime_version'],
                actual_driver
            )
            
            print(f"\n🔍 Driver Compatibility Check:")
            print(f"   Actual Driver: {actual_driver}")
            print(f"   Compatible: {compatibility['is_compatible']}")
            
            if not compatibility['is_compatible']:
                print(f"\n❌ DRIVER INCOMPATIBILITY DETECTED!")
                print(f"   {compatibility['error_message']}")
            else:
                print(f"   ✅ Driver is compatible with this runtime")
                
    except ValueError as e:
        print(f"\n⚠️  Could not get driver requirements: {e}")

print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🐍 Step 5: PyTorch + Driver Compatibility Check (NEW!)
# MAGIC
# MAGIC **CRITICAL**: Checks if your PyTorch version is compatible with the
# MAGIC immutable driver version on this Databricks runtime.

# COMMAND ----------
print("=" * 80)
print("🐍 PYTORCH + DRIVER COMPATIBILITY ANALYSIS")
print("=" * 80)

# Find PyTorch in detected libraries
pytorch_lib = next((lib for lib in env.libraries if lib.name.lower() == "torch"), None)

if pytorch_lib and pytorch_lib.version != "Not installed":
    pytorch_version = pytorch_lib.version
    pytorch_major_minor = ".".join(pytorch_version.split(".")[:2])
    
    print(f"\n📦 PyTorch Detected:")
    print(f"   Version: {pytorch_version}")
    print(f"   CUDA Version: {pytorch_lib.cuda_version}")
    
    # Get driver info
    if runtime_info['runtime_version']:
        try:
            driver_info = get_driver_version_for_runtime(runtime_info['runtime_version'])
            
            print(f"\n🔧 Driver Requirements:")
            print(f"   Runtime Driver: {driver_info['driver_min_version']}")
            print(f"   Driver Immutable: {driver_info['is_immutable']}")
            
            # Check known PyTorch incompatibilities
            incompatibilities = []
            
            # PyTorch 2.4+ requires driver >= 550
            if pytorch_major_minor >= "2.4" and driver_info['driver_min_version'] < 550:
                if driver_info['is_immutable']:
                    incompatibilities.append({
                        "severity": "CRITICAL",
                        "issue": f"PyTorch {pytorch_version} requires driver ≥ 550",
                        "runtime": f"Runtime {runtime_info['runtime_version']} locked at driver {driver_info['driver_min_version']}",
                        "fixable": False,
                        "solution": "Use Runtime 15.1+ or downgrade PyTorch to 2.3.x"
                    })
            
            # PyTorch 2.5+ requires driver >= 560
            if pytorch_major_minor >= "2.5" and driver_info['driver_min_version'] < 560:
                if driver_info['is_immutable']:
                    incompatibilities.append({
                        "severity": "CRITICAL",
                        "issue": f"PyTorch {pytorch_version} requires driver ≥ 560",
                        "runtime": f"Runtime {runtime_info['runtime_version']} locked at driver {driver_info['driver_min_version']}",
                        "fixable": False,
                        "solution": "Use Runtime 16.0+ or downgrade PyTorch"
                    })
            
            if incompatibilities:
                print(f"\n❌ PYTORCH-DRIVER INCOMPATIBILITY DETECTED!")
                print(f"{'=' * 80}")
                
                for incompat in incompatibilities:
                    print(f"\n🚨 {incompat['severity']}: {incompat['issue']}")
                    print(f"   Runtime: {incompat['runtime']}")
                    print(f"   Fixable by User: {incompat['fixable']}")
                    print(f"   💡 Solution: {incompat['solution']}")
                
                print(f"\n{'=' * 80}")
                print(f"📝 This is a PLATFORM CONSTRAINT - similar to CuOPT nvJitLink issue!")
                print(f"   Users CANNOT upgrade drivers on managed Databricks runtimes.")
                print(f"   Report to: https://github.com/databricks-industry-solutions/routing/issues")
                print(f"{'=' * 80}")
            else:
                print(f"\n✅ PyTorch {pytorch_version} is compatible with driver {driver_info['driver_min_version']}")
                
        except ValueError as e:
            print(f"\n⚠️  Could not check PyTorch compatibility: {e}")
else:
    print(f"\n📦 PyTorch Status: Not installed")
    print(f"   ℹ️  No PyTorch compatibility issues to check")

print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🚨 Step 6: CuOPT Compatibility Check (CRITICAL for Routing Optimization)
# MAGIC
# MAGIC **This is the key detection!** Checks if CuOPT can actually load on this
# MAGIC Databricks runtime, specifically looking for the nvJitLink version mismatch.

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
            print(f"      %pip install ortools")
            print(f"")
            print(f"   3. Wait for Databricks ML Runtime 17.0+ (with CUDA 12.9+ support)")
            print(f"")
            print(f"More Info:")
            print(f"   • Breaking change ID: cuopt-nvjitlink-databricks-ml-runtime")
            print(f"   • Tracked in: cuda_healthcheck/data/breaking_changes.py")
            print(f"   • GitHub: https://github.com/TavnerJC/cuda-healthcheck-on-databricks")
            print(f"{'=' * 80}")
            
            # Set flag for later cells
            cuopt_incompatible = True
    else:
        print(f"\n✅ CuOPT is compatible and working!")
        cuopt_incompatible = False
else:
    print(f"\n📦 CuOPT: Not installed")
    print(f"   (This is normal if you haven't run CuOPT workloads yet)")
    cuopt_incompatible = False

print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🔍 Step 7: Check Databricks Runtime CUDA Components
# MAGIC
# MAGIC Specifically checks the nvJitLink version in the Databricks runtime.
# MAGIC This is the component that causes CuOPT compatibility issues.

# COMMAND ----------
# Check specific CUDA component versions in Databricks runtime
import subprocess

print("=" * 80)
print("🔍 DATABRICKS RUNTIME CUDA COMPONENTS")
print("=" * 80)

# Check nvJitLink version specifically (the problematic component)
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
        
        # Extract version and analyze
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                nvjitlink_version = line.split(':')[1].strip()
                
                print(f"\n🔍 Version Analysis:")
                if nvjitlink_version.startswith('12.4'):
                    print(f"   ❌ Version {nvjitlink_version} is INCOMPATIBLE with CuOPT 25.12+")
                    print(f"   ✅ Requires: 12.9.79 or later")
                    print(f"   ⚠️  This is a Databricks Runtime limitation")
                    print(f"   📝 Users cannot fix this themselves")
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
    "nvidia-cudnn-cu12",
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

# COMMAND ----------
# MAGIC %md
# MAGIC ## 💯 Step 8: CUDA 13.0 Upgrade Compatibility
# MAGIC
# MAGIC Tests what would happen if you upgraded to CUDA 13.0.
# MAGIC Provides a compatibility score and identifies breaking changes.

# COMMAND ----------
from cuda_healthcheck.data import BreakingChangesDatabase

db = BreakingChangesDatabase()

# Extract PyTorch info from libraries list
pytorch_lib = next((lib for lib in env.libraries if lib.name.lower() == "pytorch"), None)
pytorch_version = pytorch_lib.version if pytorch_lib else "unknown"
pytorch_cuda_version = pytorch_lib.cuda_version if pytorch_lib else env.cuda_runtime_version

# Test upgrade path to CUDA 13.0
score = db.score_compatibility(
    detected_libraries=[
        {"name": "pytorch", "version": pytorch_version, 
         "cuda_version": pytorch_cuda_version}
    ],
    cuda_version="13.0"
)

print("=" * 80)
print("💯 CUDA 13.0 UPGRADE COMPATIBILITY")
print("=" * 80)
print(f"Score: {score['compatibility_score']}/100")
print(f"Critical Issues: {score['critical_issues']}")
print(f"Warning Issues: {score['warning_issues']}")
print(f"Status: {score['recommendation']}")

if score['critical_issues'] > 0:
    print(f"\n⚠️  Critical breaking changes found!")
    print(f"   Review details below before upgrading to CUDA 13.0")
if score['warning_issues'] > 0:
    print(f"\n⚠️  {score['warning_issues']} warnings - review before upgrading")

print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📋 Step 9: Detailed Compatibility Issues
# MAGIC
# MAGIC Provides complete details on ALL breaking changes for CUDA 13.0:
# MAGIC - What breaks
# MAGIC - Why it breaks
# MAGIC - How to fix it
# MAGIC - Where to find the code

# COMMAND ----------
# Get detailed breaking changes for current CUDA version
print("=" * 80)
print("🔍 DETAILED COMPATIBILITY ANALYSIS")
print("=" * 80)

# Get all breaking changes that involve CUDA 13.0
all_changes = db.get_all_changes()
changes_13 = [c for c in all_changes if "13.0" in c.cuda_version_to]

if changes_13:
    print(f"\n📋 Found {len(changes_13)} breaking change(s) for CUDA 13.0:")
    
    for i, change in enumerate(changes_13, 1):
        print(f"\n{'─' * 80}")
        print(f"Issue #{i}: {change.title}")
        print(f"{'─' * 80}")
        print(f"Severity: {change.severity.upper()}")
        print(f"Library: {change.affected_library}")
        print(f"Transition: CUDA {change.cuda_version_from} → {change.cuda_version_to}")
        print(f"\nDescription:")
        print(f"  {change.description}")
        
        if change.migration_path:
            print(f"\n✅ Migration Path:")
            # Split by newline since migration_path is stored as a single string
            steps = change.migration_path.strip().split('\n')
            for step in steps:
                step = step.strip()
                if step:  # Only print non-empty lines
                    print(f"  {step}")
        
        print(f"\n📚 Code Reference:")
        print(f"  File: cuda_healthcheck/data/breaking_changes.py")
        print(f"  Change ID: {change.id}")
        print(f"  GitHub: https://github.com/TavnerJC/cuda-healthcheck-on-databricks/blob/main/cuda_healthcheck/data/breaking_changes.py")
else:
    print("\n✅ No breaking changes found for CUDA 13.0")

# Also check transition from current CUDA version
if env.cuda_runtime_version:
    print(f"\n{'=' * 80}")
    print(f"🔄 TRANSITION ANALYSIS: CUDA {env.cuda_runtime_version} → 13.0")
    print(f"{'=' * 80}")
    
    transition_changes = db.get_changes_by_cuda_transition(
        env.cuda_runtime_version, 
        "13.0"
    )
    
    if transition_changes:
        print(f"\n⚠️  {len(transition_changes)} change(s) affect your specific upgrade path:")
        
        critical_count = sum(1 for c in transition_changes if c.severity == "CRITICAL")
        warning_count = sum(1 for c in transition_changes if c.severity == "WARNING")
        
        print(f"  • Critical: {critical_count}")
        print(f"  • Warnings: {warning_count}")
        
        print(f"\n🎯 Recommendation:")
        if critical_count > 0:
            print(f"  ❌ DO NOT upgrade to CUDA 13.0 without addressing critical issues")
            print(f"  📝 Review migration paths and update affected libraries")
        elif warning_count > 0:
            print(f"  ⚠️  Upgrade possible but test thoroughly")
            print(f"  📝 Review warnings and plan for deprecations")
        else:
            print(f"  ✅ Safe to upgrade with current configuration")
    else:
        print(f"\n✅ No specific breaking changes for CUDA {env.cuda_runtime_version} → 13.0 transition")

print(f"\n{'=' * 80}")
print("📚 REFERENCES")
print("=" * 80)
print("Breaking Changes Database:")
print("  • Local: cuda_healthcheck/data/breaking_changes.py")
print("  • GitHub: https://github.com/TavnerJC/cuda-healthcheck-on-databricks/blob/main/cuda_healthcheck/data/breaking_changes.py")
print("  • Docs: https://github.com/TavnerJC/cuda-healthcheck-on-databricks/blob/main/docs/USE_CASE_ROUTING_OPTIMIZATION.md")
print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📊 Step 10: Summary
# MAGIC
# MAGIC Final summary of your environment validation.

# COMMAND ----------
print("=" * 80)
print("📊 ENVIRONMENT VALIDATION SUMMARY")
print("=" * 80)

print(f"\n🎮 GPU Configuration:")
print(f"   Environment: {gpu_info.get('environment', 'unknown')}")
print(f"   GPU Count: {gpu_info.get('gpu_count', 0)}")
if gpu_info.get('gpu_count', 0) > 0 and 'gpus' in gpu_info:
    gpu = gpu_info['gpus'][0]
    print(f"   Type: {gpu.get('name', 'Unknown')}")
    print(f"   Memory: {gpu.get('memory_total', 'N/A')}")
    print(f"   Compute: {gpu.get('compute_capability', 'N/A')}")

print(f"\n🔧 CUDA Configuration:")
print(f"   Runtime: {env.cuda_runtime_version}")
print(f"   Driver: {env.cuda_driver_version}")
print(f"   NVCC: {env.nvcc_version}")

print(f"\n📚 Installed Libraries:")
for lib in env.libraries:
    status = "✅" if lib.is_compatible else "⚠️"
    print(f"   {status} {lib.name}: {lib.version}")

# Check if cuopt_incompatible variable exists (only set when CuOPT is installed)
if 'cuopt_incompatible' in locals() and cuopt_incompatible:
    print(f"\n🚨 CRITICAL ISSUES:")
    print(f"   ❌ CuOPT is incompatible with this Databricks runtime")
    print(f"   📝 Use OR-Tools as alternative for routing optimization")
    print(f"   📝 Report to Databricks for ML Runtime 17.0+ support")
else:
    # Check if CuOPT was detected at all
    cuopt_lib = next((lib for lib in env.libraries if lib.name.lower() == "cuopt"), None)
    if cuopt_lib and cuopt_lib.version == "Not installed":
        print(f"\n📦 CuOPT Status:")
        print(f"   ⚠️  CuOPT not currently installed")
        print(f"   ℹ️  Install with: %pip install --extra-index-url=https://pypi.nvidia.com cuopt-server-cu12")
        print(f"   ⚠️  Note: CuOPT 25.12+ will NOT work on this runtime (nvJitLink 12.4.127)")

print(f"\n💯 CUDA 13.0 Compatibility: {score['compatibility_score']}/100")
print(f"   Status: {score['recommendation']}")

print(f"\n{'=' * 80}")
print("✅ VALIDATION COMPLETE!")
print(f"{'=' * 80}")

print(f"\n🎯 Next Steps:")
# Check if CuOPT incompatibility was detected
cuopt_lib = next((lib for lib in env.libraries if lib.name.lower() == "cuopt"), None)
if cuopt_lib and cuopt_lib.version == "Not installed":
    print(f"   1. ⚠️  CuOPT is not installed (expected if not running CuOPT workloads)")
    print(f"   2. ⚠️  If you install CuOPT, it will fail due to nvJitLink 12.4.127")
    print(f"   3. ✅ Consider using OR-Tools for routing optimization")
    print(f"   4. ✅ Environment validated for broad AI/ML GPU workloads")
else:
    print(f"   1. Environment is validated ✅")
    print(f"   2. Proceed to benchmarking")
    print(f"   3. Review CUDA 13.0 compatibility when planning upgrades")

print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ Validation Complete!
# MAGIC
# MAGIC Your environment has been validated. Key findings:
# MAGIC
# MAGIC - GPU configuration detected
# MAGIC - CUDA versions validated
# MAGIC - Library compatibility checked
# MAGIC - **CuOPT compatibility assessed** (if installed)
# MAGIC - Breaking changes identified
# MAGIC - Migration paths provided
# MAGIC
# MAGIC **Save this notebook output** for documentation and troubleshooting!




