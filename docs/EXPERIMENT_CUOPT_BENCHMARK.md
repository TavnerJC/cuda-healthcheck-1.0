# 🧪 Experimental Design: CuOPT Routing Benchmark with CUDA Healthcheck

## 📋 Objective

Compare performance of [Databricks GPU Route Optimization](https://github.com/databricks-industry-solutions/routing/blob/main/06_gpu_route_optimization.ipynb) across:
- **GPU Architectures:** A10 (Ampere) vs H100 (Hopper)
- **CUDA Versions:** 12.6 (Serverless v4) vs 13.0 (future v5)
- **Problem Sizes:** Small (500 stops) to Large (50K stops)

**Tools:**
- [CUDA Healthcheck Tool](https://github.com/TavnerJC/cuda-healthcheck-1.0)
- [Databricks Routing Accelerator](https://github.com/databricks-industry-solutions/routing)
- NVIDIA CuOPT 25.10+

---

## 🎯 Experimental Design

### Phase 1: Environment Validation
Use CUDA Healthcheck to validate each test environment before benchmarking.

### Phase 2: Baseline Performance (A10 + CUDA 12.6)
Run routing optimization on A10 with current Serverless GPU v4 (CUDA 12.6).

### Phase 3: H100 Comparison (H100 + CUDA 12.6)
Run identical workload on H100 with CUDA 12.6 to measure GPU architecture impact.

### Phase 4: CUDA Version Impact (H100 + CUDA 13.0)
When available, test H100 with CUDA 13.0 to measure CUDA version impact.

---

## 📝 Step-by-Step Implementation

### **Step 1: Set Up Databricks Notebooks**

Create 3 notebooks in your Databricks workspace:

#### **Notebook 1: Environment Validator** (RECOMMENDED: Use Enhanced Version)

**🆕 Use the production-validated enhanced notebook:**

**File:** `notebooks/01_cuda_environment_validation_enhanced.py`  
**GitHub:** https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/notebooks/01_cuda_environment_validation_enhanced.py

**Features:**
- ✅ GPU detection (Classic & Serverless)
- ✅ CUDA environment detection
- ✅ **CuOPT compatibility checking** ⭐ (detects nvJitLink incompatibility)
- ✅ nvJitLink version validation
- ✅ Breaking changes analysis with migration paths
- ✅ **Production-validated on Databricks A10G**

**Quick Import:**
1. In Databricks, go to **Workspace** → **Import**
2. Select **URL**
3. Paste: `https://raw.githubusercontent.com/TavnerJC/cuda-healthcheck-1.0/main/notebooks/01_cuda_environment_validation_enhanced.py`
4. Attach to your GPU cluster and run!

**Quick Start:**
```python
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git
dbutils.library.restartPython()

# Then run the notebook cells step by step
# The enhanced notebook includes 8 comprehensive validation steps
```

**Expected Output:**
- GPU hardware detection
- CUDA version validation
- **CuOPT compatibility status** (if installed)
- **nvJitLink version check** (critical for CuOPT)
- CUDA 13.0 upgrade analysis
- Detailed breaking changes with GitHub references

---

**Legacy Notebook Code (Reference Only):**

<details>
<summary>Click to expand legacy notebook code</summary>

> **⚠️ Note:** The code below is kept for reference. Use the enhanced notebook above for production use.

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 🔍 CUDA Environment Validation
# MAGIC
# MAGIC Validate GPU and CUDA configuration before running CuOPT benchmarks.

# COMMAND ----------
# Install CUDA Healthcheck Tool
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git
dbutils.library.restartPython()

# COMMAND ----------
from cuda_healthcheck import CUDADetector, BreakingChangesDatabase
from cuda_healthcheck.databricks import detect_gpu_auto
import json
from datetime import datetime, timezone

# COMMAND ----------
# MAGIC %md
# MAGIC ## Detect GPU Configuration

# COMMAND ----------
# Auto-detect GPU (works on both Serverless and Classic)
gpu_info = detect_gpu_auto()

print("=" * 80)
print("🎮 GPU DETECTION RESULTS")
print("=" * 80)
print(f"Environment Type: {gpu_info['environment']}")
print(f"Detection Method: {gpu_info['method']}")
print(f"GPU Count: {gpu_info['gpu_count']}")

if gpu_info['gpu_count'] > 0:
    for gpu in gpu_info['gpus']:
        print(f"\n📊 GPU: {gpu['name']}")
        print(f"   Driver: {gpu['driver_version']}")
        print(f"   Memory: {gpu['memory_total']}")
        print(f"   Compute: {gpu['compute_capability']}")
        print(f"   UUID: {gpu['uuid']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Detect CUDA & Libraries

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

# Extract PyTorch info from libraries list
pytorch_lib = None
for lib in env.libraries:
    if lib.name.lower() == "pytorch":
        pytorch_lib = lib
        break

if pytorch_lib:
    print(f"PyTorch: {pytorch_lib.version}")
    print(f"PyTorch CUDA: {pytorch_lib.cuda_version}")
    print(f"Compatible: {pytorch_lib.is_compatible}")
else:
    print("PyTorch: Not installed")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compatibility Analysis

# COMMAND ----------
# Check CuOPT compatibility
db = BreakingChangesDatabase()

# Extract PyTorch version for compatibility check
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
if score['warning_issues'] > 0:
    print(f"\n⚠️  {score['warning_issues']} warnings - review before upgrading")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Detailed Compatibility Issues

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
        print(f"  GitHub: https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/cuda_healthcheck/data/breaking_changes.py")
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
print("  • GitHub: https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/cuda_healthcheck/data/breaking_changes.py")
print("  • Docs: https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/docs/USE_CASE_ROUTING_OPTIMIZATION.md")
print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Environment Snapshot

# COMMAND ----------
# Create environment snapshot for benchmark metadata
pytorch_lib = next((lib for lib in env.libraries if lib.name.lower() == "pytorch"), None)

environment_snapshot = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "gpu_info": gpu_info,
    "cuda_environment": {
        "runtime": env.cuda_runtime_version,
        "driver": env.cuda_driver_version,
        "nvcc": env.nvcc_version,
        "pytorch": pytorch_lib.version if pytorch_lib else None,
        "pytorch_cuda": pytorch_lib.cuda_version if pytorch_lib else None,
    },
    "compatibility_score": score['compatibility_score'],
    "gpu_architecture": gpu_info['gpus'][0]['name'] if gpu_info['gpu_count'] > 0 else "unknown",
    "compute_capability": gpu_info['gpus'][0]['compute_capability'] if gpu_info['gpu_count'] > 0 else "unknown",
}

# Store as widget for next notebook
dbutils.jobs.taskValues.set(key="environment", value=json.dumps(environment_snapshot))

print("✅ Environment snapshot saved to task values")
print(f"\nGPU: {environment_snapshot['gpu_architecture']}")
print(f"CUDA: {environment_snapshot['cuda_environment']['runtime']}")
print(f"Compatibility Score: {environment_snapshot['compatibility_score']}/100")
```

</details>

---

#### **Notebook 2: CuOPT Benchmark Runner** (`02_cuopt_benchmark.py`)

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 CuOPT Routing Benchmark
# MAGIC
# MAGIC Run vehicle routing optimization with performance timing.

# COMMAND ----------
# Install dependencies
%pip install cuopt-cu12  # Use cu12 for CUDA 12.x, cu13 for CUDA 13.x
%pip install pandas numpy matplotlib
dbutils.library.restartPython()

# COMMAND ----------
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt

# Import CuOPT (this will fail gracefully if not compatible)
try:
    from cuopt import routing
    cuopt_available = True
    print("✅ CuOPT imported successfully")
except Exception as e:
    cuopt_available = False
    print(f"❌ CuOPT import failed: {e}")
    dbutils.notebook.exit(json.dumps({"error": "CuOPT not available", "message": str(e)}))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Environment Snapshot

# COMMAND ----------
# Retrieve environment from previous notebook
try:
    environment_str = dbutils.jobs.taskValues.get(taskKey="01_validate_environment", 
                                                    key="environment", 
                                                    debugValue="{}")
    environment = json.loads(environment_str)
    print(f"Testing on: {environment['gpu_architecture']}")
    print(f"CUDA Version: {environment['cuda_environment']['runtime']}")
except Exception as e:
    print(f"⚠️  Could not load environment: {e}")
    environment = {"gpu_architecture": "unknown", "cuda_environment": {"runtime": "unknown"}}

# COMMAND ----------
# MAGIC %md
# MAGIC ## Define Benchmark Problems

# COMMAND ----------
# Define test cases with varying complexity
test_cases = [
    {
        "name": "Small_Local_Delivery",
        "num_stops": 100,
        "num_vehicles": 3,
        "time_windows": True,
        "capacity_constraints": True,
    },
    {
        "name": "Medium_Regional_Distribution",
        "num_stops": 1000,
        "num_vehicles": 10,
        "time_windows": True,
        "capacity_constraints": True,
    },
    {
        "name": "Large_National_Supply_Chain",
        "num_stops": 5000,
        "num_vehicles": 30,
        "time_windows": True,
        "capacity_constraints": True,
    },
    {
        "name": "VeryLarge_Global_Logistics",
        "num_stops": 20000,
        "num_vehicles": 100,
        "time_windows": True,
        "capacity_constraints": True,
    },
]

# COMMAND ----------
# MAGIC %md
# MAGIC ## Generate Synthetic Routing Data

# COMMAND ----------
def generate_routing_problem(num_stops, num_vehicles, seed=42):
    """
    Generate synthetic vehicle routing problem data.
    
    Args:
        num_stops: Number of delivery locations
        num_vehicles: Number of vehicles available
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with problem data (coordinates, demands, time windows, etc.)
    """
    np.random.seed(seed)
    
    # Generate random coordinates (lat/lon style)
    locations = np.random.rand(num_stops + 1, 2) * 100  # +1 for depot
    
    # Generate demands (packages to deliver)
    demands = np.random.randint(1, 10, size=num_stops + 1)
    demands[0] = 0  # Depot has no demand
    
    # Generate time windows (in minutes from start)
    earliest = np.random.randint(0, 480, size=num_stops + 1)  # 8 hours = 480 min
    latest = earliest + np.random.randint(30, 120, size=num_stops + 1)
    
    # Service times (minutes to complete delivery)
    service_times = np.random.randint(5, 20, size=num_stops + 1)
    service_times[0] = 0  # No service time at depot
    
    # Calculate distance matrix (Euclidean)
    from scipy.spatial.distance import cdist
    distance_matrix = cdist(locations, locations, metric='euclidean')
    
    return {
        "locations": locations,
        "demands": demands,
        "time_windows_earliest": earliest,
        "time_windows_latest": latest,
        "service_times": service_times,
        "distance_matrix": distance_matrix,
        "num_vehicles": num_vehicles,
        "vehicle_capacity": 50,  # Total packages per vehicle
    }

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run Benchmarks

# COMMAND ----------
def run_cuopt_benchmark(problem_data, problem_name):
    """
    Run CuOPT solver and measure performance.
    
    Returns:
        Dictionary with timing and solution quality metrics
    """
    print(f"\n{'=' * 80}")
    print(f"🚀 Running: {problem_name}")
    print(f"{'=' * 80}")
    
    start_time = time.time()
    
    try:
        # Create CuOPT DataModel
        data_model = routing.DataModel(
            n_locations=len(problem_data['locations']),
            n_vehicles=problem_data['num_vehicles']
        )
        
        # Set cost matrix (distance)
        data_model.add_cost_matrix(problem_data['distance_matrix'])
        
        # Set demands and capacity
        data_model.add_capacity_dimension(
            "demand",
            problem_data['demands'],
            [problem_data['vehicle_capacity']] * problem_data['num_vehicles']
        )
        
        # Set time windows
        data_model.add_time_dimension(
            "time",
            problem_data['time_windows_earliest'],
            problem_data['time_windows_latest'],
            problem_data['service_times']
        )
        
        # Create solver
        solver_settings = routing.SolverSettings()
        solver_settings.time_limit = 300  # 5 minutes max
        
        # Solve
        solve_start = time.time()
        routing_solution = routing.Solve(data_model, solver_settings)
        solve_time = time.time() - solve_start
        
        # Extract metrics
        total_time = time.time() - start_time
        
        result = {
            "problem_name": problem_name,
            "status": "success",
            "solve_time_seconds": solve_time,
            "total_time_seconds": total_time,
            "solution_cost": routing_solution.solution_cost,
            "num_routes": len(routing_solution.vehicle_data),
            "num_stops": len(problem_data['locations']) - 1,
            "num_vehicles": problem_data['num_vehicles'],
        }
        
        print(f"✅ Status: {result['status']}")
        print(f"⏱️  Solve Time: {result['solve_time_seconds']:.2f}s")
        print(f"💰 Solution Cost: {result['solution_cost']:.2f}")
        print(f"🚚 Routes Used: {result['num_routes']}/{result['num_vehicles']}")
        
        return result
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Error: {e}")
        
        return {
            "problem_name": problem_name,
            "status": "failed",
            "error": str(e),
            "total_time_seconds": total_time,
        }

# COMMAND ----------
# Run all test cases
results = []

for test_case in test_cases:
    print(f"\n\n🔄 Generating problem: {test_case['name']}")
    problem_data = generate_routing_problem(
        num_stops=test_case['num_stops'],
        num_vehicles=test_case['num_vehicles']
    )
    
    result = run_cuopt_benchmark(problem_data, test_case['name'])
    
    # Add environment metadata
    result['gpu_architecture'] = environment['gpu_architecture']
    result['cuda_version'] = environment['cuda_environment']['runtime']
    result['timestamp'] = datetime.now(timezone.utc).isoformat()
    
    results.append(result)
    
    # Save intermediate results
    dbutils.jobs.taskValues.set(
        key=f"result_{test_case['name']}", 
        value=json.dumps(result)
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Display Results

# COMMAND ----------
# Create results DataFrame
results_df = pd.DataFrame(results)
display(results_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Visualize Performance

# COMMAND ----------
# Plot solve times by problem size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Solve time vs problem size
successful_results = results_df[results_df['status'] == 'success']
ax1.plot(successful_results['num_stops'], 
         successful_results['solve_time_seconds'], 
         marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Stops', fontsize=12)
ax1.set_ylabel('Solve Time (seconds)', fontsize=12)
ax1.set_title(f'CuOPT Performance on {environment["gpu_architecture"]}', fontsize=14)
ax1.grid(True, alpha=0.3)

# Throughput (stops per second)
successful_results['throughput'] = successful_results['num_stops'] / successful_results['solve_time_seconds']
ax2.bar(range(len(successful_results)), successful_results['throughput'], color='skyblue')
ax2.set_xlabel('Problem Index', fontsize=12)
ax2.set_ylabel('Throughput (stops/sec)', fontsize=12)
ax2.set_title('Optimization Throughput', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Results

# COMMAND ----------
# Save complete results
benchmark_summary = {
    "environment": environment,
    "results": results,
    "summary": {
        "total_tests": len(results),
        "successful_tests": len([r for r in results if r['status'] == 'success']),
        "failed_tests": len([r for r in results if r['status'] == 'failed']),
        "avg_solve_time": results_df[results_df['status'] == 'success']['solve_time_seconds'].mean(),
    }
}

dbutils.jobs.taskValues.set(key="benchmark_summary", value=json.dumps(benchmark_summary))

print("=" * 80)
print("📊 BENCHMARK SUMMARY")
print("=" * 80)
print(f"GPU: {environment['gpu_architecture']}")
print(f"CUDA: {environment['cuda_environment']['runtime']}")
print(f"Total Tests: {benchmark_summary['summary']['total_tests']}")
print(f"Successful: {benchmark_summary['summary']['successful_tests']}")
print(f"Failed: {benchmark_summary['summary']['failed_tests']}")
print(f"Avg Solve Time: {benchmark_summary['summary']['avg_solve_time']:.2f}s")
print("=" * 80)
```

---

#### **Notebook 3: Cross-GPU Comparison** (`03_compare_results.py`)

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 📊 Cross-GPU Performance Comparison
# MAGIC
# MAGIC Compare A10 vs H100 benchmark results.

# COMMAND ----------
%pip install pandas matplotlib seaborn
dbutils.library.restartPython()

# COMMAND ----------
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Benchmark Results
# MAGIC
# MAGIC Run this notebook AFTER completing benchmarks on both A10 and H100.

# COMMAND ----------
# Load results from both GPU runs
# NOTE: You'll need to manually paste results or load from Delta table

# Example structure (replace with actual data):
a10_results = {
    "gpu": "A10G",
    "cuda": "12.6",
    "results": [
        {"problem_name": "Small_Local_Delivery", "num_stops": 100, "solve_time_seconds": 2.5},
        {"problem_name": "Medium_Regional_Distribution", "num_stops": 1000, "solve_time_seconds": 25.3},
        {"problem_name": "Large_National_Supply_Chain", "num_stops": 5000, "solve_time_seconds": 180.7},
    ]
}

h100_results = {
    "gpu": "H100",
    "cuda": "12.6",
    "results": [
        {"problem_name": "Small_Local_Delivery", "num_stops": 100, "solve_time_seconds": 1.2},
        {"problem_name": "Medium_Regional_Distribution", "num_stops": 1000, "solve_time_seconds": 11.8},
        {"problem_name": "Large_National_Supply_Chain", "num_stops": 5000, "solve_time_seconds": 62.3},
    ]
}

# COMMAND ----------
# MAGIC %md
# MAGIC ## Create Comparison DataFrame

# COMMAND ----------
# Combine results
a10_df = pd.DataFrame(a10_results['results'])
a10_df['gpu'] = 'A10G'

h100_df = pd.DataFrame(h100_results['results'])
h100_df['gpu'] = 'H100'

combined_df = pd.concat([a10_df, h100_df], ignore_index=True)

# Calculate speedup
comparison = a10_df.merge(h100_df, on='problem_name', suffixes=('_a10', '_h100'))
comparison['speedup'] = comparison['solve_time_seconds_a10'] / comparison['solve_time_seconds_h100']
comparison['time_saved_seconds'] = comparison['solve_time_seconds_a10'] - comparison['solve_time_seconds_h100']

display(comparison[['problem_name', 'num_stops_a10', 'solve_time_seconds_a10', 
                     'solve_time_seconds_h100', 'speedup', 'time_saved_seconds']])

# COMMAND ----------
# MAGIC %md
# MAGIC ## Visualize Comparison

# COMMAND ----------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Side-by-side solve times
ax1 = axes[0, 0]
x = range(len(comparison))
width = 0.35
ax1.bar([i - width/2 for i in x], comparison['solve_time_seconds_a10'], 
        width, label='A10G', color='skyblue')
ax1.bar([i + width/2 for i in x], comparison['solve_time_seconds_h100'], 
        width, label='H100', color='coral')
ax1.set_xlabel('Problem')
ax1.set_ylabel('Solve Time (seconds)')
ax1.set_title('Solve Time Comparison: A10G vs H100')
ax1.set_xticks(x)
ax1.set_xticklabels([name.replace('_', '\n') for name in comparison['problem_name']], 
                      rotation=0, fontsize=8)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 2. Speedup factors
ax2 = axes[0, 1]
ax2.bar(range(len(comparison)), comparison['speedup'], color='green', alpha=0.7)
ax2.axhline(y=1.0, color='red', linestyle='--', label='No speedup')
ax2.set_xlabel('Problem')
ax2.set_ylabel('Speedup Factor (A10 time / H100 time)')
ax2.set_title('H100 Speedup over A10G')
ax2.set_xticks(range(len(comparison)))
ax2.set_xticklabels([name.replace('_', '\n') for name in comparison['problem_name']], 
                      rotation=0, fontsize=8)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Throughput comparison
ax3 = axes[1, 0]
combined_df['throughput'] = combined_df['num_stops'] / combined_df['solve_time_seconds']
sns.barplot(data=combined_df, x='problem_name', y='throughput', hue='gpu', ax=ax3)
ax3.set_xlabel('Problem')
ax3.set_ylabel('Throughput (stops/second)')
ax3.set_title('Optimization Throughput')
ax3.set_xticklabels([name.replace('_', '\n') for name in comparison['problem_name']], 
                      rotation=0, fontsize=8)
ax3.legend(title='GPU')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Scaling efficiency
ax4 = axes[1, 1]
for gpu in ['A10G', 'H100']:
    gpu_data = combined_df[combined_df['gpu'] == gpu].sort_values('num_stops')
    ax4.plot(gpu_data['num_stops'], gpu_data['solve_time_seconds'], 
             marker='o', linewidth=2, markersize=8, label=gpu)
ax4.set_xlabel('Number of Stops')
ax4.set_ylabel('Solve Time (seconds)')
ax4.set_title('Scaling Behavior')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Performance Summary

# COMMAND ----------
print("=" * 80)
print("🏆 PERFORMANCE COMPARISON SUMMARY")
print("=" * 80)
print(f"\nAverage H100 Speedup: {comparison['speedup'].mean():.2f}x")
print(f"Max Speedup: {comparison['speedup'].max():.2f}x (on {comparison.loc[comparison['speedup'].idxmax(), 'problem_name']})")
print(f"Min Speedup: {comparison['speedup'].min():.2f}x (on {comparison.loc[comparison['speedup'].idxmin(), 'problem_name']})")
print(f"\nTotal Time Saved (all problems): {comparison['time_saved_seconds'].sum():.2f}s")
print(f"Average Time Saved per Problem: {comparison['time_saved_seconds'].mean():.2f}s")

print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS")
print("=" * 80)

for _, row in comparison.iterrows():
    if row['num_stops_a10'] < 1000:
        rec = "A10G sufficient (small problem)"
    elif row['speedup'] < 1.5:
        rec = "A10G recommended (marginal speedup)"
    elif row['speedup'] < 2.5:
        rec = "Consider H100 if speed critical"
    else:
        rec = "H100 recommended (significant speedup)"
    
    print(f"\n{row['problem_name']}:")
    print(f"  Stops: {row['num_stops_a10']}")
    print(f"  Speedup: {row['speedup']:.2f}x")
    print(f"  Recommendation: {rec}")

print("\n" + "=" * 80)
```

---

## 🎯 **Execution Plan**

### **Run 1: A10 + CUDA 12.6 (Baseline)**

1. **Create Serverless GPU Notebook** in Databricks
2. **Select:** `Environment: v4` (CUDA 12.6) + `GPU: A10`
3. **Run Notebook 1:** Validate environment
4. **Run Notebook 2:** Execute benchmarks
5. **Export results** to Delta table or JSON

### **Run 2: H100 + CUDA 12.6 (Architecture Comparison)**

1. **Create new Serverless GPU Notebook**
2. **Select:** `Environment: v4` (CUDA 12.6) + `GPU: H100`
3. **Run Notebook 1:** Validate environment
4. **Run Notebook 2:** Execute identical benchmarks
5. **Export results**

### **Run 3: H100 + CUDA 13.0 (Future - CUDA Version Comparison)**

1. **Wait for Serverless GPU v5** (CUDA 13.0 availability)
2. **Select:** `Environment: v5` (CUDA 13.0) + `GPU: H100`
3. **Run Notebook 1:** Validate environment
4. **Run Notebook 2:** Execute benchmarks
5. **Compare** CUDA 12.6 vs 13.0 on same hardware

### **Analysis: Cross-Run Comparison**

1. **Run Notebook 3** with collected results
2. **Generate comparison charts**
3. **Document findings**

---

## 📊 **Expected Metrics to Capture**

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Solve Time** | Total time to find optimal solution | Primary performance metric |
| **Solution Quality** | Total route cost/distance | Verify solution equivalence |
| **GPU Utilization** | % GPU usage during solve | Hardware efficiency |
| **Memory Usage** | Peak GPU memory (GB) | Validate memory estimates |
| **Throughput** | Stops solved per second | Scaling efficiency |
| **Setup Time** | Problem loading + data transfer | Overhead analysis |

---

## 📈 **Expected Results**

Based on the [USE_CASE_ROUTING_OPTIMIZATION.md](USE_CASE_ROUTING_OPTIMIZATION.md) analysis:

### **A10 vs H100 (Same CUDA 12.6)**

| Problem Size | A10 Time | H100 Time | Speedup |
|--------------|----------|-----------|---------|
| 100 stops | 2.5s | 1.2s | **2.1x** |
| 1K stops | 25s | 12s | **2.1x** |
| 5K stops | 180s | 60s | **3.0x** |
| 20K stops | 720s | 240s | **3.0x** |

### **H100: CUDA 12.6 vs 13.0**

| Problem Size | 12.6 Time | 13.0 Time | Improvement |
|--------------|-----------|-----------|-------------|
| 100 stops | 1.2s | 1.1s | **+8%** |
| 1K stops | 12s | 11s | **+8%** |
| 5K stops | 60s | 54s | **+10%** |
| 20K stops | 240s | 216s | **+10%** |

---

## 💾 **Storing Results**

### **Option 1: Delta Table (Recommended)**

```sql
CREATE TABLE IF NOT EXISTS main.benchmarks.cuopt_performance (
  test_id STRING,
  timestamp TIMESTAMP,
  gpu_architecture STRING,
  cuda_version STRING,
  problem_name STRING,
  num_stops INT,
  num_vehicles INT,
  solve_time_seconds DOUBLE,
  solution_cost DOUBLE,
  status STRING
)
USING DELTA
PARTITIONED BY (gpu_architecture, cuda_version);
```

### **Option 2: Export to JSON**

```python
# In Notebook 2
with open("/dbfs/mnt/benchmarks/results_a10_cuda126.json", "w") as f:
    json.dump(benchmark_summary, f, indent=2)
```

---

## 🔍 **Validation Checklist**

Before benchmarking:
- [ ] CUDA Healthcheck Tool installed
- [ ] GPU detected correctly (nvidia-smi)
- [ ] CuOPT package installed (`import cuopt` succeeds)
- [ ] CUDA version confirmed (12.6 or 13.0)
- [ ] Compatibility score > 80/100
- [ ] No critical breaking changes

During benchmarking:
- [ ] All test cases run successfully
- [ ] GPU utilization > 80% (not CPU-bound)
- [ ] No out-of-memory errors
- [ ] Solution quality consistent across GPUs

After benchmarking:
- [ ] Results saved to persistent storage
- [ ] Speedup factors calculated
- [ ] Visualizations generated
- [ ] Recommendations documented

---

## 🚀 **Quick Start Command**

```bash
# Clone both repositories
git clone https://github.com/TavnerJC/cuda-healthcheck-1.0.git
git clone https://github.com/databricks-industry-solutions/routing.git

# Upload notebooks to Databricks
databricks workspace import 01_validate_environment.py /Users/your-email/experiments/
databricks workspace import 02_cuopt_benchmark.py /Users/your-email/experiments/
databricks workspace import 03_compare_results.py /Users/your-email/experiments/

# Run on A10
databricks runs submit --json '{
  "run_name": "cuopt_benchmark_a10",
  "tasks": [
    {"task_key": "validate", "notebook_task": {"notebook_path": "/Users/your-email/experiments/01_validate_environment"}},
    {"task_key": "benchmark", "depends_on": [{"task_key": "validate"}], "notebook_task": {"notebook_path": "/Users/your-email/experiments/02_cuopt_benchmark"}}
  ],
  "new_cluster": {"spark_version": "latest", "node_type_id": "g4dn.xlarge"}
}'

# Run on H100
databricks runs submit --json '{
  "run_name": "cuopt_benchmark_h100",
  "tasks": [...],
  "new_cluster": {"node_type_id": "g6.xlarge"}  # H100 instance
}'
```

---

## 📚 **References**

1. [CUDA Healthcheck Tool (GitHub)](https://github.com/TavnerJC/cuda-healthcheck-1.0)
2. [Databricks Routing Accelerator (GitHub)](https://github.com/databricks-industry-solutions/routing)
3. [GPU Route Optimization Notebook](https://github.com/databricks-industry-solutions/routing/blob/main/06_gpu_route_optimization.ipynb)
4. [NVIDIA CuOPT Documentation](https://docs.nvidia.com/cuopt/)
5. [Databricks Serverless GPU Docs](https://docs.databricks.com/aws/en/compute/serverless/gpu)

---

## ✅ **Success Criteria**

Your experiment is successful when:
1. ✅ All environments validated with CUDA Healthcheck Tool
2. ✅ Identical problems run on A10 and H100
3. ✅ Performance metrics captured for all problem sizes
4. ✅ Speedup factors calculated and visualized
5. ✅ Clear GPU selection recommendation for each problem size
6. ✅ Results published for future reference

---

**Ready to benchmark? Start with Notebook 1 on Databricks Serverless GPU (A10, Environment v4)!** 🚀

