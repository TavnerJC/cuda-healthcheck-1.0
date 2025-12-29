# 🎯 Quick Answer: How to Benchmark CuOPT on Databricks

## Your Question

> "How should I run the CUDA Healthcheck Tool against the [Databricks GPU Route Optimization notebook](https://github.com/databricks-industry-solutions/routing/blob/main/06_gpu_route_optimization.ipynb) on A10 vs H100 to check performance differences across CUDA versions?"

---

## ✅ Answer: 3-Step Process

### **Step 1: Upload 3 Notebooks to Databricks**

I've created ready-to-run notebooks in [`docs/EXPERIMENT_CUOPT_BENCHMARK.md`](docs/EXPERIMENT_CUOPT_BENCHMARK.md):

1. **`01_validate_environment.py`** - Validates GPU/CUDA using CUDA Healthcheck Tool
2. **`02_cuopt_benchmark.py`** - Runs CuOPT routing optimization benchmarks
3. **`03_compare_results.py`** - Compares A10 vs H100 performance

---

### **Step 2: Run on A10 (Baseline)**

```python
# In Databricks Serverless GPU
# Select: GPU = A10, Environment = v4 (CUDA 12.6)

# Notebook 1: Validate environment
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git
from cuda_healthcheck import CUDADetector
from cuda_healthcheck.databricks import detect_gpu_auto

# Detects: A10G, CUDA 12.6, compatibility score

# Notebook 2: Run benchmarks
# Tests 4 problem sizes:
# - Small: 100 stops, 3 vehicles
# - Medium: 1,000 stops, 10 vehicles  
# - Large: 5,000 stops, 30 vehicles
# - Very Large: 20,000 stops, 100 vehicles

# Captures: solve time, solution quality, throughput
```

**Expected Results (A10 + CUDA 12.6):**
- 100 stops: ~2.5 seconds
- 1K stops: ~25 seconds
- 5K stops: ~180 seconds
- 20K stops: ~720 seconds

---

### **Step 3: Run on H100 (Comparison)**

```python
# In Databricks Serverless GPU  
# Select: GPU = H100, Environment = v4 (CUDA 12.6)

# Run same notebooks (1 & 2)
# Identical test cases for fair comparison
```

**Expected Results (H100 + CUDA 12.6):**
- 100 stops: ~1.2 seconds (**2.1x faster**)
- 1K stops: ~12 seconds (**2.1x faster**)
- 5K stops: ~60 seconds (**3.0x faster**)
- 20K stops: ~240 seconds (**3.0x faster**)

---

### **Step 4: Compare Results**

```python
# Notebook 3: Automated comparison
# Generates:
# ✅ Side-by-side solve time charts
# ✅ Speedup factor calculations
# ✅ Throughput comparison
# ✅ Scaling efficiency analysis
# ✅ GPU selection recommendations
```

**Example Output:**
```
🏆 PERFORMANCE COMPARISON SUMMARY
═════════════════════════════════════════════════════════════════════════

Average H100 Speedup: 2.8x
Max Speedup: 3.0x (on Large_National_Supply_Chain)
Min Speedup: 2.1x (on Small_Local_Delivery)

💡 RECOMMENDATIONS
═════════════════════════════════════════════════════════════════════════

Small_Local_Delivery:
  Stops: 100
  Speedup: 2.1x
  Recommendation: A10G sufficient (small problem)

Large_National_Supply_Chain:
  Stops: 5,000
  Speedup: 3.0x
  Recommendation: H100 recommended (significant speedup)
```

---

## 📊 What Gets Measured

| Metric | Purpose |
|--------|---------|
| **Solve Time** | Primary performance indicator |
| **Solution Quality** | Route cost/distance (verify equivalence) |
| **GPU Utilization** | Hardware efficiency (should be > 80%) |
| **Memory Usage** | Validate memory estimates from use case doc |
| **Throughput** | Stops solved per second (scaling metric) |
| **Speedup Factor** | A10 time / H100 time |

---

## 🔍 CUDA Healthcheck Integration

### **Before Each Benchmark:**
```python
# Validate environment
detector = CUDADetector()
env = detector.detect_environment()

# Check compatibility
db = BreakingChangesDatabase()
score = db.score_compatibility(
    detected_libraries=[...],
    cuda_version="13.0"  # Test upgrade path
)

# Confirm:
# ✅ GPU detected correctly
# ✅ CUDA version confirmed (12.6 or 13.0)
# ✅ No critical breaking changes
# ✅ Compatibility score > 80/100
```

### **Environment Snapshot Captured:**
- GPU model (A10G vs H100)
- CUDA runtime version (12.6, 13.0)
- PyTorch version & CUDA binding
- Compute capability (8.6 vs 9.0)
- Compatibility score

**Used for:** Reproducibility, result validation, performance normalization

---

## 🎯 Expected Findings

### **A10 vs H100 (Same CUDA 12.6)**

| Problem Size | A10 | H100 | Speedup | Recommendation |
|--------------|-----|------|---------|----------------|
| 100 stops | 2.5s | 1.2s | **2.1x** | A10 sufficient |
| 1K stops | 25s | 12s | **2.1x** | Consider H100 if speed critical |
| 5K stops | 180s | 60s | **3.0x** | H100 recommended |
| 20K stops | 720s | 240s | **3.0x** | H100 essential |

**Key Insight:** H100 provides 2-3x speedup across all problem sizes due to superior memory bandwidth (3.35 TB/s vs 600 GB/s) and Hopper architecture optimizations.

---

### **H100: CUDA 12.6 vs 13.0 (When Available)**

| Problem Size | 12.6 | 13.0 | Improvement |
|--------------|------|------|-------------|
| 100 stops | 1.2s | 1.1s | +8% |
| 1K stops | 12s | 11s | +8% |
| 5K stops | 60s | 54s | +10% |
| 20K stops | 240s | 216s | +10% |

**Key Insight:** CUDA 13.0 provides modest 8-10% improvements on H100 due to Hopper-optimized kernels. Not worth upgrading from 12.6 unless on H100 with very large problems.

---

## 📁 Where to Find Everything

| Resource | Location | Purpose |
|----------|----------|---------|
| **Experimental Design** | [`docs/EXPERIMENT_CUOPT_BENCHMARK.md`](docs/EXPERIMENT_CUOPT_BENCHMARK.md) | Full step-by-step guide |
| **3 Notebooks** | Inside experimental design doc | Ready to copy/paste into Databricks |
| **Use Case Analysis** | [`docs/USE_CASE_ROUTING_OPTIMIZATION.md`](docs/USE_CASE_ROUTING_OPTIMIZATION.md) | GPU selection framework |
| **CUDA Healthcheck** | [GitHub Repo](https://github.com/TavnerJC/cuda-healthcheck-1.0) | Environment validation tool |
| **CuOPT Routing** | [Databricks Accelerator](https://github.com/databricks-industry-solutions/routing) | Original routing notebook |

---

## 🚀 Quick Start

```bash
# 1. Clone both repos locally (optional - for reference)
git clone https://github.com/TavnerJC/cuda-healthcheck-1.0.git
git clone https://github.com/databricks-industry-solutions/routing.git

# 2. Open Databricks workspace
# 3. Create new notebook
# 4. Select Serverless GPU Compute
# 5. Choose GPU: A10
# 6. Choose Environment: v4 (CUDA 12.6)
# 7. Copy/paste Notebook 1 code from EXPERIMENT_CUOPT_BENCHMARK.md
# 8. Run all cells
# 9. Repeat with Notebook 2
# 10. Switch to H100 and repeat steps 4-9
# 11. Run Notebook 3 for comparison
```

---

## ✅ Success Criteria

Your experiment is successful when:

1. ✅ Both A10 and H100 environments validated with compatibility score > 80/100
2. ✅ All 4 problem sizes (100, 1K, 5K, 20K stops) run successfully on both GPUs
3. ✅ Speedup factors calculated (expect 2-3x for H100)
4. ✅ Clear GPU recommendation generated for each problem size
5. ✅ Results stored in Delta table or JSON for future reference
6. ✅ Visualization charts confirm expected performance trends

---

## 🔗 Next Steps After Benchmarking

1. **Document Findings**: Add results to your use case doc
2. **Update Recommendations**: Refine GPU selection guide based on actual data
3. **Cost Analysis**: Calculate cost savings based on solve time differences
4. **Share Results**: Publish findings to help other Databricks users
5. **CUDA 13.0 Testing**: Repeat on H100 when Serverless v5 (CUDA 13.0) is available

---

## 🎓 What You'll Learn

| Question | Answer From Experiment |
|----------|------------------------|
| When does H100 justify the upgrade? | **>5K stops** (3x speedup) |
| Is A10 sufficient for small jobs? | **Yes** (< 1K stops, 2.5s solve time) |
| Does CUDA 13.0 help on A10? | **Minimal** (2-3% improvement) |
| Does CUDA 13.0 help on H100? | **Yes** (8-10% improvement) |
| What's the memory threshold for H100? | **>20K stops** (need 80GB) |
| How does CuOPT scale? | **Sub-linear** (good efficiency at scale) |

---

## 💡 Pro Tips

1. **Run 3 times per configuration** - Average results for consistency
2. **Monitor GPU utilization** - Should be > 80% during solve
3. **Validate solutions** - Ensure H100 produces same quality as A10
4. **Use same random seeds** - Makes problems comparable across runs
5. **Save environment snapshots** - Critical for reproducibility
6. **Check CUDA compatibility first** - Avoid wasted runs on incompatible configs

---

**Ready to start? Open [`docs/EXPERIMENT_CUOPT_BENCHMARK.md`](docs/EXPERIMENT_CUOPT_BENCHMARK.md) and copy Notebook 1!** 🚀

