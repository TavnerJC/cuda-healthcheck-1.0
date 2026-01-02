# NeMo DataDesigner Feature Detection - Quick Reference

## Feature Detection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  NeMo DataDesigner Detection                    │
│                         (Layer 4)                               │
└─────────────────────────────────────────────────────────────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │  4 Detection Methods (Priority Order)   │
        └─────────────────────────────────────────┘
                              ▼
    ┌──────────┬──────────┬──────────┬──────────┐
    │  Config  │   Env    │ Packages │ Notebook │
    │  Files   │   Vars   │ Installed│  Cells   │
    └──────────┴──────────┴──────────┴──────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │     Detected Features (4 Types)         │
        └─────────────────────────────────────────┘
                              ▼
    ┌───────────────┬───────────────┬────────────┬───────────┐
    │ cloud_llm_   │ local_llm_   │  sampler_  │   seed_   │
    │  inference   │  inference   │ generation │ processing│
    └───────────────┴───────────────┴────────────┴───────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │   Validate CUDA Requirements            │
        └─────────────────────────────────────────┘
                              ▼
    ┌────────────┬────────────┬────────────┬────────────┐
    │  PyTorch   │   CUDA     │   CUDA     │    GPU     │
    │  Presence  │  Available │   Branch   │   Memory   │
    └────────────┴────────────┴────────────┴────────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │        Validation Status                │
        └─────────────────────────────────────────┘
                              ▼
    ┌──────────┬──────────┬──────────┬──────────┐
    │    OK    │ WARNING  │ BLOCKER  │ SKIPPED  │
    └──────────┴──────────┴──────────┴──────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │    Comprehensive Report with:           │
        │    • Summary (enabled, blockers, warns) │
        │    • Fix commands for blockers          │
        │    • Recommendations for warnings       │
        │    • Environment info                   │
        └─────────────────────────────────────────┘
```

---

## Feature Requirements Matrix

```
┌────────────────────────┬─────────┬──────────┬──────────────┬──────────────┐
│ Feature                │ PyTorch │   CUDA   │ CUDA Branch  │  GPU Memory  │
├────────────────────────┼─────────┼──────────┼──────────────┼──────────────┤
│ cloud_llm_inference    │    ❌   │    ❌    │     N/A      │     N/A      │
│ local_llm_inference    │    ✅   │    ✅    │ cu121, cu124 │   ≥40 GB     │
│ sampler_generation     │    ❌   │    ❌    │     N/A      │     N/A      │
│ seed_processing        │    ❌   │    ❌    │     N/A      │     N/A      │
└────────────────────────┴─────────┴──────────┴──────────────┴──────────────┘
```

---

## Detection Methods

### Method 1: Config File (HIGHEST PRIORITY)
```json
{
  "inference": {
    "mode": "local",          // or "cloud"
    "model": "llama-3.3-70b"
  },
  "samplers": {
    "enabled": ["category", "person", "uniform"]
  },
  "seed_data": {
    "enabled": true,
    "path": "/data/seeds"
  }
}
```

### Method 2: Environment Variables
```bash
export DATADESIGNER_INFERENCE_MODE=local  # or "cloud"
export DATADESIGNER_ENABLE_SAMPLERS=true
export DATADESIGNER_ENABLE_SEED_PROCESSING=true
```

### Method 3: Installed Packages
```python
# System checks for:
import nemo.datadesigner.cloud      # → cloud_llm_inference
import nemo.datadesigner.local      # → local_llm_inference
import nemo.datadesigner.samplers   # → sampler_generation
import nemo.datadesigner.seed       # → seed_processing
```

### Method 4: Notebook Cell Analysis (LOWEST PRIORITY)
```python
# System scans for:
from nemo.datadesigner.cloud import CloudLLM
from nemo.datadesigner.local import LocalLLM
sampler = CategorySampler()
loader = SeedDataLoader()
```

---

## Validation Status Flow

```
Feature Enabled?
      │
      ├─ NO  ──────────────► SKIPPED
      │
      └─ YES
            │
            ▼
      Requires PyTorch?
            │
            ├─ YES ─► PyTorch Installed?
            │              │
            │              ├─ NO ───► BLOCKER (fix: pip install torch)
            │              └─ YES ──► Continue
            │
            ├─ NO ──────────────────► Continue
            │
            ▼
      Requires CUDA?
            │
            ├─ YES ─► CUDA Available?
            │              │
            │              ├─ NO ───► BLOCKER (fix: use GPU cluster)
            │              └─ YES ──► Check CUDA Branch
            │                              │
            │                              ├─ Incompatible ─► BLOCKER (fix: pip install torch cu121)
            │                              └─ Compatible ───► Check GPU Memory
            │                                                      │
            │                                                      ├─ Below Min ─► WARNING
            │                                                      └─ Sufficient ► OK
            │
            └─ NO ──────────────────────────────────────────────► OK
```

---

## Common Scenarios

### Scenario 1: Cloud-Only (No GPU)
```
Environment:
  - Runtime: 14.3
  - PyTorch: Not installed
  - CUDA: Not available

Detection:
  export DATADESIGNER_INFERENCE_MODE=cloud

Result: ✅ OK (no requirements)
```

### Scenario 2: Local on Runtime 14.3 (Incompatible)
```
Environment:
  - Runtime: 14.3 (Driver 535, CUDA 12.0)
  - PyTorch: 2.4.1+cu124
  - GPU: A100 80GB

Detection:
  export DATADESIGNER_INFERENCE_MODE=local

Result: ❌ BLOCKER
  Issue: PyTorch cu124 incompatible with CUDA 12.0
  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu120
```

### Scenario 3: Local with Small GPU (Warning)
```
Environment:
  - Runtime: 16.4 (Driver 550, CUDA 12.4)
  - PyTorch: 2.5.1+cu124
  - GPU: T4 16GB

Detection:
  export DATADESIGNER_INFERENCE_MODE=local

Result: ⚠️ WARNING
  Issue: GPU memory (16 GB) below minimum (40 GB)
  Recommendation: Use A100 or quantization
```

### Scenario 4: Mixed Features (OK)
```
Environment:
  - Runtime: 16.4 (Driver 550, CUDA 12.4)
  - PyTorch: 2.5.1+cu124
  - GPU: A100 80GB

Detection:
  {
    "inference": {"mode": "local"},
    "samplers": {"enabled": ["category"]}
  }

Result: ✅ OK (all requirements met)
```

---

## API Quick Reference

### Basic Detection
```python
from cuda_healthcheck.nemo import detect_enabled_features

features = detect_enabled_features(
    check_env_vars=True,
    check_packages=True
)

for name, feature in features.items():
    if feature.is_enabled:
        print(f"✓ {name}: {feature.detection_method}")
```

### Validate Requirements
```python
from cuda_healthcheck.nemo import get_feature_validation_report

report = get_feature_validation_report(
    features=features,
    torch_version="2.4.1",
    torch_cuda_branch="cu124",
    cuda_available=True,
    gpu_memory_gb=80.0
)

print(f"Blockers: {report['summary']['blockers']}")
print(f"Warnings: {report['summary']['warnings']}")

for blocker in report['blockers']:
    print(f"\n❌ {blocker['feature']}")
    print(f"   Fix: {blocker['fix_commands'][0]}")
```

### Manual Validation
```python
from cuda_healthcheck.nemo import (
    DataDesignerFeature,
    FeatureRequirements,
    validate_feature_requirements
)

feature = DataDesignerFeature(
    feature_name="local_llm_inference",
    is_enabled=True,
    requirements=FeatureRequirements(
        feature_name="local_llm_inference",
        requires_torch=True,
        requires_cuda=True,
        compatible_cuda_branches=["cu121", "cu124"],
        min_gpu_memory_gb=40.0
    )
)

validated = validate_feature_requirements(
    feature=feature,
    torch_version="2.4.1",
    torch_cuda_branch="cu124",
    cuda_available=True,
    gpu_memory_gb=80.0
)

print(f"Status: {validated.validation_status}")
print(f"Message: {validated.validation_message}")
```

---

## Databricks Notebook Integration

### Step 11 Output Example
```
🔍 Detecting NeMo DataDesigner features...
================================================================================

📊 Feature Detection Results:
   Total features checked: 4
   Enabled features: 2

   Detected features:
      ✓ local_llm_inference
        Detection: environment_variable
        Description: GPU-based local LLM inference (e.g., Llama 3.3 70B)
      ✓ sampler_generation
        Detection: installed_package
        Description: Pure Python data samplers (category, person, uniform)

🔧 Validating Feature Requirements...
================================================================================

📋 Validation Summary:
   Enabled features: 2
   🚨 Blockers: 0
   ⚠️  Warnings: 1

⚠️  WARNINGS:
================================================================================

⚠️  Feature: local_llm_inference
   GPU memory (20.0 GB) is below recommended minimum (40.0 GB).
   Performance may be degraded or OOM errors may occur.
================================================================================

📊 Detailed Feature Status:
================================================================================

✅ local_llm_inference
   Status: WARNING
   Message: GPU memory (20.0 GB) is below recommended minimum (40.0 GB). Performance may be degraded or OOM errors may occur.
   Requirements:
      - PyTorch: Required
      - CUDA: Required
      - CUDA Branches: cu121, cu124
      - Min GPU Memory: 40.0 GB

✅ sampler_generation
   Status: OK
   Message: All requirements met
   Requirements:
      - PyTorch: Not required
      - CUDA: Not required
```

---

## Troubleshooting

### No features detected
**Cause:** No config, env vars, or installed packages  
**Fix:** Set environment variables:
```bash
export DATADESIGNER_INFERENCE_MODE=local
```

### BLOCKER: PyTorch missing
**Cause:** Feature requires PyTorch but not installed  
**Fix:** Install PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### BLOCKER: CUDA not available
**Cause:** Feature requires GPU but running on CPU cluster  
**Fix:** Switch to GPU-enabled cluster in Databricks

### BLOCKER: Incompatible CUDA branch
**Cause:** PyTorch CUDA branch doesn't match runtime  
**Fix:** Install compatible PyTorch:
```bash
# For Runtime 14.3 (CUDA 12.0)
pip install torch --index-url https://download.pytorch.org/whl/cu120

# For Runtime 15.2+ (CUDA 12.4)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### WARNING: Insufficient GPU memory
**Cause:** GPU has less than 40 GB for large models  
**Fix Options:**
1. Upgrade to A100 80GB
2. Use model quantization (4-bit/8-bit)
3. Use gradient checkpointing
4. Use cloud-based inference

---

## Testing

### Run All Tests
```bash
pytest tests/nemo/test_datadesigner_detector.py -v
```

**Expected:** 41 passed

### Run Specific Test
```bash
pytest tests/nemo/test_datadesigner_detector.py::TestValidateFeatureRequirements -v
```

### Test Coverage
```bash
pytest tests/nemo/test_datadesigner_detector.py --cov=cuda_healthcheck.nemo --cov-report=term-missing
```

---

## Files

### Implementation
- `cuda_healthcheck/nemo/__init__.py` - Module exports
- `cuda_healthcheck/nemo/datadesigner_detector.py` - Core logic (535 lines)

### Tests
- `tests/nemo/__init__.py` - Test module init
- `tests/nemo/test_datadesigner_detector.py` - Unit tests (564 lines, 41 tests)

### Documentation
- `docs/NEMO_DATADESIGNER_DETECTION.md` - Full documentation (520 lines)
- `DATADESIGNER_FEATURE_DETECTION_SUMMARY.md` - Implementation summary

### Notebook
- `notebooks/01_cuda_environment_validation_enhanced.py` - Step 11 added

---

## Version Info

**Package:** `cuda-healthcheck-on-databricks`  
**Version:** `0.5.0`  
**Git Commit:** `e15a721`  
**GitHub:** https://github.com/TavnerJC/cuda-healthcheck-on-databricks  
**Tests:** 41 passed (100%)  
**Lines Added:** 1,838  

---

## See Also

- [Full Documentation](docs/NEMO_DATADESIGNER_DETECTION.md)
- [Implementation Summary](DATADESIGNER_FEATURE_DETECTION_SUMMARY.md)
- [Databricks Runtime Detection](docs/DATABRICKS_RUNTIME_DETECTION.md)
- [Driver Version Mapping](docs/DRIVER_VERSION_MAPPING.md)
- [CUDA Package Parser](docs/CUDA_PACKAGE_PARSER.md)

