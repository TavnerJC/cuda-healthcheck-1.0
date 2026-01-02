# NeMo DataDesigner Feature Detection and Validation

Automatically detects which NeMo DataDesigner features are enabled in your environment and validates that the necessary CUDA/PyTorch requirements are met.

## Overview

NeMo DataDesigner has multiple features with different CUDA requirements. This module helps you:

1. **Auto-detect** which features are enabled in your environment
2. **Validate** that your CUDA/PyTorch configuration meets the requirements
3. **Get actionable fix commands** for any blockers or warnings

## Supported Features

### 1. `cloud_llm_inference` - API-based LLM Inference

**Requirements:**
- PyTorch: ❌ Not required
- CUDA: ❌ Not required
- GPU Memory: N/A

**Use Case:** Using build.nvidia.com, OpenAI, or other cloud APIs for inference.

**Example:**
```python
from nemo.datadesigner.cloud import CloudLLM
llm = CloudLLM(api_key="...")
response = llm.generate("Translate this text...")
```

---

### 2. `local_llm_inference` - Local GPU Inference

**Requirements:**
- PyTorch: ✅ Required
- CUDA: ✅ Required (`cu121` or `cu124`)
- GPU Memory: ≥40 GB (recommended for Llama 3.3 70B)

**Use Case:** Running local Llama 3.3 70B or other large models on your GPU.

**Example:**
```python
from nemo.datadesigner.local import LocalLLM
llm = LocalLLM(model="llama-3.3-70b")
response = llm.generate("Translate this text...")
```

**Common Errors:**
- `RuntimeError: CUDA not available` → You need a GPU-enabled cluster
- `CUDA error: out of memory` → You need more GPU memory (80GB A100 recommended)
- `undefined symbol: __nvJitLinkAddData_12_4` → cuBLAS/nvJitLink version mismatch

---

### 3. `sampler_generation` - Data Samplers (Pure Python)

**Requirements:**
- PyTorch: ❌ Not required
- CUDA: ❌ Not required
- GPU Memory: N/A

**Use Case:** Using category, person, uniform, or other samplers for data generation.

**Example:**
```python
from nemo.datadesigner.samplers import CategorySampler, PersonSampler
category_sampler = CategorySampler()
person_sampler = PersonSampler()

category = category_sampler.sample()
name = person_sampler.sample()
```

---

### 4. `seed_processing` - Seed Data Loading

**Requirements:**
- PyTorch: ❌ Not required
- CUDA: ❌ Not required
- GPU Memory: N/A

**Use Case:** Loading and processing seed data for data generation pipelines.

**Example:**
```python
from nemo.datadesigner.seed import SeedDataLoader
loader = SeedDataLoader()
seed_data = loader.load("/data/seeds.json")
```

---

## Detection Methods

The system uses **4 detection methods** in priority order:

### 1. Config Files (Highest Priority)

**Supported formats:** JSON, YAML

**Example config (`datadesigner_config.json`):**
```json
{
  "inference": {
    "mode": "local",  // or "cloud"
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

### 2. Environment Variables

```bash
# Set inference mode
export DATADESIGNER_INFERENCE_MODE=local  # or "cloud"

# Enable samplers
export DATADESIGNER_ENABLE_SAMPLERS=true

# Enable seed processing
export DATADESIGNER_ENABLE_SEED_PROCESSING=true
```

### 3. Installed Packages

The system checks for:
- `nemo.datadesigner.cloud` → enables `cloud_llm_inference`
- `nemo.datadesigner.local` → enables `local_llm_inference`
- `nemo.datadesigner.samplers` → enables `sampler_generation`
- `nemo.datadesigner.seed` → enables `seed_processing`

### 4. Notebook Cell Analysis (Lowest Priority)

Scans notebook code for import patterns like:
```python
from nemo.datadesigner.cloud import CloudLLM
from nemo.datadesigner.local import LocalLLM
```

---

## API Usage

### Basic Detection

```python
from cuda_healthcheck.nemo import detect_enabled_features

# Auto-detect all features
features = detect_enabled_features(
    check_env_vars=True,
    check_packages=True,
)

# Check what was detected
for feature_name, feature in features.items():
    if feature.is_enabled:
        print(f"✓ {feature_name}")
        print(f"  Detected via: {feature.detection_method}")
```

### Validate Requirements

```python
from cuda_healthcheck.nemo import (
    detect_enabled_features,
    get_feature_validation_report,
)

# Detect features
features = detect_enabled_features()

# Validate against current environment
report = get_feature_validation_report(
    features=features,
    torch_version="2.4.1",
    torch_cuda_branch="cu124",
    cuda_available=True,
    gpu_memory_gb=80.0,
)

# Check summary
print(f"Enabled: {report['summary']['enabled_features']}")
print(f"Blockers: {report['summary']['blockers']}")
print(f"Warnings: {report['summary']['warnings']}")

# Show blockers
for blocker in report['blockers']:
    print(f"\n❌ {blocker['feature']}")
    print(f"   Issue: {blocker['message']}")
    print(f"   Fix:")
    for cmd in blocker['fix_commands']:
        print(f"      {cmd}")
```

### Manual Feature Validation

```python
from cuda_healthcheck.nemo import (
    DataDesignerFeature,
    FeatureRequirements,
    validate_feature_requirements,
)

# Create feature
feature = DataDesignerFeature(
    feature_name="local_llm_inference",
    is_enabled=True,
    requirements=FeatureRequirements(
        feature_name="local_llm_inference",
        requires_torch=True,
        requires_cuda=True,
        compatible_cuda_branches=["cu121", "cu124"],
        min_gpu_memory_gb=40.0,
    ),
)

# Validate
validated = validate_feature_requirements(
    feature=feature,
    torch_version="2.4.1",
    torch_cuda_branch="cu124",
    cuda_available=True,
    gpu_memory_gb=80.0,
)

print(f"Status: {validated.validation_status}")  # OK, BLOCKER, WARNING
print(f"Message: {validated.validation_message}")
```

---

## Validation Report Structure

```python
{
    "features": {
        "local_llm_inference": DataDesignerFeature(...),
        "cloud_llm_inference": DataDesignerFeature(...),
        # ...
    },
    "summary": {
        "total_features": 4,
        "enabled_features": 2,
        "blockers": 0,
        "warnings": 1,
    },
    "blockers": [
        {
            "feature": "local_llm_inference",
            "message": "PyTorch CUDA branch cu120 is not compatible. Required: cu121, cu124",
            "fix_commands": [
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            ]
        }
    ],
    "warnings": [
        {
            "feature": "local_llm_inference",
            "message": "GPU memory (20.0 GB) is below recommended minimum (40.0 GB). Performance may be degraded or OOM errors may occur."
        }
    ],
    "environment": {
        "torch_version": "2.4.1",
        "torch_cuda_branch": "cu124",
        "cuda_available": True,
        "gpu_memory_gb": 80.0,
    }
}
```

---

## Common Scenarios

### Scenario 1: Cloud-only inference (No GPU required)

**Config:**
```json
{
  "inference": {"mode": "cloud"}
}
```

**Validation Result:** ✅ OK (no CUDA requirements)

---

### Scenario 2: Local inference on Databricks Runtime 14.3

**Config:**
```json
{
  "inference": {"mode": "local", "model": "llama-3.3-70b"}
}
```

**Environment:**
- Runtime: 14.3 (Driver 535, CUDA 12.0)
- PyTorch: 2.4.1+cu124

**Validation Result:** ❌ BLOCKER

**Issue:** PyTorch `cu124` is incompatible with Runtime 14.3 (Driver 535 only supports CUDA 12.0)

**Fix Options:**
1. Downgrade PyTorch to `cu120`:
   ```bash
   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu120
   ```
2. Upgrade Databricks runtime to 15.2+ (supports CUDA 12.4)

---

### Scenario 3: Local inference with insufficient GPU memory

**Config:**
```json
{
  "inference": {"mode": "local", "model": "llama-3.3-70b"}
}
```

**Environment:**
- Runtime: 15.2 (Driver 550, CUDA 12.4)
- PyTorch: 2.4.1+cu124
- GPU: NVIDIA T4 (16 GB)

**Validation Result:** ⚠️ WARNING

**Issue:** GPU memory (16 GB) is below recommended minimum (40 GB)

**Recommendations:**
1. Use A100 (40GB or 80GB) or A10G (24GB)
2. Use model quantization (4-bit or 8-bit)
3. Use gradient checkpointing to reduce memory
4. Consider cloud-based inference instead

---

### Scenario 4: Mixed features (samplers + local inference)

**Config:**
```json
{
  "inference": {"mode": "local"},
  "samplers": {"enabled": ["category", "person"]}
}
```

**Environment:**
- Runtime: 16.4 (Driver 550, CUDA 12.4)
- PyTorch: 2.5.1+cu124
- GPU: A100 80GB

**Validation Result:** ✅ OK

**Note:** Samplers have no CUDA requirements, so even if local inference had issues, samplers would still work.

---

## Databricks Notebook Integration

The enhanced CUDA healthcheck notebook (`01_cuda_environment_validation_enhanced.py`) includes automatic DataDesigner feature detection in **Step 11**.

**What it does:**
1. Auto-detects enabled features using environment variables and installed packages
2. Validates CUDA requirements against your current environment
3. Shows critical blockers with fix commands
4. Displays warnings for sub-optimal configurations
5. Provides detailed status for each feature

**Example output:**
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
   GPU memory (20.0 GB) is below recommended minimum (40.0 GB). Performance may be degraded or OOM errors may occur.
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

**Cause:** No config files, environment variables, or installed packages found.

**Solution:**
1. Set environment variables:
   ```bash
   export DATADESIGNER_INFERENCE_MODE=local
   ```
2. Or create a config file
3. Or install NeMo DataDesigner packages

---

### False positive detection

**Cause:** Package installed but not actively used.

**Solution:** The system will validate requirements and show "SKIPPED" for disabled features. No action needed.

---

### Blocker for local inference

**Cause:** Missing PyTorch or incompatible CUDA branch.

**Solution:** Follow the fix commands in the validation report:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### GPU memory warning

**Cause:** GPU has less than 40 GB for large models.

**Solution:**
1. Upgrade to larger GPU (A100 80GB)
2. Use model quantization
3. Use cloud-based inference instead

---

## Testing

Run the test suite:
```bash
pytest tests/nemo/test_datadesigner_detector.py -v
```

**Test coverage:**
- Feature definition validation
- Config file detection (JSON, invalid JSON, missing files)
- Environment variable detection
- Installed package detection
- Notebook cell analysis
- Requirement validation (PyTorch, CUDA, GPU memory)
- Validation report generation
- All 4 feature types

---

## API Reference

### `detect_enabled_features()`

```python
def detect_enabled_features(
    config_paths: Optional[List[Path]] = None,
    notebook_path: Optional[Path] = None,
    check_env_vars: bool = True,
    check_packages: bool = True,
) -> Dict[str, DataDesignerFeature]:
```

**Returns:** Dictionary mapping feature names to `DataDesignerFeature` objects.

---

### `validate_feature_requirements()`

```python
def validate_feature_requirements(
    feature: DataDesignerFeature,
    torch_version: Optional[str] = None,
    torch_cuda_branch: Optional[str] = None,
    cuda_available: bool = False,
    gpu_memory_gb: Optional[float] = None,
) -> DataDesignerFeature:
```

**Returns:** Updated `DataDesignerFeature` with validation status.

---

### `get_feature_validation_report()`

```python
def get_feature_validation_report(
    features: Dict[str, DataDesignerFeature],
    torch_version: Optional[str] = None,
    torch_cuda_branch: Optional[str] = None,
    cuda_available: bool = False,
    gpu_memory_gb: Optional[float] = None,
) -> Dict[str, Any]:
```

**Returns:** Comprehensive validation report with blockers, warnings, and environment info.

---

## See Also

- [Databricks Runtime Detection](./DATABRICKS_RUNTIME_DETECTION.md)
- [Driver Version Mapping](./DRIVER_VERSION_MAPPING.md)
- [CUDA Package Parser](./CUDA_PACKAGE_PARSER.md)
- [Enhanced Notebook](../notebooks/01_cuda_environment_validation_enhanced.py)

