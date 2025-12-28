# CUDA Breaking Changes Reference

Quick reference guide for CUDA version incompatibilities detected by this tool.

## PyTorch

### CUDA 12.x → 13.0
- **Severity**: 🚨 CRITICAL
- **Issue**: PyTorch built for CUDA 12.x will not work with CUDA 13.x
- **Fix**: Install PyTorch built for CUDA 13.x
- **Command**: `pip install torch --index-url https://download.pytorch.org/whl/cu130`

### CUDA 12.4 → 12.6
- **Severity**: ⚠️ WARNING
- **Issue**: May work but not guaranteed, possible performance issues
- **Fix**: Test thoroughly or rebuild for CUDA 12.6

## TensorFlow

### CUDA 13.x Support
- **Severity**: 🚨 CRITICAL
- **Issue**: TensorFlow < 2.18 does not support CUDA 13.x
- **Fix**: Upgrade to TensorFlow 2.18+
- **Command**: `pip install tensorflow[and-cuda]>=2.18.0`

### Compute Capability 9.0 (H100/H200)
- **Severity**: 🚨 CRITICAL
- **Issue**: TensorFlow < 2.16 doesn't support Hopper GPUs
- **Fix**: Upgrade to TensorFlow 2.16+
- **Applies to**: H100, H200, future Hopper+ GPUs

## cuDF/RAPIDS

### CUDA 13.x Support
- **Severity**: 🚨 CRITICAL
- **Issue**: RAPIDS < 24.12 doesn't support CUDA 13.x
- **Fix**: Upgrade to RAPIDS 24.12+
- **Command**: `pip install cudf-cu13==24.12.*`

### Package Naming
- **Severity**: 🚨 CRITICAL
- **Issue**: Must use correct package for CUDA version
- **CUDA 12.x**: `pip install cudf-cu12`
- **CUDA 13.x**: `pip install cudf-cu13`

## NVIDIA Containers

### Isaac Sim
- **Required CUDA**: 12.2+
- **Required Driver**: 535.104.05+
- **Container**: `nvcr.io/nvidia/isaac-sim:2024.1.0`

### BioNeMo
- **Required CUDA**: 12.4+
- **Required Driver**: 550.54.15+
- **Container**: `nvcr.io/nvidia/clara/bionemo-framework:2.0`

### Modulus (Physics ML)
- **Required CUDA**: 12.1+ (recommended)
- **Container**: `nvcr.io/nvidia/modulus/modulus:24.01`

## CUDA Runtime

### Compute Capability 5.0 Deprecation (CUDA 13.x)
- **Severity**: ⚠️ WARNING
- **Issue**: Maxwell GPUs (GTX 900 series, Quadro M series) deprecated
- **Affected**: Compute capability 5.0
- **Fix**: Upgrade GPU hardware or stay on CUDA 12.x
- **Check**: `nvidia-smi --query-gpu=compute_cap --format=csv`

### cuDNN 9.x API Changes
- **Severity**: ⚠️ WARNING
- **Issue**: cuDNN 9.x (with CUDA 13.0) has API changes
- **Affects**: Custom CUDA kernels
- **Fix**: Review migration guide and update custom kernels

## Quick Commands

### Check Current Environment
```bash
# CUDA version
nvidia-smi

# PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# TensorFlow
python -c "import tensorflow as tf; print(f'TF: {tf.__version__}, GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"

# cuDF
python -c "import cudf; print(f'cuDF: {cudf.__version__}')"

# GPU compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### Run Healthcheck
```bash
# Complete analysis
python main.py healthcheck

# Just detection
python main.py detect

# Scan Databricks clusters
python main.py scan

# View breaking changes
python main.py breaking-changes --library pytorch
```

## Compatibility Matrix

| Library | CUDA 12.4 | CUDA 12.6 | CUDA 13.0 |
|---------|-----------|-----------|-----------|
| PyTorch 2.1.x | ✅ Native | ⚠️ Compatible | ❌ Incompatible |
| PyTorch 2.2+ (cu130) | ⚠️ Compatible | ⚠️ Compatible | ✅ Native |
| TensorFlow 2.15 | ✅ Compatible | ✅ Compatible | ❌ Incompatible |
| TensorFlow 2.18+ | ✅ Compatible | ✅ Compatible | ✅ Native |
| RAPIDS 23.x | ✅ Native | ⚠️ Compatible | ❌ Incompatible |
| RAPIDS 24.12+ | ✅ Compatible | ✅ Compatible | ✅ Native |

Legend:
- ✅ Native: Officially supported, built for this CUDA version
- ⚠️ Compatible: May work but not officially tested
- ❌ Incompatible: Will not work

## For More Information

- Full migration guide: `docs/MIGRATION_GUIDE.md`
- Setup instructions: `docs/SETUP.md`
- Contributing: `CONTRIBUTING.md`








