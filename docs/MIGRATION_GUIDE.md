# CUDA Migration Guide

## Overview

This guide helps you navigate CUDA version transitions on Databricks clusters, focusing on the most common scenarios and breaking changes.

## CUDA 12.4 → CUDA 12.6

### Compatibility Level: **High** ✅

CUDA 12.4 and 12.6 are minor version updates with good backward compatibility.

### What Works
- Most PyTorch 2.x binaries built for CUDA 12.4 will work with 12.6
- TensorFlow 2.13+ should work without issues
- cuDF/RAPIDS libraries are generally compatible

### What to Watch
- **Performance**: Minor version differences may affect kernel performance
- **Custom CUDA Code**: If you have custom kernels, test thoroughly
- **Driver Version**: Ensure driver is at least 535.x for CUDA 12.6

### Recommended Actions
1. ✅ Test your code in a dev cluster with CUDA 12.6
2. ✅ Monitor for CUDA-related warnings in logs
3. ⚠️ Consider rebuilding PyTorch if you see performance degradation
4. ✅ Update to latest patch versions of libraries

## CUDA 12.x → CUDA 13.0

### Compatibility Level: **Low** ⚠️

CUDA 13.0 introduces major changes requiring library updates.

### Breaking Changes

#### 1. PyTorch
**Impact**: 🚨 **CRITICAL**

- PyTorch binaries for CUDA 12.x **will not work** with CUDA 13.0
- Must use PyTorch specifically built for CUDA 13.x

**Migration Steps**:
```bash
# Uninstall old PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA 13.x version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**Timeline**: Wait for official PyTorch CUDA 13.x builds (check pytorch.org)

#### 2. TensorFlow
**Impact**: 🚨 **CRITICAL**

- TensorFlow < 2.18 does not support CUDA 13.x
- Requires TensorFlow 2.18 or later

**Migration Steps**:
```bash
# Upgrade TensorFlow
pip install --upgrade tensorflow[and-cuda]>=2.18.0

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### 3. cuDF/RAPIDS
**Impact**: 🚨 **CRITICAL**

- RAPIDS < 24.12 does not support CUDA 13.x
- Package name must match: `cudf-cu13` not `cudf-cu12`

**Migration Steps**:
```bash
# For CUDA 13.x
pip uninstall cudf-cu12  # Remove old version
pip install cudf-cu13==24.12.*

# Verify
python -c "import cudf; print(cudf.__version__)"
```

#### 4. Compute Capability Changes
**Impact**: ⚠️ **WARNING**

- CUDA 13.x deprecates compute capability 5.0 (Maxwell GPUs)
- GTX 900 series and Quadro M series may not work correctly

**Check Your GPU**:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

If compute capability is 5.0, stay on CUDA 12.x or upgrade hardware.

### Full Migration Checklist

#### Pre-Migration
- [ ] Identify all GPU-enabled Databricks clusters
- [ ] Run CUDA healthcheck on all clusters: `python main.py scan`
- [ ] Document current library versions
- [ ] Create backup/snapshot of working environment
- [ ] Check for custom CUDA kernels in codebase

#### Migration Phase
- [ ] Update Databricks runtime to one with CUDA 13.x
- [ ] Upgrade PyTorch to CUDA 13.x build
- [ ] Upgrade TensorFlow to 2.18+
- [ ] Upgrade RAPIDS to 24.12+
- [ ] Update any NVIDIA containers (Isaac Sim, BioNeMo, etc.)
- [ ] Rebuild custom Docker images with CUDA 13.x base

#### Testing Phase
- [ ] Run full test suite on dev cluster
- [ ] Verify GPU detection: `nvidia-smi`
- [ ] Check library imports work
- [ ] Run training jobs end-to-end
- [ ] Monitor GPU memory usage
- [ ] Check inference latency
- [ ] Validate model outputs match expected results

#### Post-Migration
- [ ] Run healthcheck again: `python main.py healthcheck`
- [ ] Monitor production clusters for CUDA errors
- [ ] Document any issues encountered
- [ ] Update team documentation
- [ ] Schedule review after 1 week

## Library-Specific Guides

### PyTorch Migration

**Check Current Version**:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Available: {torch.cuda.is_available()}")
```

**Common Issues**:
- `RuntimeError: No CUDA GPUs are available` → Wrong CUDA version
- `ImportError: cannot import name '_C'` → Mismatched CUDA libraries
- Performance degradation → Rebuild for target CUDA version

### TensorFlow Migration

**Check Current Version**:
```python
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
print(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
```

**Common Issues**:
- `Could not load dynamic library 'libcudart.so.13.0'` → Update TensorFlow
- No GPUs detected → Check driver and TensorFlow version compatibility

### cuDF/RAPIDS Migration

**Check Current Version**:
```python
import cudf
print(f"cuDF: {cudf.__version__}")
```

**Common Issues**:
- `ImportError: libcuda.so.1: cannot open shared object file` → Wrong cuDF package
- Version conflicts → Ensure all RAPIDS libraries match (cuDF, cuML, cuGraph)

## NVIDIA Container Migrations

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

## Emergency Rollback

If migration fails, rollback steps:

1. **Revert Databricks Runtime**:
   - Use previous runtime version in cluster configuration
   - Restart cluster

2. **Reinstall Previous Libraries**:
```bash
# Rollback PyTorch
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Rollback TensorFlow
pip install tensorflow==2.15.0

# Rollback RAPIDS
pip install cudf-cu12==23.12.*
```

3. **Verify Rollback**:
```bash
python main.py healthcheck
```

## Databricks-Specific Considerations

### Runtime Selection
- Check Databricks Runtime release notes for CUDA version
- GPU-enabled runtimes: ML Runtime for ML workflows
- Use LTS (Long Term Support) runtimes when possible

### Cluster Configuration
```python
# Example Databricks cluster config with CUDA 13.0
{
    "spark_version": "14.0.x-gpu-ml-scala2.12",
    "node_type_id": "g5.xlarge",  # AWS example
    "driver_node_type_id": "g5.xlarge",
    "num_workers": 2,
    "spark_conf": {
        "spark.databricks.delta.preview.enabled": "true"
    }
}
```

### Init Scripts
Create init script to verify CUDA on cluster start:

```bash
#!/bin/bash
# /dbfs/init-scripts/cuda-check.sh

echo "Checking CUDA installation..."
nvidia-smi
nvcc --version

echo "Testing PyTorch CUDA..."
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"

echo "Testing TensorFlow CUDA..."
python -c "import tensorflow as tf; print(f'TF GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

## Support Resources

- **NVIDIA CUDA Documentation**: https://docs.nvidia.com/cuda/
- **Databricks GPU Support**: https://docs.databricks.com/en/compute/gpu.html
- **PyTorch CUDA Guide**: https://pytorch.org/get-started/locally/
- **TensorFlow GPU Guide**: https://www.tensorflow.org/install/gpu
- **RAPIDS Documentation**: https://docs.rapids.ai/

## Getting Help

Run the healthcheck tool for personalized guidance:

```bash
python main.py healthcheck
```

This will analyze your specific environment and provide tailored recommendations.








