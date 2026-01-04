# ⚡ Megatron-Core & PyTorch Lightning Installation Guide

## Overview

Comprehensive Databricks notebook cell for installing and validating Megatron-Core and PyTorch Lightning with critical compatibility checks for BioNeMo Framework.

**File:** `megatron_lightning_install_validate.py`

---

## 🚨 Critical Compatibility Issue

### **PyTorch Lightning >= 2.5.0 Breaks Megatron Callbacks**

**Problem:**
- PyTorch Lightning 2.5.0+ introduces breaking changes to callback system
- Megatron-Core callbacks fail with Lightning >= 2.5.0
- Training will crash with callback-related errors

**Solution:**
- **ALWAYS use:** `pytorch-lightning>=2.0.7,<2.5.0`
- This notebook enforces this constraint automatically

**References:**
- PyTorch Lightning: https://pypi.org/project/pytorch-lightning/
- NeMo Issues: https://github.com/NVIDIA/NeMo/issues/

---

## 📦 What Gets Installed

### **1. PyTorch Lightning (with version constraint)**
```python
%pip install 'pytorch-lightning>=2.0.7,<2.5.0'
```

**Why this version?**
- Minimum 2.0.7: Modern features and GPU strategy support
- Maximum < 2.5.0: Prevents Megatron callback breakage

### **2. NeMo Toolkit (>= 1.22.0)**
```python
%pip install 'nemo-toolkit[all]>=1.22.0'
```

**What it provides:**
- NeMo Framework for large language models
- **Megatron-Core** (bundled as dependency)
- Training utilities and callbacks
- Model parallelism strategies

### **3. Megatron-Core** (via NeMo)
- Not installed separately
- Comes bundled with NeMo Toolkit
- GitHub: https://github.com/NVIDIA/Megatron-LM

---

## 🧪 Validation Tests (8 Sections)

### **Section 1: PyTorch Lightning Validation**

| Test | What It Checks | Critical? |
|------|---------------|-----------|
| **1.1** | Import pytorch_lightning | ✅ Yes |
| **1.2** | Version < 2.5.0 check | ✅ **CRITICAL** |

**Critical Warning:**
If PyTorch Lightning >= 2.5.0 detected, displays large red warning with:
- Issue description
- Impact statement
- Downgrade command
- Reference links

---

### **Section 2: NeMo Toolkit Validation**

| Test | What It Checks | Critical? |
|------|---------------|-----------|
| **2.1** | Import nemo and get version | ✅ Yes |
| **2.2** | Version >= 1.22.0 check | ⚠️ Warn |
| **2.3** | Import nemo.core.ModelPT | ⚠️ Warn |

---

### **Section 3: Megatron-Core Validation**

| Test | What It Checks | Critical? |
|------|---------------|-----------|
| **3.1** | Import megatron.core.parallel_state | ⚠️ Warn |
| **3.2** | Get Megatron version (if available) | ℹ️ Info |

**Note:** Megatron absence is non-fatal if NeMo not installed.

---

### **Section 4: PyTorch Lightning GPU Strategy Testing**

| Test | What It Checks | Critical? |
|------|---------------|-----------|
| **4.1** | Trainer instantiation with GPU | ✅ Yes |
| **4.2** | GPU strategy auto-detection | ⚠️ Warn |
| **4.3** | Strategy class verification | ℹ️ Info |

**What it tests:**
```python
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    ...
)
```

---

### **Section 5: Distributed Environment Checks**

| Test | What It Checks | Result Key |
|------|---------------|------------|
| **5.1** | torch.cuda.nccl.is_available() | `nccl_available` |
| **5.2** | torch.distributed backends | Info only |
| **5.3** | FSDP support (PyTorch + Lightning) | `fsdp_available` |

---

### **Section 6: Compatibility Matrix**

Builds comprehensive compatibility report with:

```python
{
    "pytorch_lightning": {
        "version": "2.4.0",
        "safe_version": True,
        "constraint": ">=2.0.7,<2.5.0"
    },
    "nemo_toolkit": {
        "version": "1.23.0",
        "constraint": ">=1.22.0"
    },
    "megatron_core": {
        "available": True,
        "source": "bundled with NeMo Toolkit"
    },
    "distributed": {
        "nccl": True,
        "fsdp": True,
        "gpu_strategy": True
    }
}
```

---

### **Section 7: Critical Warnings Summary**

Aggregates all critical warnings found during validation:
- PyTorch Lightning version issues
- Missing dependencies
- Strategy initialization failures

Provides actionable fix commands for each issue.

---

### **Section 8: Final Summary**

- Overall compatibility status
- Component availability checklist
- Documentation links
- JSON results dictionary

---

## 📋 Results Dictionary

```python
{
    "pytorch_lightning_version": "2.4.0",
    "pytorch_lightning_safe": True,            # < 2.5.0
    "nemo_toolkit_version": "1.23.0",
    "megatron_available": True,
    "gpu_strategy_available": True,
    "nccl_available": True,
    "fsdp_available": True,
    "compatibility_matrix": {...},
    "critical_warnings": [],
    "installation_commands": [],
    "documentation_links": [
        "https://pypi.org/project/pytorch-lightning/",
        "https://pypi.org/project/nemo-toolkit/",
        "https://github.com/NVIDIA/Megatron-LM",
        "https://docs.nvidia.com/bionemo-framework/latest/"
    ]
}
```

---

## 📊 Example Output

### ✅ Successful Validation:

```
================================================================================
⚡ MEGATRON-CORE & PYTORCH LIGHTNING VALIDATION
================================================================================

================================================================================
SECTION 1: PYTORCH LIGHTNING VALIDATION
================================================================================

📦 Test 1.1: Import PyTorch Lightning and Check Version
────────────────────────────────────────────────────────────────────────────────
   ✅ PyTorch Lightning imported successfully
   ℹ️  Version: 2.4.0
   ✅ Version check: 2.4.0 < 2.5.0 (Megatron compatible)

================================================================================
SECTION 2: NEMO TOOLKIT VALIDATION
================================================================================

📦 Test 2.1: Import NeMo Toolkit and Check Version
────────────────────────────────────────────────────────────────────────────────
   ✅ NeMo Toolkit imported successfully
   ℹ️  Version: 1.23.0
   ✅ Version check: 1.23.0 >= 1.22.0 (BioNeMo compatible)

📦 Test 2.2: Import NeMo Core Modules
────────────────────────────────────────────────────────────────────────────────
   ✅ nemo.core.ModelPT imported successfully

================================================================================
SECTION 3: MEGATRON-CORE VALIDATION
================================================================================

📦 Test 3.1: Attempt Megatron-Core Import
────────────────────────────────────────────────────────────────────────────────
   ℹ️  Megatron-Core is provided by NeMo Toolkit as a dependency
   ✅ megatron.core.parallel_state imported successfully
   ℹ️  Megatron-Core version: Unknown (bundled with NeMo)

================================================================================
SECTION 4: PYTORCH LIGHTNING GPU STRATEGY TESTING
================================================================================

📦 Test 4.1: GPU Strategy Auto-Detection
────────────────────────────────────────────────────────────────────────────────
   ✅ Trainer instantiated with GPU accelerator
      Accelerator: CUDAAccelerator
      Strategy: SingleDeviceStrategy
      Devices: 1

================================================================================
SECTION 5: DISTRIBUTED ENVIRONMENT CHECKS
================================================================================

📦 Test 5.1: NCCL Availability
────────────────────────────────────────────────────────────────────────────────
   ✅ NCCL available
   ℹ️  NCCL version: (2, 18, 5)

📦 Test 5.2: torch.distributed Availability
────────────────────────────────────────────────────────────────────────────────
   ✅ torch.distributed is available
   ℹ️  Available backends: nccl, gloo

📦 Test 5.3: FSDP Strategy Support
────────────────────────────────────────────────────────────────────────────────
   ✅ FSDP (FullyShardedDataParallel) available
   ✅ ShardingStrategy available
   ℹ️  Available strategies: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
   ✅ PyTorch Lightning FSDPStrategy available

================================================================================
SECTION 6: COMPATIBILITY MATRIX
================================================================================

📊 Compatibility Report:
────────────────────────────────────────────────────────────────────────────────
✅ PyTorch Lightning: 2.4.0
   ✅ Version is Megatron compatible (< 2.5.0)

✅ NeMo Toolkit: 1.23.0

✅ Megatron-Core: Available
   ℹ️  Provided by NeMo Toolkit

📡 Distributed Training Support:
   ✅ NCCL: Available
   ✅ FSDP: Available
   ✅ GPU Strategy: Available

================================================================================
SECTION 7: CRITICAL WARNINGS & RECOMMENDATIONS
================================================================================

✅ No critical warnings - all compatibility checks passed!

📋 Known Compatibility Issues:
────────────────────────────────────────────────────────────────────────────────
1. PyTorch Lightning >= 2.5.0 breaks Megatron callbacks
   Status: ✅ Not present
   Solution: Use pytorch-lightning>=2.0.7,<2.5.0

================================================================================
FINAL SUMMARY
================================================================================

✅ ALL COMPATIBILITY CHECKS PASSED
   Environment is ready for BioNeMo training with Megatron-Core

📚 Documentation & Resources:
────────────────────────────────────────────────────────────────────────────────
1. https://pypi.org/project/pytorch-lightning/
2. https://pypi.org/project/nemo-toolkit/
3. https://github.com/NVIDIA/Megatron-LM
4. https://docs.nvidia.com/bionemo-framework/latest/

================================================================================
```

---

### ❌ With Critical Warning (Lightning >= 2.5.0):

```
📦 Test 1.1: Import PyTorch Lightning and Check Version
────────────────────────────────────────────────────────────────────────────────
   ✅ PyTorch Lightning imported successfully
   ℹ️  Version: 2.5.1

   🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
   ❌ CRITICAL WARNING: PyTorch Lightning 2.5.1 >= 2.5.0 - BREAKS MEGATRON CALLBACKS!
   🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
   Known Issue: Megatron callbacks fail with PyTorch Lightning >= 2.5.0
   Impact: Training will fail with callback errors
   
   💡 REQUIRED ACTION: Downgrade PyTorch Lightning
   Run in a new cell:
      %pip install 'pytorch-lightning>=2.0.7,<2.5.0'
      dbutils.library.restartPython()
   
   📚 References:
      - https://github.com/NVIDIA/NeMo/issues/
      - https://pypi.org/project/pytorch-lightning/
   🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨

... (rest of output)

================================================================================
SECTION 7: CRITICAL WARNINGS & RECOMMENDATIONS
================================================================================

🚨 CRITICAL WARNINGS (1):
────────────────────────────────────────────────────────────────────────────────

1. PyTorch Lightning 2.5.1 >= 2.5.0 - BREAKS MEGATRON CALLBACKS!

💡 REQUIRED ACTIONS:
────────────────────────────────────────────────────────────────────────────────
1. %pip install 'pytorch-lightning>=2.0.7,<2.5.0'

After running commands, execute:
   dbutils.library.restartPython()
```

---

## 🚀 Quick Start

### **Minimal Setup:**

```python
# Cell 1: Install
%pip install 'pytorch-lightning>=2.0.7,<2.5.0'
%pip install 'nemo-toolkit[all]>=1.22.0'

# Cell 2: Restart
dbutils.library.restartPython()

# Cell 3: Validate
import pytorch_lightning as pl
import nemo
from megatron.core import parallel_state

print(f"✅ PyTorch Lightning: {pl.__version__}")
print(f"✅ NeMo Toolkit: {nemo.__version__}")
print(f"✅ Megatron-Core: Available")
```

### **Full Validation:**

Import the entire `megatron_lightning_install_validate.py` notebook and run all cells.

---

## 🔧 Troubleshooting

### **Issue: PyTorch Lightning >= 2.5.0 installed**

**Solution:**
```python
%pip install --force-reinstall 'pytorch-lightning>=2.0.7,<2.5.0'
dbutils.library.restartPython()
```

### **Issue: NeMo Toolkit installation fails**

**Common causes:**
- Conflicting dependencies
- Insufficient memory during install

**Solution:**
```python
# Try with no-cache-dir
%pip install --no-cache-dir 'nemo-toolkit[all]>=1.22.0'
```

### **Issue: Megatron-Core not found**

**This means NeMo not installed correctly.**

**Solution:**
```python
# Reinstall NeMo with all dependencies
%pip uninstall -y nemo-toolkit
%pip install 'nemo-toolkit[all]>=1.22.0'
dbutils.library.restartPython()
```

### **Issue: GPU Strategy fails to initialize**

**Check:**
1. CUDA available: `torch.cuda.is_available()`
2. GPU runtime: Running on GPU-enabled cluster
3. Lightning version: Should be < 2.5.0

---

## 📚 Official Documentation

- **PyTorch Lightning:** https://pytorch-lightning.readthedocs.io/
- **NeMo Toolkit:** https://docs.nvidia.com/deeplearning/nemo/user-guide/
- **Megatron-LM:** https://github.com/NVIDIA/Megatron-LM
- **BioNeMo:** https://docs.nvidia.com/bionemo-framework/latest/

---

## ✅ Checklist

- [ ] Install PyTorch Lightning with constraint `>=2.0.7,<2.5.0`
- [ ] Install NeMo Toolkit >= 1.22.0
- [ ] Restart Python kernel
- [ ] Run validation cells
- [ ] Verify `pytorch_lightning_safe = True`
- [ ] Verify `megatron_available = True`
- [ ] Check for critical warnings
- [ ] Resolve any issues before training

---

## 🎯 Key Takeaways

1. **ALWAYS use PyTorch Lightning < 2.5.0 for Megatron**
2. **NeMo Toolkit bundles Megatron-Core** (don't install separately)
3. **Restart Python after pip install** (critical in Databricks)
4. **Check critical warnings** before starting training
5. **NCCL + FSDP required** for distributed training

---

**Created:** 2026-01-04  
**Compatible with:** Databricks Runtime 14.3+, ML Runtime with GPU  
**Tested on:** Python 3.10+, PyTorch 2.2+, CUDA 12.0+

