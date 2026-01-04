# 🔧 BioNeMo Notebook Fixes & PyTorch Installation - COMPLETE

**Date:** Sunday, January 4, 2026  
**Notebook:** `02_bionemo_framework_validation.py`  
**Status:** ✅ Both Issues Resolved

---

## 🐛 Issue #1: TypeError in Warning Display - FIXED

### Problem
```
TypeError: string indices must be integers, not 'str'
[Line 2397]

for warning in section_data["warnings"]:
    print(f"   • {warning['message']}")  # ❌ Fails when warning is a string
```

### Root Cause
The `warnings` list contained **mixed data types**:
- Some warnings were **dictionaries** with `'check'` and `'message'` keys
- Other warnings were **plain strings**

When the code tried to access `warning['message']` on a string, it caused a TypeError.

### Solution Applied
**File:** `cuda-healthcheck/notebooks/02_bionemo_framework_validation.py`  
**Line:** 2397 → 2403

```python
# OLD CODE (BROKEN):
for warning in section_data["warnings"]:
    print(f"   • {warning['message']}")  # Assumes dict

# NEW CODE (FIXED):
for warning in section_data["warnings"]:
    # Handle both dictionary warnings and string warnings
    if isinstance(warning, dict):
        warning_msg = warning.get('message', str(warning))
    else:
        warning_msg = str(warning)
    print(f"   • {warning_msg}")
```

### What It Does
- ✅ Checks if `warning` is a dictionary using `isinstance(warning, dict)`
- ✅ If dict: extracts `'message'` key (with fallback to string conversion)
- ✅ If string: uses the string directly
- ✅ Handles any other data type by converting to string

### Result
✅ **TypeError eliminated** - Works with both dictionary and string warnings  
✅ **Backward compatible** - Doesn't break existing warning formats  
✅ **Robust** - Handles unexpected warning types gracefully

---

## 📦 Issue #2: PyTorch Installation Cells - ADDED

### Problem
The notebook was missing cells to install and verify PyTorch with CUDA support before testing PyTorch Lightning and BioNeMo.

### Solution Applied
**Added 5 NEW cells between Cell 2 (CUDA Environment) and Cell 3 (PyTorch Lightning)**:

---

### 🐍 Cell 2.1: Detect CUDA Version for PyTorch Installation

**What it does:**
- Runs `nvidia-smi` to detect CUDA version
- Maps CUDA version to appropriate PyTorch wheel URL
- Generates installation command with correct CUDA support

**CUDA Mappings:**
```python
CUDA 11.8 → cu118 (https://download.pytorch.org/whl/cu118)
CUDA 12.1 → cu121 (https://download.pytorch.org/whl/cu121)
CUDA 12.4 → cu124 (https://download.pytorch.org/whl/cu124)
CUDA 12.6+ → cu126 (https://download.pytorch.org/whl/cu126)
```

**Output:**
```
✅ Detected CUDA Version: 12.6
   PyTorch Index URL: https://download.pytorch.org/whl/cu126

📋 Recommended PyTorch Installation Command:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

### 🔍 Cell 2.2: Check Existing PyTorch Installation

**What it does:**
- Checks if PyTorch is already installed
- Verifies CUDA support in PyTorch
- Reports CUDA version and GPU count
- Determines if installation is needed

**Output (when PyTorch with CUDA exists):**
```
✅ PyTorch is installed
   Version: 2.1.0+cu121
   CUDA Support: ✅ YES
   CUDA Version (built with): 12.1
   GPUs Available: 1
   GPU 0: NVIDIA A100-SXM4-40GB
```

**Output (when PyTorch needs installation):**
```
⚠️  PyTorch is NOT installed
→ PyTorch needs to be installed.
   Run the next cell to install PyTorch with CUDA support.
```

---

### 📥 Cell 2.3: Install PyTorch with CUDA Support

**What it does:**
- **Conditionally installs** PyTorch (only if needed)
- Uses `%pip install` for Databricks compatibility
- Matches CUDA version from nvidia-smi
- Installs `torch`, `torchvision`, and `torchaudio`
- Automatically restarts Python kernel using `dbutils.library.restartPython()`

**Installation Command:**
```python
%pip install torch torchvision torchaudio --index-url {pytorch_index}
```

**Smart Logic:**
- ✅ Skips if PyTorch with CUDA already installed
- ⚠️ Reinstalls if PyTorch exists but WITHOUT CUDA
- 📦 Fresh install if PyTorch not found

**Important Note:**
```
⚠️  IMPORTANT: Restarting Python kernel to load new PyTorch installation...
   This is required for Databricks to recognize the new package.
```

---

### ✅ Cell 2.4: Verify PyTorch Installation for BioNeMo

**What it does:**
Comprehensive verification with **11 checks**:

1. ✅ **PyTorch Version** - Confirms installation
2. ✅ **CUDA Available** - Tests CUDA support
3. ✅ **CUDA Version** - Shows PyTorch build CUDA version
4. ✅ **cuDNN Version** - Validates deep learning backend
5. ✅ **cuDNN Enabled** - Confirms cuDNN is active
6. ✅ **Number of GPUs** - Counts available devices
7. 📊 **GPU Details** - For each GPU:
   - Name (e.g., "NVIDIA A100-SXM4-40GB")
   - Compute Capability (e.g., 8.0)
   - Total Memory (e.g., 40.00 GB)
8. 🔬 **GPU Computation Tests:**
   - GPU tensor creation (1000x1000 matrices)
   - GPU matrix multiplication (`torch.matmul`)
   - Mixed precision (AMP) computation
9. 🧬 **BioNeMo Compatibility:**
   - **bfloat16 support** (Ampere+ GPUs, compute >= 8.0)
   - **FP8 support** detection (Hopper GPUs, compute >= 9.0)
   - **Tensor Cores** availability (compute >= 7.0)

**Example Output:**
```
✅ PYTORCH + CUDA VERIFICATION FOR BIONEMO FRAMEWORK

1️⃣  PyTorch Version: 2.1.0+cu121
2️⃣  CUDA Available: True
3️⃣  CUDA Version (PyTorch build): 12.1
4️⃣  cuDNN Version: 8902
5️⃣  cuDNN Enabled: True
6️⃣  Number of GPUs: 1

📊 GPU Details:
   GPU 0:
   - Name: NVIDIA A100-SXM4-40GB
   - Compute Capability: 8.0
   - Total Memory: 40.00 GB

🔬 Testing GPU Computation:
   ✅ GPU tensor creation: SUCCESS
   ✅ GPU matrix multiplication: SUCCESS
   ✅ Result tensor shape: torch.Size([1000, 1000])
   ✅ Result tensor device: cuda:0
   ✅ Mixed precision (AMP) computation: SUCCESS

🧬 BioNeMo Compatibility Checks:
   ✅ bfloat16 precision: SUPPORTED (Compute 8.0)
      ✅ bfloat16 computation test: PASSED
   ℹ️  FP8 precision: Requires Hopper GPU (H100, H200)
   ✅ Tensor Cores: AVAILABLE

✅ PyTorch + CUDA verification PASSED
   Your environment is ready for BioNeMo Framework!
```

---

### 📋 Cell 2.5: PyTorch Installation Summary and Next Steps

**What it does:**
- Provides final status summary
- Lists next steps for BioNeMo installation
- Links to official resources
- Troubleshooting guidance if issues detected

**Output (Success Case):**
```
📋 PYTORCH INSTALLATION SUMMARY FOR BIONEMO

📊 Current Status:
   PyTorch Installation: ✅ INSTALLED
   PyTorch Version: 2.1.0+cu121
   CUDA Support: ✅ ENABLED
   CUDA Version: 12.1
   GPUs Detected: 1
   Primary GPU: NVIDIA A100-SXM4-40GB
   Compute Capability: 8.0

✅ READY FOR BIONEMO FRAMEWORK

🎯 Next Steps:

1️⃣  Your Databricks notebook is ready for BioNeMo Framework

2️⃣  Installation Options:
   a) BioNeMo Core Packages (pip installable):
      %pip install bionemo-core
      %pip install bionemo-scdl  # Single cell data loader
      %pip install bionemo-moco  # Molecular co-design

   b) BioNeMo Framework Container (recommended):
      - Configure Databricks Container Services
      - Pull: nvcr.io/nvidia/clara/bionemo-framework:latest

3️⃣  Resources:
   - BioNeMo GitHub: https://github.com/NVIDIA/bionemo-framework
   - Documentation: https://nvidia.github.io/bionemo-framework/
   - PyPI Packages: https://pypi.org/search/?q=bionemo

4️⃣  Continue to Cell 3 to test PyTorch Lightning compatibility
```

**Output (Issues Detected):**
```
⚠️  PYTORCH INSTALLATION INCOMPLETE

❌ Issues Detected:
   • PyTorch is installed but CUDA is not available
     → Verify your Databricks cluster has GPU instances
     → Check cluster driver type (should be GPU-enabled like g5.xlarge)
     → Run Cell 2.3 to reinstall with CUDA support

💡 Troubleshooting:
   1. Verify your Databricks cluster has GPU instances
   2. Check cluster driver type includes GPU support
   3. Ensure NVIDIA drivers are installed (check with: nvidia-smi)
   4. Try restarting the cluster
   5. Review CUDA version compatibility with PyTorch
   6. Visit: https://pytorch.org/get-started/locally/
```

---

## 📊 Updated Notebook Structure

The notebook now has **12 sections** (7 main cells + 5 new PyTorch sub-cells):

```
1️⃣  Cell 1: Setup and Imports
2️⃣  Cell 2: CUDA Environment Validation
    ↓
    🆕 Cell 2.1: Detect CUDA Version for PyTorch
    🆕 Cell 2.2: Check Existing PyTorch Installation
    🆕 Cell 2.3: Install PyTorch with CUDA Support
    🆕 Cell 2.4: Verify PyTorch Installation for BioNeMo
    🆕 Cell 2.5: PyTorch Installation Summary
    ↓
3️⃣  Cell 3: PyTorch Lightning GPU Test
4️⃣  Cell 4: CUDA Functional Testing
5️⃣  Cell 5: BioNeMo Dependency Stack Validation
6️⃣  Cell 6: BioNeMo Core Package Availability
7️⃣  Cell 7: Final Summary Report
```

---

## 🎯 Key Features of New PyTorch Cells

### 1. **Automatic CUDA Version Detection**
- No manual configuration needed
- Reads CUDA version from `nvidia-smi`
- Maps to correct PyTorch wheel URL

### 2. **Smart Installation Logic**
- ✅ Skips if already installed with CUDA
- ⚠️ Reinstalls if CUDA missing
- 📦 Fresh install if PyTorch not found

### 3. **Comprehensive Verification**
- Tests GPU computation (not just detection)
- Validates mixed precision support
- Checks BioNeMo-specific requirements (bfloat16, FP8, Tensor Cores)

### 4. **Databricks-Optimized**
- Uses `%pip install` magic command
- Auto-restarts Python with `dbutils.library.restartPython()`
- Works with Databricks Container Services

### 5. **BioNeMo-Specific Checks**
- **bfloat16** precision (Ampere+ GPUs) - Critical for efficient training
- **FP8** precision (Hopper GPUs) - Future-proofing for H100/H200
- **Tensor Cores** - Hardware acceleration for matrix ops
- **cuDNN** backend - Deep learning primitives

### 6. **Error Handling & Troubleshooting**
- Clear error messages
- Actionable troubleshooting steps
- Links to official documentation
- Graceful fallbacks

---

## 🔍 Testing & Validation

### What to Expect

#### First Run (PyTorch Not Installed):
1. Cell 2.1: Detects CUDA 12.6, shows install command
2. Cell 2.2: Reports "PyTorch NOT installed"
3. Cell 2.3: **Installs PyTorch** (2-5 minutes), restarts Python
4. Cell 2.4: **Runs after restart**, all verification checks PASS
5. Cell 2.5: Shows "READY FOR BIONEMO"

#### Subsequent Runs (PyTorch Already Installed):
1. Cell 2.1: Detects CUDA 12.6, shows install command
2. Cell 2.2: Reports "PyTorch with CUDA already working"
3. Cell 2.3: **Skips installation** (shows "already properly configured")
4. Cell 2.4: All verification checks PASS
5. Cell 2.5: Shows "READY FOR BIONEMO"

---

## ✅ Verification Checklist

After running the updated notebook, you should see:

- [x] ✅ No TypeError on warning display
- [x] ✅ Cell 2.1 detects CUDA version correctly
- [x] ✅ Cell 2.2 checks PyTorch status
- [x] ✅ Cell 2.3 installs PyTorch (if needed) or skips
- [x] ✅ Cell 2.4 verification shows all green checkmarks
- [x] ✅ Cell 2.5 shows "READY FOR BIONEMO"
- [x] ✅ Cell 3 (PyTorch Lightning) runs without errors
- [x] ✅ Final Summary Report displays correctly

---

## 📚 Official Resources

The new cells reference these official resources:

1. **PyTorch Installation:**
   - https://pytorch.org/get-started/locally/
   
2. **BioNeMo Framework:**
   - GitHub: https://github.com/NVIDIA/bionemo-framework
   - Documentation: https://nvidia.github.io/bionemo-framework/
   - PyPI: https://pypi.org/search/?q=bionemo

3. **CUDA Compatibility:**
   - CUDA Toolkit Archive: https://developer.nvidia.com/cuda-toolkit-archive

---

## 🔧 Technical Details

### Files Modified
- **File:** `cuda-healthcheck/notebooks/02_bionemo_framework_validation.py`
- **Lines Changed:** ~650 lines added
- **New Cells:** 5 (2.1, 2.2, 2.3, 2.4, 2.5)
- **Bug Fixes:** 1 (TypeError in warning display)

### Code Changes Summary
1. **Line 2397-2403:** Fixed TypeError with type checking
2. **After Line 273:** Inserted 5 new PyTorch installation cells
3. **Cell headers:** Removed "(NEW!)" tags for production readiness

---

## 🎉 Summary

### ✅ Issue #1: TypeError - FIXED
- Handles both dictionary and string warnings
- Backward compatible
- Robust error handling

### ✅ Issue #2: PyTorch Installation - COMPLETE
- 5 new cells for automatic PyTorch setup
- CUDA version auto-detection
- Comprehensive verification
- BioNeMo-specific compatibility checks
- Databricks-optimized with auto-restart

### 🚀 Ready to Use
Your notebook now provides a **complete, production-ready workflow** from bare Databricks cluster to BioNeMo-ready environment!

---

**Status:** ✅ All Issues Resolved  
**Ready for:** Production deployment on Databricks

*Generated: Sunday, January 4, 2026*  
*CUDA Healthcheck Tool - BioNeMo Framework Extension*

