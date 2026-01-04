# 🧬 BioNeMo Core Installation & Validation for Databricks

## Overview

Two versions of the BioNeMo Core installation and validation notebook cell are provided:

1. **`bionemo_core_install_databricks.py`** - Multi-cell Databricks notebook (RECOMMENDED)
2. **`bionemo_core_install_validate.py`** - Single-cell standalone version

---

## 📄 Version 1: Multi-Cell Databricks Notebook (RECOMMENDED)

**File:** `bionemo_core_install_databricks.py`

### Features:
- ✅ Uses Databricks `%pip` magic commands
- ✅ Includes `dbutils.library.restartPython()` for proper package loading
- ✅ Split into logical cells for better Databricks workflow
- ✅ Clear markdown documentation between cells
- ✅ Optimized for Databricks environment

### Cell Structure:

```
Cell 1: Check Existing Installation
Cell 2: Install BioNeMo Core (%pip install)
Cell 3: Restart Python Kernel (dbutils.library.restartPython())
Cell 4: Validate Installation & Test Functionality
```

### How to Use:

1. **Import to Databricks:**
   ```
   Workspace → Create → Notebook
   Copy and paste the contents of bionemo_core_install_databricks.py
   ```

2. **Run Cells Sequentially:**
   - Cell 1: Check if already installed
   - Cell 2: Install with %pip
   - Cell 3: Restart Python (required!)
   - Cell 4: Validate and test

3. **Result:**
   - Returns dictionary with installation status
   - Tests get_autocast_dtype functionality
   - Checks optional model packages

---

## 📄 Version 2: Single-Cell Standalone

**File:** `bionemo_core_install_validate.py`

### Features:
- ✅ Complete validation in one cell
- ✅ Auto-installation with graceful fallback
- ✅ Comprehensive error handling
- ✅ Works outside Databricks (with modifications)
- ✅ Detailed status reporting

### Sections:

```
Section 1: Installation (with auto-install)
Section 2: Import Validation
Section 3: Optional Packages (non-fatal)
Section 4: Final Summary
```

### How to Use:

1. **Copy entire script to one Databricks cell**

2. **Uncomment %pip line:**
   ```python
   # Change this line (around line 71):
   # %pip install "bionemo-core>=0.2.0"
   
   # To:
   %pip install "bionemo-core>=0.2.0"
   ```

3. **Uncomment restart line:**
   ```python
   # At the end, uncomment:
   if bionemo_install_results["bionemo_core_installed"]:
       dbutils.library.restartPython()
   ```

4. **Run cell and wait for restart**

---

## 📋 Results Dictionary Structure

Both versions return the same results structure:

```python
{
    "bionemo_core_installed": bool,
    "bionemo_version": "0.2.0",
    "autocast_functional": bool,
    "optional_packages_available": {
        "bionemo_llm": bool,
        "bionemo_esm2": bool,
        "bionemo_evo2": bool
    },
    "documentation_links": [
        "https://pypi.org/project/bionemo-core/",
        "https://docs.nvidia.com/bionemo-framework/latest/",
        "https://github.com/NVIDIA/bionemo-framework"
    ]
}
```

---

## 🧪 What Gets Tested

### ✅ Core Functionality Tests:

| Test | What It Validates | Pass Criteria |
|------|-------------------|---------------|
| **Import Test** | `from bionemo.core import __version__` | Import succeeds |
| **Dtype Utils** | `from bionemo.core.utils.dtype import get_autocast_dtype` | Import succeeds |
| **BFloat16 Test** | `get_autocast_dtype('bfloat16')` | Returns `torch.bfloat16` |
| **Float16 Test** | `get_autocast_dtype('float16')` | Returns `torch.float16` |

### ⚠️ Optional Package Tests (Non-Fatal):

| Package | Description | Required? |
|---------|-------------|-----------|
| `bionemo_llm` | BioNeMo LLM models (BioBert) | ❌ No |
| `bionemo_esm2` | ESM2 protein language model | ❌ No |
| `bionemo_evo2` | Evo2 genomics foundation model | ❌ No |

---

## 📊 Example Output

### Successful Installation:

```
================================================================================
🧬 BIONEMO CORE VALIDATION
================================================================================

📦 Test 1: Import bionemo.core
────────────────────────────────────────────────────────────────────────────────
   ✅ Successfully imported bionemo.core
   ℹ️  Version: 0.2.0

📦 Test 2: Import bionemo.core.utils.dtype.get_autocast_dtype
────────────────────────────────────────────────────────────────────────────────
   ✅ Successfully imported get_autocast_dtype

📦 Test 3: Test get_autocast_dtype('bfloat16')
────────────────────────────────────────────────────────────────────────────────
   ✅ get_autocast_dtype('bfloat16') = torch.bfloat16
   ✅ Correct: dtype matches torch.bfloat16
   ✅ get_autocast_dtype('float16') = torch.float16
   ✅ Correct: dtype matches torch.float16

📦 Test 4: Optional Model Packages (Non-Fatal)
────────────────────────────────────────────────────────────────────────────────
   ℹ️  bionemo_llm: Not installed (optional)
   ℹ️  bionemo_esm2: Not installed (optional)
   ℹ️  bionemo_evo2: Not installed (optional)

================================================================================
📊 VALIDATION SUMMARY
================================================================================

BioNeMo Core:
   Installed: ✅ Yes
   Version: 0.2.0
   Autocast: ✅ Functional

Optional Models: 0/3 available

📚 Documentation:
   • https://pypi.org/project/bionemo-core/
   • https://docs.nvidia.com/bionemo-framework/latest/
   • https://github.com/NVIDIA/bionemo-framework
================================================================================
```

---

## 🔧 Installation Commands Reference

### Install BioNeMo Core:
```python
%pip install "bionemo-core>=0.2.0"
```

### Install Optional Models:
```python
# BioNeMo LLM (BioBert)
%pip install bionemo-llm

# ESM2 Protein Model
%pip install bionemo-esm2

# Evo2 Genomics Model
%pip install bionemo-evo2
```

### Restart Python (Required after installation):
```python
dbutils.library.restartPython()
```

---

## 🚨 Troubleshooting

### Issue: Import fails after installation

**Solution:**
```python
# Always restart Python after installing packages in Databricks
dbutils.library.restartPython()
```

### Issue: get_autocast_dtype not found

**Problem:** BioNeMo Core version < 0.2.0

**Solution:**
```python
%pip install --upgrade "bionemo-core>=0.2.0"
dbutils.library.restartPython()
```

### Issue: Optional packages not available

**This is normal!** Optional packages are not installed by default.

**To install:**
```python
%pip install bionemo-llm bionemo-esm2 bionemo-evo2
dbutils.library.restartPython()
```

---

## 📚 Official Resources

### PyPI:
- **bionemo-core:** https://pypi.org/project/bionemo-core/
- **bionemo-llm:** https://pypi.org/project/bionemo-llm/
- **bionemo-esm2:** https://pypi.org/project/bionemo-esm2/
- **bionemo-evo2:** https://pypi.org/project/bionemo-evo2/

### Documentation:
- **Official Docs:** https://docs.nvidia.com/bionemo-framework/latest/
- **GitHub:** https://github.com/NVIDIA/bionemo-framework
- **Installation Guide:** https://docs.nvidia.com/bionemo-framework/latest/user-guide/

---

## ✅ Checklist for Databricks

- [ ] Import notebook to Databricks workspace
- [ ] Run Cell 1: Check existing installation
- [ ] Run Cell 2: Install BioNeMo Core
- [ ] Run Cell 3: Restart Python kernel (CRITICAL!)
- [ ] Run Cell 4: Validate installation
- [ ] Verify: `bionemo_core_installed = True`
- [ ] Verify: `autocast_functional = True`
- [ ] Review optional package status
- [ ] Install optional packages if needed
- [ ] Restart Python again if installed optional packages

---

## 🎯 Quick Start Commands

### Minimal Setup (Core Only):
```python
# Cell 1
%pip install "bionemo-core>=0.2.0"

# Cell 2
dbutils.library.restartPython()

# Cell 3
from bionemo.core import __version__
from bionemo.core.utils.dtype import get_autocast_dtype
print(f"BioNeMo Core v{__version__} installed!")
print(f"get_autocast_dtype('bfloat16') = {get_autocast_dtype('bfloat16')}")
```

### Full Setup (Core + Models):
```python
# Cell 1
%pip install "bionemo-core>=0.2.0" bionemo-llm bionemo-esm2 bionemo-evo2

# Cell 2
dbutils.library.restartPython()

# Cell 3
from bionemo.core import __version__
import bionemo_llm
import bionemo_esm2
import bionemo_evo2
print("✅ All BioNeMo packages installed!")
```

---

## 📝 Notes

1. **Python Restart is Required:** Always run `dbutils.library.restartPython()` after pip install
2. **Version Pinning:** Use `>=0.2.0` to ensure latest compatible version
3. **Optional Packages:** Not required for core functionality
4. **Error Handling:** Both versions handle errors gracefully - won't crash notebook
5. **Results Dictionary:** Always returned for programmatic access

---

**Created:** 2026-01-04  
**Compatible with:** Databricks Runtime 14.3+, ML Runtime with GPU  
**Tested on:** Python 3.10+, PyTorch 2.2+

