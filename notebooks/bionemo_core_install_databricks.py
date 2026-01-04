# Databricks notebook source
# MAGIC %md
# MAGIC ## 🧬 BioNeMo Core - Installation & Validation
# MAGIC
# MAGIC **Official Resources:**
# MAGIC - PyPI: https://pypi.org/project/bionemo-core/
# MAGIC - Documentation: https://docs.nvidia.com/bionemo-framework/latest/
# MAGIC - GitHub: https://github.com/NVIDIA/bionemo-framework

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 1: Check Existing Installation

# COMMAND ----------
# Check if bionemo-core is already installed
import subprocess
import sys

print("🔍 Checking for existing bionemo-core installation...")

try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "bionemo-core"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                version = line.split(':')[1].strip()
                print(f"✅ bionemo-core is already installed: v{version}")
                break
    else:
        print("ℹ️  bionemo-core not found - will install in next cell")
except Exception as e:
    print(f"⚠️  Could not check: {str(e)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 2: Install BioNeMo Core
# MAGIC
# MAGIC **PyPI:** https://pypi.org/project/bionemo-core/

# COMMAND ----------
# Install bionemo-core (>= 0.2.0)
%pip install "bionemo-core>=0.2.0"

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 3: Restart Python Kernel
# MAGIC
# MAGIC **Important:** Restart Python to load the newly installed packages.

# COMMAND ----------
# Restart Python kernel to load new packages
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 4: Validate Installation & Test Functionality

# COMMAND ----------
print("=" * 80)
print("🧬 BIONEMO CORE VALIDATION")
print("=" * 80)

# Initialize results
results = {
    "bionemo_core_installed": False,
    "bionemo_version": None,
    "autocast_functional": False,
    "optional_packages_available": {
        "bionemo_llm": False,
        "bionemo_esm2": False,
        "bionemo_evo2": False
    },
    "documentation_links": [
        "https://pypi.org/project/bionemo-core/",
        "https://docs.nvidia.com/bionemo-framework/latest/",
        "https://github.com/NVIDIA/bionemo-framework"
    ]
}

# Test 1: Import bionemo.core
print("\n📦 Test 1: Import bionemo.core")
print("─" * 80)

try:
    from bionemo.core import __version__
    results["bionemo_core_installed"] = True
    results["bionemo_version"] = __version__
    print(f"   ✅ Successfully imported bionemo.core")
    print(f"   ℹ️  Version: {__version__}")
except ImportError as e:
    print(f"   ❌ Import failed: {str(e)}")
    print(f"   💡 Try reinstalling: %pip install \"bionemo-core>=0.2.0\"")

# Test 2: Import and test get_autocast_dtype
print("\n📦 Test 2: Import bionemo.core.utils.dtype.get_autocast_dtype")
print("─" * 80)

if results["bionemo_core_installed"]:
    try:
        from bionemo.core.utils.dtype import get_autocast_dtype
        print(f"   ✅ Successfully imported get_autocast_dtype")
        
        # Test 3: Test get_autocast_dtype functionality
        print("\n📦 Test 3: Test get_autocast_dtype('bfloat16')")
        print("─" * 80)
        
        import torch
        
        # Test bfloat16
        dtype_bf16 = get_autocast_dtype('bfloat16')
        print(f"   ✅ get_autocast_dtype('bfloat16') = {dtype_bf16}")
        
        if dtype_bf16 == torch.bfloat16:
            print(f"   ✅ Correct: dtype matches torch.bfloat16")
            results["autocast_functional"] = True
        
        # Test float16
        dtype_fp16 = get_autocast_dtype('float16')
        print(f"   ✅ get_autocast_dtype('float16') = {dtype_fp16}")
        
        if dtype_fp16 == torch.float16:
            print(f"   ✅ Correct: dtype matches torch.float16")
            
    except ImportError as e:
        print(f"   ❌ Import failed: {str(e)}")
    except Exception as e:
        print(f"   ⚠️  Test failed: {str(e)}")

# Test 4: Optional model packages (non-fatal)
print("\n📦 Test 4: Optional Model Packages (Non-Fatal)")
print("─" * 80)

import importlib

optional_packages = {
    "bionemo_llm": "BioNeMo LLM (BioBert)",
    "bionemo_esm2": "ESM2 protein model",
    "bionemo_evo2": "Evo2 genomics model"
}

for package, description in optional_packages.items():
    try:
        importlib.import_module(package)
        results["optional_packages_available"][package] = True
        print(f"   ✅ {package}: Available ({description})")
    except ImportError:
        print(f"   ℹ️  {package}: Not installed (optional)")

# Final Summary
print("\n" + "=" * 80)
print("📊 VALIDATION SUMMARY")
print("=" * 80)

print(f"\nBioNeMo Core:")
print(f"   Installed: {'✅ Yes' if results['bionemo_core_installed'] else '❌ No'}")
if results['bionemo_version']:
    print(f"   Version: {results['bionemo_version']}")
print(f"   Autocast: {'✅ Functional' if results['autocast_functional'] else '⚠️ Not Tested'}")

available = sum(results['optional_packages_available'].values())
total = len(results['optional_packages_available'])
print(f"\nOptional Models: {available}/{total} available")

print(f"\n📚 Documentation:")
for link in results['documentation_links']:
    print(f"   • {link}")

print("=" * 80)

# Return results for programmatic access
results

