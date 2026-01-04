# Databricks notebook source
# MAGIC %md
# MAGIC ## 🧬 BioNeMo Core Installation and Validation
# MAGIC
# MAGIC This cell installs and validates NVIDIA BioNeMo Framework Core package.
# MAGIC
# MAGIC **Official Resources:**
# MAGIC - PyPI: https://pypi.org/project/bionemo-core/
# MAGIC - Documentation: https://docs.nvidia.com/bionemo-framework/latest/
# MAGIC - GitHub: https://github.com/NVIDIA/bionemo-framework
# MAGIC
# MAGIC **What This Cell Does:**
# MAGIC 1. Checks if bionemo-core is already installed
# MAGIC 2. Installs bionemo-core >= 0.2.0 if needed
# MAGIC 3. Validates core functionality (autocast, dtype utils)
# MAGIC 4. Attempts to import optional model packages (non-fatal)
# MAGIC 5. Returns comprehensive validation results

# COMMAND ----------
import sys
import subprocess
import importlib
from typing import Dict, Any, List

print("=" * 80)
print("🧬 BIONEMO CORE INSTALLATION AND VALIDATION")
print("=" * 80)

# Initialize results dictionary
bionemo_install_results: Dict[str, Any] = {
    "bionemo_core_installed": False,
    "bionemo_version": None,
    "autocast_functional": False,
    "optional_packages_available": {
        "bionemo_llm": False,
        "bionemo_esm2": False,
        "bionemo_evo2": False
    },
    "installation_commands": [],
    "documentation_links": [
        "https://pypi.org/project/bionemo-core/",
        "https://docs.nvidia.com/bionemo-framework/latest/",
        "https://github.com/NVIDIA/bionemo-framework"
    ],
    "errors": [],
    "warnings": []
}

# ============================================================================
# SECTION 1: INSTALLATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: BIONEMO CORE INSTALLATION")
print("=" * 80)

print("\n📦 Checking if bionemo-core is already installed...")

# Check if already installed
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "bionemo-core"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        # Extract version
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                installed_version = line.split(':')[1].strip()
                print(f"   ✅ bionemo-core already installed: v{installed_version}")
                bionemo_install_results["bionemo_core_installed"] = True
                bionemo_install_results["bionemo_version"] = installed_version
                break
    else:
        print(f"   ℹ️  bionemo-core not found")
        
except Exception as e:
    print(f"   ⚠️  Could not check installation: {str(e)}")

# Install if not already installed
if not bionemo_install_results["bionemo_core_installed"]:
    print("\n🔧 Installing bionemo-core...")
    print("   PyPI: https://pypi.org/project/bionemo-core/")
    
    try:
        # Use %pip for Databricks compatibility
        # Note: In actual notebook, uncomment the %pip line below
        # %pip install "bionemo-core>=0.2.0"
        
        # For Python execution (when not in notebook):
        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "bionemo-core>=0.2.0"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if install_result.returncode == 0:
            print(f"   ✅ Installation successful!")
            
            # Get installed version
            version_result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "bionemo-core"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if version_result.returncode == 0:
                for line in version_result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        installed_version = line.split(':')[1].strip()
                        bionemo_install_results["bionemo_version"] = installed_version
                        print(f"   ℹ️  Installed version: v{installed_version}")
                        break
            
            bionemo_install_results["bionemo_core_installed"] = True
            
        else:
            error_msg = f"Installation failed: {install_result.stderr}"
            bionemo_install_results["errors"].append(error_msg)
            print(f"   ❌ {error_msg}")
            
            # Add manual installation command
            bionemo_install_results["installation_commands"].append(
                '%pip install "bionemo-core>=0.2.0"'
            )
            print(f"\n💡 Manual Installation Command:")
            print(f'   %pip install "bionemo-core>=0.2.0"')
            
    except subprocess.TimeoutExpired:
        error_msg = "Installation timed out (>300s)"
        bionemo_install_results["errors"].append(error_msg)
        print(f"   ❌ {error_msg}")
        
        bionemo_install_results["installation_commands"].append(
            '%pip install "bionemo-core>=0.2.0"'
        )
        
    except Exception as e:
        error_msg = f"Installation error: {str(e)}"
        bionemo_install_results["errors"].append(error_msg)
        print(f"   ❌ {error_msg}")
        
        bionemo_install_results["installation_commands"].append(
            '%pip install "bionemo-core>=0.2.0"'
        )

# ============================================================================
# SECTION 2: IMPORT VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: IMPORT VALIDATION")
print("=" * 80)

if bionemo_install_results["bionemo_core_installed"]:
    
    # Test 2.1: Import bionemo.core.__version__
    print("\n📦 Test 2.1: Import bionemo.core.__version__")
    print("─" * 80)
    
    try:
        from bionemo.core import __version__
        bionemo_install_results["bionemo_version"] = __version__
        print(f"   ✅ Successfully imported bionemo.core")
        print(f"   ℹ️  Version: {__version__}")
        
    except ImportError as e:
        error_msg = f"Failed to import bionemo.core: {str(e)}"
        bionemo_install_results["errors"].append(error_msg)
        print(f"   ❌ {error_msg}")
        print(f"   📚 Documentation: https://docs.nvidia.com/bionemo-framework/latest/")
    
    # Test 2.2: Import bionemo.core.utils.dtype
    print("\n📦 Test 2.2: Import bionemo.core.utils.dtype")
    print("─" * 80)
    
    try:
        from bionemo.core.utils.dtype import get_autocast_dtype
        print(f"   ✅ Successfully imported get_autocast_dtype")
        
        # Test 2.3: Test get_autocast_dtype('bfloat16') functionality
        print("\n📦 Test 2.3: Test get_autocast_dtype('bfloat16') Functionality")
        print("─" * 80)
        
        try:
            import torch
            
            # Test with bfloat16
            autocast_dtype = get_autocast_dtype('bfloat16')
            print(f"   ✅ get_autocast_dtype('bfloat16') returned: {autocast_dtype}")
            
            # Verify it's a valid PyTorch dtype
            if autocast_dtype == torch.bfloat16:
                print(f"   ✅ Returned dtype matches torch.bfloat16")
                bionemo_install_results["autocast_functional"] = True
            else:
                warning_msg = f"Unexpected dtype: {autocast_dtype} (expected torch.bfloat16)"
                bionemo_install_results["warnings"].append(warning_msg)
                print(f"   ⚠️  {warning_msg}")
            
            # Test with float16
            autocast_dtype_fp16 = get_autocast_dtype('float16')
            print(f"   ✅ get_autocast_dtype('float16') returned: {autocast_dtype_fp16}")
            
            if autocast_dtype_fp16 == torch.float16:
                print(f"   ✅ Returned dtype matches torch.float16")
            
        except Exception as e:
            error_msg = f"get_autocast_dtype test failed: {str(e)}"
            bionemo_install_results["errors"].append(error_msg)
            print(f"   ❌ {error_msg}")
            
    except ImportError as e:
        error_msg = f"Failed to import get_autocast_dtype: {str(e)}"
        bionemo_install_results["errors"].append(error_msg)
        print(f"   ❌ {error_msg}")
        print(f"   📚 API Reference: https://docs.nvidia.com/bionemo-framework/latest/api/")

else:
    print("\n⏭️  Skipping import validation (bionemo-core not installed)")
    print(f"   💡 Install with: %pip install \"bionemo-core>=0.2.0\"")

# ============================================================================
# SECTION 3: OPTIONAL PACKAGES (NON-FATAL)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: OPTIONAL MODEL PACKAGES")
print("=" * 80)

print("\nℹ️  Attempting to import optional BioNeMo model packages...")
print("   (These are optional - failures will not stop the notebook)")

optional_packages = {
    "bionemo_llm": {
        "description": "BioNeMo LLM models (BioBert)",
        "install_cmd": "pip install bionemo-llm"
    },
    "bionemo_esm2": {
        "description": "ESM2 protein language model",
        "install_cmd": "pip install bionemo-esm2"
    },
    "bionemo_evo2": {
        "description": "Evo2 genomics foundation model",
        "install_cmd": "pip install bionemo-evo2"
    }
}

for package_name, package_info in optional_packages.items():
    print(f"\n📦 Testing: {package_name}")
    print(f"   Description: {package_info['description']}")
    
    try:
        # Attempt import
        importlib.import_module(package_name)
        bionemo_install_results["optional_packages_available"][package_name] = True
        print(f"   ✅ Available")
        
    except ImportError:
        print(f"   ℹ️  Not installed (optional)")
        print(f"   💡 Install with: {package_info['install_cmd']}")

# ============================================================================
# SECTION 4: FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\n📊 BioNeMo Core Status:")
print(f"   Installed: {'✅ Yes' if bionemo_install_results['bionemo_core_installed'] else '❌ No'}")
if bionemo_install_results["bionemo_version"]:
    print(f"   Version: {bionemo_install_results['bionemo_version']}")
print(f"   Autocast Functional: {'✅ Yes' if bionemo_install_results['autocast_functional'] else '⚠️ Not Tested'}")

available_models = sum(bionemo_install_results["optional_packages_available"].values())
total_models = len(bionemo_install_results["optional_packages_available"])
print(f"\n📦 Optional Models: {available_models}/{total_models} available")
for model_name, is_available in bionemo_install_results["optional_packages_available"].items():
    status = "✅" if is_available else "ℹ️"
    print(f"   {status} {model_name}")

if bionemo_install_results["errors"]:
    print(f"\n🚨 Errors ({len(bionemo_install_results['errors'])}):")
    for i, error in enumerate(bionemo_install_results["errors"], 1):
        print(f"   {i}. {error}")

if bionemo_install_results["warnings"]:
    print(f"\n⚠️  Warnings ({len(bionemo_install_results['warnings'])}):")
    for i, warning in enumerate(bionemo_install_results["warnings"], 1):
        print(f"   {i}. {warning}")

if bionemo_install_results["installation_commands"]:
    print(f"\n💡 Manual Installation Commands:")
    for i, cmd in enumerate(bionemo_install_results["installation_commands"], 1):
        print(f"   {i}. {cmd}")

print(f"\n📚 Documentation Links:")
for i, link in enumerate(bionemo_install_results["documentation_links"], 1):
    print(f"   {i}. {link}")

print("\n" + "=" * 80)

# Display results as a dictionary (useful for programmatic access)
print("\n📋 Results Dictionary:")
print("─" * 80)
import json
print(json.dumps({
    "bionemo_core_installed": bionemo_install_results["bionemo_core_installed"],
    "bionemo_version": bionemo_install_results["bionemo_version"],
    "autocast_functional": bionemo_install_results["autocast_functional"],
    "optional_packages_available": bionemo_install_results["optional_packages_available"],
    "errors_count": len(bionemo_install_results["errors"]),
    "warnings_count": len(bionemo_install_results["warnings"])
}, indent=2))

print("=" * 80)

# For Databricks: uncomment to restart Python after installation
# if bionemo_install_results["bionemo_core_installed"]:
#     dbutils.library.restartPython()

