# Databricks notebook source
# MAGIC %md
# MAGIC ## ⚡ Megatron-Core & PyTorch Lightning Installation and Validation
# MAGIC
# MAGIC Installs and validates PyTorch Lightning and NeMo Toolkit (with Megatron-Core) 
# MAGIC with critical compatibility checks for BioNeMo Framework.
# MAGIC
# MAGIC **Critical Compatibility Note:**
# MAGIC - PyTorch Lightning >= 2.5.0 breaks Megatron callbacks
# MAGIC - This cell enforces version constraint: `pytorch-lightning>=2.0.7,<2.5.0`
# MAGIC
# MAGIC **Official Resources:**
# MAGIC - PyTorch Lightning: https://pypi.org/project/pytorch-lightning/
# MAGIC - NeMo Toolkit: https://pypi.org/project/nemo-toolkit/
# MAGIC - Megatron-Core: https://github.com/NVIDIA/Megatron-LM
# MAGIC - BioNeMo Docs: https://docs.nvidia.com/bionemo-framework/latest/

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 1: Install PyTorch Lightning (with version constraint)

# COMMAND ----------
# Install PyTorch Lightning with version constraint for Megatron compatibility
# PyPI: https://pypi.org/project/pytorch-lightning/
# Version constraint: >=2.0.7,<2.5.0 (prevents Megatron callback breakage)

%pip install 'pytorch-lightning>=2.0.7,<2.5.0'

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 2: Install NeMo Toolkit (includes Megatron-Core)

# COMMAND ----------
# Install NeMo Toolkit (>= 1.22.0)
# PyPI: https://pypi.org/project/nemo-toolkit/
# Note: NeMo Toolkit provides Megatron-Core as a dependency

%pip install 'nemo-toolkit[all]>=1.22.0'

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 3: Restart Python Kernel
# MAGIC
# MAGIC **Critical:** Restart Python to load newly installed packages

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Step 4: Comprehensive Validation and Compatibility Checks

# COMMAND ----------
import sys
import subprocess
from typing import Dict, Any, List
from packaging import version

print("=" * 80)
print("⚡ MEGATRON-CORE & PYTORCH LIGHTNING VALIDATION")
print("=" * 80)

# Initialize comprehensive results dictionary
validation_results: Dict[str, Any] = {
    "pytorch_lightning_version": None,
    "pytorch_lightning_safe": False,
    "nemo_toolkit_version": None,
    "megatron_available": False,
    "gpu_strategy_available": False,
    "nccl_available": False,
    "fsdp_available": False,
    "compatibility_matrix": {},
    "critical_warnings": [],
    "installation_commands": [],
    "documentation_links": [
        "https://pypi.org/project/pytorch-lightning/",
        "https://pypi.org/project/nemo-toolkit/",
        "https://github.com/NVIDIA/Megatron-LM",
        "https://docs.nvidia.com/bionemo-framework/latest/"
    ]
}

# ============================================================================
# SECTION 1: PYTORCH LIGHTNING VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: PYTORCH LIGHTNING VALIDATION")
print("=" * 80)

print("\n📦 Test 1.1: Import PyTorch Lightning and Check Version")
print("─" * 80)

try:
    import pytorch_lightning as pl
    
    pl_version = pl.__version__
    validation_results["pytorch_lightning_version"] = pl_version
    
    print(f"   ✅ PyTorch Lightning imported successfully")
    print(f"   ℹ️  Version: {pl_version}")
    
    # Critical version check: < 2.5.0
    try:
        if version.parse(pl_version) < version.parse("2.5.0"):
            validation_results["pytorch_lightning_safe"] = True
            print(f"   ✅ Version check: {pl_version} < 2.5.0 (Megatron compatible)")
        else:
            validation_results["pytorch_lightning_safe"] = False
            
            # CRITICAL WARNING
            warning_msg = f"PyTorch Lightning {pl_version} >= 2.5.0 - BREAKS MEGATRON CALLBACKS!"
            validation_results["critical_warnings"].append(warning_msg)
            
            print(f"\n   {'🚨' * 40}")
            print(f"   ❌ CRITICAL WARNING: {warning_msg}")
            print(f"   {'🚨' * 40}")
            print(f"   Known Issue: Megatron callbacks fail with PyTorch Lightning >= 2.5.0")
            print(f"   Impact: Training will fail with callback errors")
            print(f"")
            print(f"   💡 REQUIRED ACTION: Downgrade PyTorch Lightning")
            print(f"   Run in a new cell:")
            print(f"      %pip install 'pytorch-lightning>=2.0.7,<2.5.0'")
            print(f"      dbutils.library.restartPython()")
            print(f"")
            print(f"   📚 References:")
            print(f"      - https://github.com/NVIDIA/NeMo/issues/")
            print(f"      - https://pypi.org/project/pytorch-lightning/")
            print(f"   {'🚨' * 40}\n")
            
            validation_results["installation_commands"].append(
                "%pip install 'pytorch-lightning>=2.0.7,<2.5.0'"
            )
            
    except Exception as e:
        print(f"   ⚠️  Could not parse version: {str(e)}")
        
except ImportError as e:
    print(f"   ❌ PyTorch Lightning not installed: {str(e)}")
    print(f"   📚 PyPI: https://pypi.org/project/pytorch-lightning/")
    print(f"   💡 Install with: %pip install 'pytorch-lightning>=2.0.7,<2.5.0'")
    
    validation_results["installation_commands"].append(
        "%pip install 'pytorch-lightning>=2.0.7,<2.5.0'"
    )

# ============================================================================
# SECTION 2: NEMO TOOLKIT VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: NEMO TOOLKIT VALIDATION")
print("=" * 80)

print("\n📦 Test 2.1: Import NeMo Toolkit and Check Version")
print("─" * 80)

try:
    import nemo
    
    nemo_version = nemo.__version__
    validation_results["nemo_toolkit_version"] = nemo_version
    
    print(f"   ✅ NeMo Toolkit imported successfully")
    print(f"   ℹ️  Version: {nemo_version}")
    
    # Check version >= 1.22.0
    try:
        if version.parse(nemo_version) >= version.parse("1.22.0"):
            print(f"   ✅ Version check: {nemo_version} >= 1.22.0 (BioNeMo compatible)")
        else:
            warning_msg = f"NeMo Toolkit {nemo_version} < 1.22.0 (may not be BioNeMo compatible)"
            validation_results["critical_warnings"].append(warning_msg)
            print(f"   ⚠️  {warning_msg}")
            print(f"   💡 Upgrade with: %pip install --upgrade 'nemo-toolkit[all]>=1.22.0'")
            
            validation_results["installation_commands"].append(
                "%pip install --upgrade 'nemo-toolkit[all]>=1.22.0'"
            )
    except Exception as e:
        print(f"   ⚠️  Could not parse version: {str(e)}")
    
    # Test 2.2: Import NeMo Core
    print("\n📦 Test 2.2: Import NeMo Core Modules")
    print("─" * 80)
    
    try:
        from nemo.core import ModelPT
        print(f"   ✅ nemo.core.ModelPT imported successfully")
    except ImportError as e:
        print(f"   ⚠️  Could not import nemo.core: {str(e)}")
        
except ImportError as e:
    print(f"   ❌ NeMo Toolkit not installed: {str(e)}")
    print(f"   📚 PyPI: https://pypi.org/project/nemo-toolkit/")
    print(f"   💡 Install with: %pip install 'nemo-toolkit[all]>=1.22.0'")
    
    validation_results["installation_commands"].append(
        "%pip install 'nemo-toolkit[all]>=1.22.0'"
    )

# ============================================================================
# SECTION 3: MEGATRON-CORE VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: MEGATRON-CORE VALIDATION")
print("=" * 80)

print("\n📦 Test 3.1: Attempt Megatron-Core Import")
print("─" * 80)
print("   ℹ️  Megatron-Core is provided by NeMo Toolkit as a dependency")

try:
    from megatron.core import parallel_state
    
    validation_results["megatron_available"] = True
    print(f"   ✅ megatron.core.parallel_state imported successfully")
    
    # Try to get Megatron version
    try:
        import megatron
        if hasattr(megatron, '__version__'):
            meg_version = megatron.__version__
            validation_results["compatibility_matrix"]["megatron_version"] = meg_version
            print(f"   ℹ️  Megatron-Core version: {meg_version}")
        elif hasattr(megatron, 'core') and hasattr(megatron.core, '__version__'):
            meg_version = megatron.core.__version__
            validation_results["compatibility_matrix"]["megatron_version"] = meg_version
            print(f"   ℹ️  Megatron-Core version: {meg_version}")
        else:
            print(f"   ℹ️  Megatron-Core version: Unknown (bundled with NeMo)")
    except Exception as e:
        print(f"   ℹ️  Could not determine Megatron version: {str(e)}")
        
except ImportError as e:
    print(f"   ⚠️  Megatron-Core not available: {str(e)}")
    print(f"   ℹ️  This is expected if NeMo Toolkit installation failed")
    print(f"   ℹ️  Megatron-Core is bundled with NeMo Toolkit")
    print(f"   📚 GitHub: https://github.com/NVIDIA/Megatron-LM")

# ============================================================================
# SECTION 4: PYTORCH LIGHTNING GPU STRATEGY TESTING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: PYTORCH LIGHTNING GPU STRATEGY TESTING")
print("=" * 80)

if validation_results["pytorch_lightning_version"]:
    print("\n📦 Test 4.1: GPU Strategy Auto-Detection")
    print("─" * 80)
    
    try:
        import torch
        import pytorch_lightning as pl
        
        if torch.cuda.is_available():
            # Test Trainer instantiation with GPU
            try:
                trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=1,
                    max_epochs=1,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    logger=False,
                    enable_checkpointing=False
                )
                
                validation_results["gpu_strategy_available"] = True
                
                print(f"   ✅ Trainer instantiated with GPU accelerator")
                print(f"      Accelerator: {trainer.accelerator.__class__.__name__}")
                print(f"      Strategy: {trainer.strategy.__class__.__name__}")
                print(f"      Devices: {trainer.num_devices}")
                
            except Exception as e:
                print(f"   ❌ Trainer instantiation failed: {str(e)}")
                validation_results["critical_warnings"].append(
                    f"GPU strategy initialization failed: {str(e)}"
                )
        else:
            print(f"   ⏭️  CUDA not available - skipping GPU strategy test")
            
    except Exception as e:
        print(f"   ❌ GPU strategy test failed: {str(e)}")
else:
    print("\n⏭️  Skipping GPU strategy tests (PyTorch Lightning not available)")

# ============================================================================
# SECTION 5: DISTRIBUTED ENVIRONMENT CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: DISTRIBUTED ENVIRONMENT CHECKS")
print("=" * 80)

print("\n📦 Test 5.1: NCCL Availability")
print("─" * 80)

try:
    import torch
    
    if torch.cuda.is_available():
        if torch.cuda.nccl.is_available():
            validation_results["nccl_available"] = True
            
            try:
                nccl_version = torch.cuda.nccl.version()
                print(f"   ✅ NCCL available")
                print(f"   ℹ️  NCCL version: {nccl_version}")
            except:
                print(f"   ✅ NCCL available (version unknown)")
        else:
            print(f"   ⚠️  NCCL not available")
            print(f"   ℹ️  NCCL is required for multi-GPU distributed training")
    else:
        print(f"   ⏭️  CUDA not available - cannot check NCCL")
        
except Exception as e:
    print(f"   ⚠️  NCCL check failed: {str(e)}")

print("\n📦 Test 5.2: torch.distributed Availability")
print("─" * 80)

try:
    import torch.distributed as dist
    
    if dist.is_available():
        print(f"   ✅ torch.distributed is available")
        
        # Check available backends
        backends = []
        if dist.is_nccl_available():
            backends.append("nccl")
        if dist.is_gloo_available():
            backends.append("gloo")
        if dist.is_mpi_available():
            backends.append("mpi")
        
        print(f"   ℹ️  Available backends: {', '.join(backends)}")
    else:
        print(f"   ⚠️  torch.distributed not available")
        
except Exception as e:
    print(f"   ⚠️  torch.distributed check failed: {str(e)}")

print("\n📦 Test 5.3: FSDP Strategy Support")
print("─" * 80)

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    
    validation_results["fsdp_available"] = True
    
    print(f"   ✅ FSDP (FullyShardedDataParallel) available")
    print(f"   ✅ ShardingStrategy available")
    
    # List available sharding strategies
    strategies = [s.name for s in ShardingStrategy]
    print(f"   ℹ️  Available strategies: {', '.join(strategies)}")
    
except ImportError:
    print(f"   ⚠️  FSDP not available (requires PyTorch >= 2.0.0)")

# Check PyTorch Lightning FSDP strategy
if validation_results["pytorch_lightning_version"]:
    try:
        from pytorch_lightning.strategies import FSDPStrategy
        print(f"   ✅ PyTorch Lightning FSDPStrategy available")
    except ImportError:
        print(f"   ℹ️  PyTorch Lightning FSDPStrategy not available (Lightning >= 1.9.0 required)")

# ============================================================================
# SECTION 6: COMPATIBILITY MATRIX
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: COMPATIBILITY MATRIX")
print("=" * 80)

# Build comprehensive compatibility matrix
validation_results["compatibility_matrix"] = {
    "pytorch_lightning": {
        "version": validation_results["pytorch_lightning_version"],
        "safe_version": validation_results["pytorch_lightning_safe"],
        "constraint": ">=2.0.7,<2.5.0"
    },
    "nemo_toolkit": {
        "version": validation_results["nemo_toolkit_version"],
        "constraint": ">=1.22.0"
    },
    "megatron_core": {
        "available": validation_results["megatron_available"],
        "source": "bundled with NeMo Toolkit"
    },
    "distributed": {
        "nccl": validation_results["nccl_available"],
        "fsdp": validation_results["fsdp_available"],
        "gpu_strategy": validation_results["gpu_strategy_available"]
    }
}

print("\n📊 Compatibility Report:")
print("─" * 80)

# PyTorch Lightning
pl_status = "✅" if validation_results["pytorch_lightning_safe"] else "❌"
print(f"{pl_status} PyTorch Lightning: {validation_results['pytorch_lightning_version']}")
if validation_results["pytorch_lightning_version"]:
    if validation_results["pytorch_lightning_safe"]:
        print(f"   ✅ Version is Megatron compatible (< 2.5.0)")
    else:
        print(f"   ❌ Version breaks Megatron (>= 2.5.0)")

# NeMo Toolkit
nemo_status = "✅" if validation_results["nemo_toolkit_version"] else "❌"
print(f"\n{nemo_status} NeMo Toolkit: {validation_results['nemo_toolkit_version'] or 'Not installed'}")

# Megatron-Core
meg_status = "✅" if validation_results["megatron_available"] else "⚠️"
print(f"\n{meg_status} Megatron-Core: {'Available' if validation_results['megatron_available'] else 'Not available'}")
if validation_results["megatron_available"]:
    print(f"   ℹ️  Provided by NeMo Toolkit")

# Distributed Support
print(f"\n📡 Distributed Training Support:")
nccl_status = "✅" if validation_results["nccl_available"] else "⚠️"
fsdp_status = "✅" if validation_results["fsdp_available"] else "⚠️"
gpu_strat_status = "✅" if validation_results["gpu_strategy_available"] else "⚠️"

print(f"   {nccl_status} NCCL: {'Available' if validation_results['nccl_available'] else 'Not available'}")
print(f"   {fsdp_status} FSDP: {'Available' if validation_results['fsdp_available'] else 'Not available'}")
print(f"   {gpu_strat_status} GPU Strategy: {'Available' if validation_results['gpu_strategy_available'] else 'Not tested'}")

# ============================================================================
# SECTION 7: CRITICAL WARNINGS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: CRITICAL WARNINGS & RECOMMENDATIONS")
print("=" * 80)

if validation_results["critical_warnings"]:
    print(f"\n🚨 CRITICAL WARNINGS ({len(validation_results['critical_warnings'])}):")
    print("─" * 80)
    
    for i, warning in enumerate(validation_results["critical_warnings"], 1):
        print(f"\n{i}. {warning}")
    
    if validation_results["installation_commands"]:
        print(f"\n💡 REQUIRED ACTIONS:")
        print("─" * 80)
        for i, cmd in enumerate(validation_results["installation_commands"], 1):
            print(f"{i}. {cmd}")
        print(f"\nAfter running commands, execute:")
        print(f"   dbutils.library.restartPython()")
else:
    print(f"\n✅ No critical warnings - all compatibility checks passed!")

# Known Issues List
print(f"\n📋 Known Compatibility Issues:")
print("─" * 80)
print(f"1. PyTorch Lightning >= 2.5.0 breaks Megatron callbacks")
print(f"   Status: {'❌ DETECTED' if not validation_results['pytorch_lightning_safe'] else '✅ Not present'}")
print(f"   Solution: Use pytorch-lightning>=2.0.7,<2.5.0")

# ============================================================================
# SECTION 8: FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

# Overall status
all_compatible = (
    validation_results["pytorch_lightning_safe"] and
    validation_results["nemo_toolkit_version"] is not None and
    validation_results["megatron_available"]
)

if all_compatible:
    print(f"\n✅ ALL COMPATIBILITY CHECKS PASSED")
    print(f"   Environment is ready for BioNeMo training with Megatron-Core")
elif validation_results["critical_warnings"]:
    print(f"\n❌ CRITICAL ISSUES DETECTED")
    print(f"   {len(validation_results['critical_warnings'])} critical warning(s) require attention")
else:
    print(f"\n⚠️  PARTIAL COMPATIBILITY")
    print(f"   Some components missing but no critical issues detected")

print(f"\n📚 Documentation & Resources:")
print("─" * 80)
for i, link in enumerate(validation_results["documentation_links"], 1):
    print(f"{i}. {link}")

print("\n" + "=" * 80)

# Display results dictionary
print("\n📋 Results Dictionary:")
import json
print(json.dumps({
    "pytorch_lightning_version": validation_results["pytorch_lightning_version"],
    "pytorch_lightning_safe": validation_results["pytorch_lightning_safe"],
    "nemo_toolkit_version": validation_results["nemo_toolkit_version"],
    "megatron_available": validation_results["megatron_available"],
    "gpu_strategy_available": validation_results["gpu_strategy_available"],
    "nccl_available": validation_results["nccl_available"],
    "fsdp_available": validation_results["fsdp_available"],
    "critical_warnings_count": len(validation_results["critical_warnings"]),
    "all_compatible": all_compatible
}, indent=2))

print("=" * 80)

# Return results for programmatic access
validation_results

