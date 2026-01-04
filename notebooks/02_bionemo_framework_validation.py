# Databricks notebook source
# MAGIC %md
# MAGIC # 🧬 BioNeMo Framework Validation for Databricks
# MAGIC
# MAGIC Comprehensive validation notebook for NVIDIA BioNeMo Framework installation on Databricks.
# MAGIC This notebook extends the CUDA Healthcheck Tool with BioNeMo-specific validations.
# MAGIC
# MAGIC ## What This Notebook Does:
# MAGIC
# MAGIC 1. ✅ Validates Databricks environment (leverages existing healthcheck)
# MAGIC 2. ✅ Checks CUDA availability for BioNeMo workloads
# MAGIC 3. ✅ Validates PyTorch installation and CUDA linkage
# MAGIC 4. ✅ **Tests PyTorch Lightning GPU compatibility (NEW!)**
# MAGIC 5. ✅ **Performs CUDA functional testing (NEW!)**
# MAGIC 6. ✅ **Validates BioNeMo core packages availability (NEW!)**
# MAGIC
# MAGIC ## BioNeMo Framework Overview:
# MAGIC
# MAGIC BioNeMo provides three types of packages:
# MAGIC
# MAGIC ### 1. Recipes (Native PyTorch + Megatron-FSDP + TE)
# MAGIC - `bionemo-recipes/` - Lightweight, pip-installable recipes
# MAGIC - ESM2, Geneformer, DNABERT, Llama3, ViT models
# MAGIC - Supports Megatron-FSDP, Transformer Engine, FP8
# MAGIC - Ideal for Databricks environments
# MAGIC
# MAGIC ### 2. 5D Parallelism Models (NeMo + Megatron-Core)
# MAGIC - `sub-packages/` - Explicit 5D parallelism
# MAGIC - bionemo-evo2, bionemo-geneformer, bionemo-llm
# MAGIC - Requires Docker image with NeMo pre-installed
# MAGIC
# MAGIC ### 3. Tooling (Dataloading & Processing)
# MAGIC - `sub-packages/` - Lightweight utilities
# MAGIC - bionemo-scdl (Single Cell Data Loader)
# MAGIC - bionemo-moco (Molecular Co-design)
# MAGIC - bionemo-noodles (Fast FASTA I/O)
# MAGIC
# MAGIC ## Requirements:
# MAGIC
# MAGIC - GPU-enabled Databricks cluster (A100/V100/T4)
# MAGIC - ML Runtime 14.3+ with CUDA 12.0+
# MAGIC - PyTorch 2.0+ with CUDA support
# MAGIC - PyTorch Lightning 2.0+
# MAGIC - Python 3.10+
# MAGIC
# MAGIC ## References:
# MAGIC
# MAGIC - GitHub: https://github.com/NVIDIA/bionemo-framework
# MAGIC - Documentation: https://nvidia.github.io/bionemo-framework/
# MAGIC - CUDA Healthcheck: https://github.com/TavnerJC/cuda-healthcheck-on-databricks

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📦 Cell 1: Setup and Imports
# MAGIC
# MAGIC Install CUDA Healthcheck Tool and import all required dependencies.
# MAGIC This cell reuses the existing healthcheck infrastructure.

# COMMAND ----------
import sys
import json
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("=" * 80)
print("🧬 BIONEMO FRAMEWORK VALIDATION - SETUP")
print("=" * 80)

# Install CUDA Healthcheck Tool if not already installed
try:
    from cuda_healthcheck import __version__, CUDADetector
    from cuda_healthcheck.databricks import detect_databricks_runtime, detect_gpu_auto
    from cuda_healthcheck.utils import get_cuda_packages_from_pip
    print(f"\n✅ CUDA Healthcheck v{__version__} already installed")
except ImportError:
    print("\n📦 Installing CUDA Healthcheck Tool...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
        "git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git"
    ], check=True)
    
    # Import after installation
    from cuda_healthcheck import __version__, CUDADetector
    from cuda_healthcheck.databricks import detect_databricks_runtime, detect_gpu_auto
    from cuda_healthcheck.utils import get_cuda_packages_from_pip
    print(f"✅ CUDA Healthcheck v{__version__} installed successfully")

# Import standard libraries
import importlib.util

print("\n📚 Imported dependencies:")
print(f"   • cuda_healthcheck v{__version__}")
print(f"   • Python {sys.version.split()[0]}")
print(f"   • subprocess, json, datetime")

print("\n" + "=" * 80)
print("✅ SETUP COMPLETE")
print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🎮 Cell 2: CUDA Environment Validation
# MAGIC
# MAGIC Reuses existing CUDA detection from healthcheck tool.
# MAGIC Validates GPU, CUDA runtime, driver versions, and PyTorch.
# MAGIC
# MAGIC **Note:** This leverages existing functions to avoid duplication.

# COMMAND ----------
print("=" * 80)
print("🎮 CUDA ENVIRONMENT VALIDATION FOR BIONEMO")
print("=" * 80)

# Initialize results dictionary
cuda_validation_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "databricks_runtime": None,
    "gpu_info": None,
    "cuda_environment": None,
    "pytorch_info": None,
    "status": "PENDING",
    "blockers": [],
    "warnings": []
}

try:
    # 1. Detect Databricks Runtime
    print("\n🏃 Detecting Databricks Runtime...")
    runtime_info = detect_databricks_runtime()
    cuda_validation_results["databricks_runtime"] = runtime_info
    
    print(f"   Runtime Version: {runtime_info['runtime_version']}")
    print(f"   Is ML Runtime: {runtime_info['is_ml_runtime']}")
    print(f"   Is GPU Runtime: {runtime_info['is_gpu_runtime']}")
    print(f"   Expected CUDA: {runtime_info['cuda_version']}")
    
    if not runtime_info['is_gpu_runtime']:
        cuda_validation_results["blockers"].append({
            "check": "gpu_runtime",
            "message": "Not running on a GPU-enabled runtime",
            "severity": "BLOCKER"
        })
    
    # 2. Detect GPU Hardware
    print("\n🎮 Detecting GPU Hardware...")
    gpu_info = detect_gpu_auto()
    cuda_validation_results["gpu_info"] = gpu_info
    
    gpu_count = gpu_info.get('gpu_count', 0)
    print(f"   GPU Count: {gpu_count}")
    print(f"   Environment: {gpu_info.get('environment', 'unknown')}")
    
    if gpu_count > 0 and 'gpus' in gpu_info:
        for i, gpu in enumerate(gpu_info['gpus']):
            print(f"\n   GPU {i}: {gpu.get('name', 'Unknown')}")
            print(f"      Driver: {gpu.get('driver_version', 'N/A')}")
            print(f"      Memory: {gpu.get('memory_total', 'N/A')}")
            print(f"      Compute: {gpu.get('compute_capability', 'N/A')}")
    else:
        cuda_validation_results["blockers"].append({
            "check": "gpu_detection",
            "message": f"No GPU detected: {gpu_info.get('error', 'Unknown')}",
            "severity": "BLOCKER"
        })
    
    # 3. Detect CUDA Environment
    print("\n🔧 Detecting CUDA Environment...")
    detector = CUDADetector()
    env = detector.detect_environment()
    
    cuda_env_summary = {
        "cuda_runtime": env.cuda_runtime_version,
        "cuda_driver": env.cuda_driver_version,
        "nvcc_version": env.nvcc_version,
        "libraries": []
    }
    
    print(f"   CUDA Runtime: {env.cuda_runtime_version}")
    print(f"   CUDA Driver: {env.cuda_driver_version}")
    print(f"   NVCC Version: {env.nvcc_version}")
    
    if env.cuda_runtime_version == "Not available":
        cuda_validation_results["blockers"].append({
            "check": "cuda_runtime",
            "message": "CUDA runtime not detected",
            "severity": "BLOCKER"
        })
    
    # 4. Check PyTorch Installation
    print("\n🐍 Checking PyTorch Installation...")
    pytorch_lib = next((lib for lib in env.libraries if lib.name.lower() == "torch"), None)
    
    if pytorch_lib and pytorch_lib.version != "Not installed":
        pytorch_info = {
            "version": pytorch_lib.version,
            "cuda_version": pytorch_lib.cuda_version,
            "is_compatible": pytorch_lib.is_compatible
        }
        cuda_validation_results["pytorch_info"] = pytorch_info
        
        print(f"   PyTorch Version: {pytorch_lib.version}")
        print(f"   CUDA Version: {pytorch_lib.cuda_version}")
        print(f"   Compatible: {pytorch_lib.is_compatible}")
        
        if not pytorch_lib.is_compatible:
            cuda_validation_results["warnings"].append({
                "check": "pytorch_compatibility",
                "message": f"PyTorch {pytorch_lib.version} may have compatibility issues",
                "severity": "WARNING"
            })
    else:
        cuda_validation_results["blockers"].append({
            "check": "pytorch_installation",
            "message": "PyTorch not installed",
            "severity": "BLOCKER"
        })
    
    # Store full environment
    cuda_validation_results["cuda_environment"] = cuda_env_summary
    
    # Determine overall status
    if cuda_validation_results["blockers"]:
        cuda_validation_results["status"] = "BLOCKED"
        print("\n❌ CUDA VALIDATION FAILED")
        print(f"   Blockers: {len(cuda_validation_results['blockers'])}")
        for blocker in cuda_validation_results["blockers"]:
            print(f"      • {blocker['message']}")
    else:
        cuda_validation_results["status"] = "PASSED"
        print("\n✅ CUDA VALIDATION PASSED")
        if cuda_validation_results["warnings"]:
            print(f"   Warnings: {len(cuda_validation_results['warnings'])}")

except Exception as e:
    print(f"\n❌ Error during CUDA validation: {str(e)}")
    cuda_validation_results["status"] = "ERROR"
    cuda_validation_results["blockers"].append({
        "check": "validation_error",
        "message": str(e),
        "severity": "BLOCKER"
    })

print("\n" + "=" * 80)
print(f"CUDA VALIDATION STATUS: {cuda_validation_results['status']}")
print("=" * 80)

# Display results as DataFrame
import pandas as pd

summary_data = {
    "Check": ["Databricks Runtime", "GPU Detection", "CUDA Runtime", "PyTorch"],
    "Status": [
        "✅ PASS" if runtime_info['is_gpu_runtime'] else "❌ FAIL",
        f"✅ PASS ({gpu_count} GPU)" if gpu_count > 0 else "❌ FAIL",
        "✅ PASS" if env.cuda_runtime_version != "Not available" else "❌ FAIL",
        "✅ PASS" if pytorch_lib and pytorch_lib.version != "Not installed" else "❌ FAIL"
    ],
    "Details": [
        f"Runtime {runtime_info['runtime_version']}, CUDA {runtime_info['cuda_version']}",
        gpu_info['gpus'][0]['name'] if gpu_count > 0 and 'gpus' in gpu_info else "No GPU",
        f"Runtime {env.cuda_runtime_version}, Driver {env.cuda_driver_version}",
        f"v{pytorch_lib.version} (cu{pytorch_lib.cuda_version})" if pytorch_lib and pytorch_lib.version != "Not installed" else "Not installed"
    ]
}

df_summary = pd.DataFrame(summary_data)
display(df_summary)

# COMMAND ----------
# MAGIC %md
# MAGIC ## ⚡ Cell 3: PyTorch Lightning GPU Test (NEW!)
# MAGIC
# MAGIC Tests PyTorch Lightning compatibility with GPU acceleration.
# MAGIC This is critical for BioNeMo recipes which use Lightning for training.
# MAGIC
# MAGIC **Validates:**
# MAGIC - PyTorch Lightning installation
# MAGIC - GPU device detection via Lightning
# MAGIC - Trainer initialization with GPU strategy
# MAGIC - Simple forward pass on GPU
# MAGIC - Mixed precision (FP16) support

# COMMAND ----------
print("=" * 80)
print("⚡ PYTORCH LIGHTNING GPU COMPATIBILITY TEST")
print("=" * 80)

# Initialize results dictionary
lightning_test_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "pytorch_lightning_installed": False,
    "version": None,
    "gpu_devices_detected": 0,
    "trainer_initialization": False,
    "gpu_forward_pass": False,
    "mixed_precision_support": False,
    "status": "PENDING",
    "blockers": [],
    "warnings": [],
    "benchmark_results": {}
}

try:
    # 1. Check PyTorch Lightning Installation
    print("\n📦 Checking PyTorch Lightning Installation...")
    
    try:
        import pytorch_lightning as pl
        lightning_test_results["pytorch_lightning_installed"] = True
        lightning_test_results["version"] = pl.__version__
        print(f"   ✅ PyTorch Lightning v{pl.__version__} installed")
    except ImportError:
        print("   ❌ PyTorch Lightning not installed")
        print("   💡 Installing PyTorch Lightning...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning>=2.0.0"
        ], check=True)
        
        import pytorch_lightning as pl
        lightning_test_results["pytorch_lightning_installed"] = True
        lightning_test_results["version"] = pl.__version__
        print(f"   ✅ PyTorch Lightning v{pl.__version__} installed successfully")
    
    # 2. Check GPU Availability via PyTorch
    print("\n🎮 Checking GPU Availability via PyTorch...")
    import torch
    
    if not torch.cuda.is_available():
        lightning_test_results["blockers"].append({
            "check": "torch_cuda_available",
            "message": "torch.cuda.is_available() returned False",
            "severity": "BLOCKER"
        })
        print("   ❌ torch.cuda.is_available() = False")
    else:
        gpu_count = torch.cuda.device_count()
        lightning_test_results["gpu_devices_detected"] = gpu_count
        print(f"   ✅ torch.cuda.is_available() = True")
        print(f"   ✅ GPU Devices: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            print(f"      GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 3. Test Lightning Trainer Initialization
    if lightning_test_results["gpu_devices_detected"] > 0:
        print("\n⚡ Testing Lightning Trainer Initialization...")
        
        try:
            # Create a simple Lightning module for testing
            class SimpleLightningModule(pl.LightningModule):
                def __init__(self):
                    super().__init__()
                    self.layer = torch.nn.Linear(128, 64)
                
                def forward(self, x):
                    return self.layer(x)
                
                def training_step(self, batch, batch_idx):
                    x, y = batch
                    y_hat = self(x)
                    loss = torch.nn.functional.mse_loss(y_hat, y)
                    return loss
                
                def configure_optimizers(self):
                    return torch.optim.Adam(self.parameters(), lr=1e-3)
            
            # Initialize trainer with GPU
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                max_epochs=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                enable_checkpointing=False
            )
            
            lightning_test_results["trainer_initialization"] = True
            print(f"   ✅ Trainer initialized successfully")
            print(f"      Accelerator: {trainer.accelerator.__class__.__name__}")
            print(f"      Devices: {trainer.num_devices}")
            
        except Exception as e:
            lightning_test_results["blockers"].append({
                "check": "trainer_initialization",
                "message": f"Trainer initialization failed: {str(e)}",
                "severity": "BLOCKER"
            })
            print(f"   ❌ Trainer initialization failed: {str(e)}")
        
        # 4. Test GPU Forward Pass
        print("\n🚀 Testing GPU Forward Pass...")
        
        try:
            model = SimpleLightningModule()
            model = model.cuda()
            
            # Create test input
            test_input = torch.randn(32, 128).cuda()
            
            # Perform forward pass
            with torch.no_grad():
                output = model(test_input)
            
            # Verify output is on GPU
            if output.is_cuda:
                lightning_test_results["gpu_forward_pass"] = True
                print(f"   ✅ GPU forward pass successful")
                print(f"      Input shape: {test_input.shape}")
                print(f"      Output shape: {output.shape}")
                print(f"      Device: {output.device}")
            else:
                lightning_test_results["warnings"].append({
                    "check": "gpu_forward_pass",
                    "message": "Output tensor not on GPU",
                    "severity": "WARNING"
                })
                print(f"   ⚠️  Output tensor not on GPU")
                
        except Exception as e:
            lightning_test_results["blockers"].append({
                "check": "gpu_forward_pass",
                "message": f"GPU forward pass failed: {str(e)}",
                "severity": "BLOCKER"
            })
            print(f"   ❌ GPU forward pass failed: {str(e)}")
        
        # 5. Test Mixed Precision (FP16) Support
        print("\n🔥 Testing Mixed Precision (FP16) Support...")
        
        try:
            # Check if GPU supports FP16
            gpu_capability = torch.cuda.get_device_capability(0)
            supports_fp16 = gpu_capability[0] >= 7  # Volta and newer
            
            if supports_fp16:
                # Test with automatic mixed precision
                with torch.cuda.amp.autocast():
                    test_input_fp16 = torch.randn(32, 128).cuda()
                    output_fp16 = model(test_input_fp16)
                
                lightning_test_results["mixed_precision_support"] = True
                print(f"   ✅ Mixed precision (FP16) supported")
                print(f"      GPU Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
                print(f"      Output dtype: {output_fp16.dtype}")
            else:
                lightning_test_results["warnings"].append({
                    "check": "mixed_precision",
                    "message": f"GPU compute capability {gpu_capability[0]}.{gpu_capability[1]} < 7.0, limited FP16 support",
                    "severity": "WARNING"
                })
                print(f"   ⚠️  GPU compute capability {gpu_capability[0]}.{gpu_capability[1]} < 7.0")
                print(f"      Mixed precision may have limited performance benefits")
                
        except Exception as e:
            lightning_test_results["warnings"].append({
                "check": "mixed_precision",
                "message": f"Mixed precision test failed: {str(e)}",
                "severity": "WARNING"
            })
            print(f"   ⚠️  Mixed precision test failed: {str(e)}")
        
        # 6. Benchmark GPU Performance
        print("\n📊 Benchmarking GPU Performance...")
        
        try:
            import time
            
            model = SimpleLightningModule().cuda()
            model.eval()
            
            # Warm-up
            for _ in range(10):
                with torch.no_grad():
                    _ = model(torch.randn(256, 128).cuda())
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            iterations = 100
            for _ in range(iterations):
                with torch.no_grad():
                    _ = model(torch.randn(256, 128).cuda())
            
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            
            throughput = iterations / elapsed_time
            avg_latency_ms = (elapsed_time / iterations) * 1000
            
            lightning_test_results["benchmark_results"] = {
                "iterations": iterations,
                "elapsed_time_s": round(elapsed_time, 4),
                "throughput_iter_per_s": round(throughput, 2),
                "avg_latency_ms": round(avg_latency_ms, 4)
            }
            
            print(f"   ✅ Benchmark completed")
            print(f"      Iterations: {iterations}")
            print(f"      Elapsed Time: {elapsed_time:.4f}s")
            print(f"      Throughput: {throughput:.2f} iter/s")
            print(f"      Avg Latency: {avg_latency_ms:.4f}ms")
            
        except Exception as e:
            lightning_test_results["warnings"].append({
                "check": "benchmark",
                "message": f"Benchmark failed: {str(e)}",
                "severity": "WARNING"
            })
            print(f"   ⚠️  Benchmark failed: {str(e)}")
    
    # Determine overall status
    if lightning_test_results["blockers"]:
        lightning_test_results["status"] = "BLOCKED"
        print("\n❌ PYTORCH LIGHTNING TEST FAILED")
        print(f"   Blockers: {len(lightning_test_results['blockers'])}")
        for blocker in lightning_test_results["blockers"]:
            print(f"      • {blocker['message']}")
    else:
        lightning_test_results["status"] = "PASSED"
        print("\n✅ PYTORCH LIGHTNING TEST PASSED")
        if lightning_test_results["warnings"]:
            print(f"   Warnings: {len(lightning_test_results['warnings'])}")

except Exception as e:
    print(f"\n❌ Error during Lightning test: {str(e)}")
    lightning_test_results["status"] = "ERROR"
    lightning_test_results["blockers"].append({
        "check": "lightning_test_error",
        "message": str(e),
        "severity": "BLOCKER"
    })

print("\n" + "=" * 80)
print(f"PYTORCH LIGHTNING TEST STATUS: {lightning_test_results['status']}")
print("=" * 80)

# Display results as DataFrame
summary_data = {
    "Check": [
        "PyTorch Lightning Installed",
        "GPU Devices Detected",
        "Trainer Initialization",
        "GPU Forward Pass",
        "Mixed Precision (FP16)"
    ],
    "Status": [
        "✅ PASS" if lightning_test_results["pytorch_lightning_installed"] else "❌ FAIL",
        f"✅ PASS ({lightning_test_results['gpu_devices_detected']})" if lightning_test_results["gpu_devices_detected"] > 0 else "❌ FAIL",
        "✅ PASS" if lightning_test_results["trainer_initialization"] else "❌ FAIL",
        "✅ PASS" if lightning_test_results["gpu_forward_pass"] else "❌ FAIL",
        "✅ PASS" if lightning_test_results["mixed_precision_support"] else "⚠️ WARN"
    ],
    "Details": [
        f"v{lightning_test_results['version']}" if lightning_test_results["version"] else "Not installed",
        f"{lightning_test_results['gpu_devices_detected']} GPU(s) available",
        "Trainer initialized with GPU accelerator" if lightning_test_results["trainer_initialization"] else "Failed",
        "Forward pass successful on GPU" if lightning_test_results["gpu_forward_pass"] else "Failed",
        f"FP16 supported" if lightning_test_results["mixed_precision_support"] else "Limited support"
    ]
}

df_lightning = pd.DataFrame(summary_data)
display(df_lightning)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🔥 Cell 4: CUDA Functional Testing (NEW!)
# MAGIC
# MAGIC Performs actual CUDA operations to validate GPU functionality beyond availability checks.
# MAGIC This cell tests real computational workloads that BioNeMo training will depend on.
# MAGIC
# MAGIC **Core Tests (1-7):**
# MAGIC - CUDA tensor creation on device
# MAGIC - Matrix multiplication performance (GFLOPS)
# MAGIC - CUDA memory allocation and tracking
# MAGIC - CUDA stream synchronization
# MAGIC - Mixed precision support (FP16, BF16)
# MAGIC - cuDNN availability and version
# MAGIC - NCCL availability (for distributed training)
# MAGIC
# MAGIC **Advanced CUDA Library Benchmarks (8-12):**
# MAGIC - **cuBLAS:** GEMM (General Matrix Multiply) performance at multiple sizes
# MAGIC - **cuFFT:** Fast Fourier Transform (1D, 2D) performance
# MAGIC - **cuSOLVER:** Matrix inversion and linear algebra operations
# MAGIC - **Tensor Cores:** FP16 vs FP32 performance comparison (Volta+)
# MAGIC - **Memory Bandwidth:** Device-to-device transfer speed (GB/s)
# MAGIC
# MAGIC **Why this matters for BioNeMo:**
# MAGIC - BioNeMo models require efficient tensor operations across all CUDA libraries
# MAGIC - Training uses mixed precision (FP16/BF16) and Tensor Cores for performance
# MAGIC - Multi-GPU training depends on NCCL and high memory bandwidth
# MAGIC - Memory management is critical for large protein/DNA models
# MAGIC - cuBLAS, cuFFT, cuSOLVER used in various model architectures

# COMMAND ----------
print("=" * 80)
print("🔥 CUDA FUNCTIONAL TESTING")
print("=" * 80)

# Initialize results dictionary
cuda_functional_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "cuda_functional": False,
    "memory_test_passed": False,
    "tensor_ops_speed_gflops": 0.0,
    "mixed_precision_support": {
        "float16": False,
        "bfloat16": False,
        "tf32": False
    },
    "cudnn_available": False,
    "cudnn_version": None,
    "nccl_available": False,
    "nccl_version": None,
    "tests_run": [],
    "errors": [],
    "status": "PENDING"
}

try:
    import torch
    import time
    
    if not torch.cuda.is_available():
        cuda_functional_results["errors"].append("CUDA not available - skipping functional tests")
        cuda_functional_results["status"] = "SKIPPED"
        print("\n⚠️  CUDA not available - skipping functional tests")
    else:
        # Get GPU info
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_capability = torch.cuda.get_device_capability(0)
        
        print(f"\n🎮 Testing on: {gpu_name}")
        print(f"   Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
        
        # ========================================================================
        # TEST 1: CUDA Tensor Creation
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 1: CUDA Tensor Creation")
        print("─" * 80)
        
        try:
            # Create tensors on GPU
            tensor_sizes = [(1000, 1000), (2000, 2000), (4000, 4000)]
            
            for size in tensor_sizes:
                start_time = time.time()
                tensor = torch.randn(size, device=device)
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                
                print(f"   ✅ Created {size[0]}×{size[1]} tensor in {elapsed*1000:.2f}ms")
                del tensor
            
            cuda_functional_results["tests_run"].append("tensor_creation")
            print(f"   Status: PASSED")
            
        except Exception as e:
            cuda_functional_results["errors"].append(f"Tensor creation failed: {str(e)}")
            print(f"   ❌ Tensor creation failed: {str(e)}")
        
        # ========================================================================
        # TEST 2: Matrix Multiplication Performance (GFLOPS)
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 2: Matrix Multiplication Performance")
        print("─" * 80)
        
        try:
            # Matrix multiplication benchmark
            matrix_size = 4096
            num_iterations = 10
            
            print(f"   Running {num_iterations} iterations of {matrix_size}×{matrix_size} matmul...")
            
            # Create test matrices
            A = torch.randn(matrix_size, matrix_size, device=device)
            B = torch.randn(matrix_size, matrix_size, device=device)
            
            # Warm-up
            for _ in range(5):
                C = torch.matmul(A, B)
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                C = torch.matmul(A, B)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            # Calculate GFLOPS
            # Matrix multiplication: 2*N^3 FLOPs for N×N matrices
            flops_per_matmul = 2 * (matrix_size ** 3)
            total_flops = flops_per_matmul * num_iterations
            gflops = (total_flops / elapsed) / 1e9
            
            cuda_functional_results["tensor_ops_speed_gflops"] = round(gflops, 2)
            cuda_functional_results["tests_run"].append("matrix_multiplication")
            
            print(f"   ✅ Performance: {gflops:.2f} GFLOPS")
            print(f"   ✅ Avg time per matmul: {(elapsed/num_iterations)*1000:.2f}ms")
            print(f"   Status: PASSED")
            
            # Cleanup
            del A, B, C
            
        except Exception as e:
            cuda_functional_results["errors"].append(f"Matrix multiplication failed: {str(e)}")
            print(f"   ❌ Matrix multiplication failed: {str(e)}")
        
        # ========================================================================
        # TEST 3: CUDA Memory Allocation and Tracking
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 3: CUDA Memory Allocation and Tracking")
        print("─" * 80)
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated(0)
            print(f"   Initial memory: {initial_memory / 1024**2:.2f} MB")
            
            # Allocate tensors
            tensors = []
            allocation_sizes_mb = [100, 200, 500]  # MB
            
            for size_mb in allocation_sizes_mb:
                num_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
                tensor = torch.randn(num_elements, device=device)
                tensors.append(tensor)
                
                current_memory = torch.cuda.memory_allocated(0)
                print(f"   ✅ Allocated {size_mb}MB → Total: {current_memory / 1024**2:.2f} MB")
            
            # Check peak memory
            peak_memory = torch.cuda.max_memory_allocated(0)
            print(f"   Peak memory usage: {peak_memory / 1024**2:.2f} MB")
            
            # Free memory
            del tensors
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated(0)
            print(f"   Final memory: {final_memory / 1024**2:.2f} MB")
            
            # Verify memory was freed
            memory_freed = (peak_memory - final_memory) / 1024**2
            if memory_freed > 700:  # Should have freed ~800MB
                cuda_functional_results["memory_test_passed"] = True
                cuda_functional_results["tests_run"].append("memory_management")
                print(f"   ✅ Memory freed: {memory_freed:.2f} MB")
                print(f"   Status: PASSED")
            else:
                cuda_functional_results["errors"].append("Memory not properly freed")
                print(f"   ⚠️  Only {memory_freed:.2f} MB freed (expected ~800MB)")
                
        except Exception as e:
            cuda_functional_results["errors"].append(f"Memory test failed: {str(e)}")
            print(f"   ❌ Memory test failed: {str(e)}")
        
        # ========================================================================
        # TEST 4: CUDA Stream Operations
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 4: CUDA Stream Synchronization")
        print("─" * 80)
        
        try:
            # Create multiple CUDA streams
            num_streams = 4
            streams = [torch.cuda.Stream() for _ in range(num_streams)]
            
            print(f"   Created {num_streams} CUDA streams")
            
            # Launch operations on different streams
            results = []
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    A = torch.randn(2000, 2000, device=device)
                    B = torch.randn(2000, 2000, device=device)
                    C = torch.matmul(A, B)
                    results.append(C)
            
            # Synchronize all streams
            start_time = time.time()
            for stream in streams:
                stream.synchronize()
            sync_time = time.time() - start_time
            
            print(f"   ✅ {num_streams} concurrent operations completed")
            print(f"   ✅ Stream synchronization: {sync_time*1000:.2f}ms")
            
            cuda_functional_results["tests_run"].append("cuda_streams")
            print(f"   Status: PASSED")
            
            # Cleanup
            del results
            
        except Exception as e:
            cuda_functional_results["errors"].append(f"Stream operations failed: {str(e)}")
            print(f"   ❌ Stream operations failed: {str(e)}")
        
        # ========================================================================
        # TEST 5: Mixed Precision Support (FP16, BF16, TF32)
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 5: Mixed Precision Support")
        print("─" * 80)
        
        try:
            # Test FP16 (float16)
            print("   Testing float16 (FP16)...")
            try:
                A_fp16 = torch.randn(1000, 1000, device=device, dtype=torch.float16)
                B_fp16 = torch.randn(1000, 1000, device=device, dtype=torch.float16)
                C_fp16 = torch.matmul(A_fp16, B_fp16)
                torch.cuda.synchronize()
                
                cuda_functional_results["mixed_precision_support"]["float16"] = True
                print(f"      ✅ float16 (FP16): Supported")
                
                del A_fp16, B_fp16, C_fp16
            except Exception as e:
                print(f"      ❌ float16 (FP16): Not supported - {str(e)}")
            
            # Test BF16 (bfloat16) - requires Ampere or newer
            print("   Testing bfloat16 (BF16)...")
            try:
                if gpu_capability[0] >= 8:  # Ampere (A100, etc.)
                    A_bf16 = torch.randn(1000, 1000, device=device, dtype=torch.bfloat16)
                    B_bf16 = torch.randn(1000, 1000, device=device, dtype=torch.bfloat16)
                    C_bf16 = torch.matmul(A_bf16, B_bf16)
                    torch.cuda.synchronize()
                    
                    cuda_functional_results["mixed_precision_support"]["bfloat16"] = True
                    print(f"      ✅ bfloat16 (BF16): Supported")
                    
                    del A_bf16, B_bf16, C_bf16
                else:
                    print(f"      ⚠️  bfloat16 (BF16): Requires Ampere+ GPU (compute capability 8.0+)")
                    print(f"         Your GPU: {gpu_capability[0]}.{gpu_capability[1]}")
            except Exception as e:
                print(f"      ❌ bfloat16 (BF16): Not supported - {str(e)}")
            
            # Test TF32 (available on Ampere+)
            print("   Testing TensorFloat-32 (TF32)...")
            try:
                if gpu_capability[0] >= 8:
                    # TF32 is enabled by default on Ampere
                    tf32_enabled = torch.backends.cuda.matmul.allow_tf32
                    cuda_functional_results["mixed_precision_support"]["tf32"] = tf32_enabled
                    
                    if tf32_enabled:
                        print(f"      ✅ TensorFloat-32 (TF32): Enabled")
                    else:
                        print(f"      ⚠️  TensorFloat-32 (TF32): Available but disabled")
                else:
                    print(f"      ⚠️  TensorFloat-32 (TF32): Requires Ampere+ GPU")
            except Exception as e:
                print(f"      ❌ TensorFloat-32 (TF32): Check failed - {str(e)}")
            
            cuda_functional_results["tests_run"].append("mixed_precision")
            print(f"   Status: PASSED")
            
        except Exception as e:
            cuda_functional_results["errors"].append(f"Mixed precision test failed: {str(e)}")
            print(f"   ❌ Mixed precision test failed: {str(e)}")
        
        # ========================================================================
        # TEST 6: cuDNN Availability and Version
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 6: cuDNN Availability")
        print("─" * 80)
        
        try:
            if torch.backends.cudnn.is_available():
                cuda_functional_results["cudnn_available"] = True
                cuda_functional_results["cudnn_version"] = torch.backends.cudnn.version()
                
                print(f"   ✅ cuDNN available")
                print(f"   ✅ cuDNN version: {torch.backends.cudnn.version()}")
                print(f"   ✅ cuDNN enabled: {torch.backends.cudnn.enabled}")
                
                cuda_functional_results["tests_run"].append("cudnn")
                print(f"   Status: PASSED")
            else:
                cuda_functional_results["errors"].append("cuDNN not available")
                print(f"   ❌ cuDNN not available")
                
        except Exception as e:
            cuda_functional_results["errors"].append(f"cuDNN check failed: {str(e)}")
            print(f"   ❌ cuDNN check failed: {str(e)}")
        
        # ========================================================================
        # TEST 7: NCCL Availability (for distributed training)
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 7: NCCL Availability (Distributed Training)")
        print("─" * 80)
        
        try:
            # Check if NCCL is available
            if torch.cuda.nccl.is_available():
                cuda_functional_results["nccl_available"] = True
                
                # Try to get NCCL version
                try:
                    nccl_version = torch.cuda.nccl.version()
                    cuda_functional_results["nccl_version"] = nccl_version
                    
                    print(f"   ✅ NCCL available")
                    print(f"   ✅ NCCL version: {nccl_version}")
                except:
                    print(f"   ✅ NCCL available (version unknown)")
                
                cuda_functional_results["tests_run"].append("nccl")
                print(f"   Status: PASSED")
            else:
                print(f"   ⚠️  NCCL not available")
                print(f"   ℹ️  NCCL is required for multi-GPU distributed training")
                
        except Exception as e:
            # NCCL not being available is not critical for single-GPU
            print(f"   ⚠️  NCCL check failed: {str(e)}")
            print(f"   ℹ️  Single-GPU training will still work")
        
        # ========================================================================
        # TEST 8: cuBLAS Performance Benchmark (GEMM)
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 8: cuBLAS Performance (GEMM Benchmark)")
        print("─" * 80)
        
        try:
            # cuBLAS GEMM (General Matrix Multiply) benchmark
            # This tests the cuBLAS library specifically, not just PyTorch
            import torch.nn.functional as F
            
            matrix_sizes = [1024, 2048, 4096, 8192]
            cublas_results = []
            
            print(f"   Running cuBLAS GEMM benchmarks...")
            
            for size in matrix_sizes:
                A = torch.randn(size, size, device=device)
                B = torch.randn(size, size, device=device)
                
                # Warm-up
                for _ in range(3):
                    C = torch.mm(A, B)
                torch.cuda.synchronize()
                
                # Benchmark
                iterations = 10
                start_time = time.time()
                for _ in range(iterations):
                    C = torch.mm(A, B)  # Uses cuBLAS sgemm
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                
                # Calculate GFLOPS
                flops = 2 * (size ** 3) * iterations
                gflops = (flops / elapsed) / 1e9
                cublas_results.append((size, gflops))
                
                print(f"      {size}×{size}: {gflops:.2f} GFLOPS")
                
                del A, B, C
            
            # Store best result
            if 'cublas_gflops' not in cuda_functional_results:
                cuda_functional_results['cublas_gflops'] = {}
            for size, gflops in cublas_results:
                cuda_functional_results['cublas_gflops'][f'{size}x{size}'] = round(gflops, 2)
            
            cuda_functional_results["tests_run"].append("cublas_gemm")
            print(f"   ✅ cuBLAS GEMM benchmark complete")
            print(f"   Status: PASSED")
            
        except Exception as e:
            cuda_functional_results["errors"].append(f"cuBLAS benchmark failed: {str(e)}")
            print(f"   ❌ cuBLAS benchmark failed: {str(e)}")
        
        # ========================================================================
        # TEST 9: cuFFT Performance (Fast Fourier Transform)
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 9: cuFFT Performance (FFT Benchmark)")
        print("─" * 80)
        
        try:
            # Test 1D, 2D FFT operations
            print(f"   Testing 1D FFT...")
            
            # 1D FFT
            signal_length = 2**20  # 1M points
            signal = torch.randn(signal_length, device=device, dtype=torch.complex64)
            
            # Warm-up
            for _ in range(3):
                fft_result = torch.fft.fft(signal)
            torch.cuda.synchronize()
            
            # Benchmark
            iterations = 20
            start_time = time.time()
            for _ in range(iterations):
                fft_result = torch.fft.fft(signal)
            torch.cuda.synchronize()
            elapsed_1d = time.time() - start_time
            
            time_per_fft_1d = (elapsed_1d / iterations) * 1000
            print(f"      1D FFT ({signal_length} points): {time_per_fft_1d:.2f}ms/iter")
            
            # 2D FFT
            print(f"   Testing 2D FFT...")
            image_size = 2048
            image = torch.randn(image_size, image_size, device=device, dtype=torch.complex64)
            
            # Warm-up
            for _ in range(3):
                fft_result = torch.fft.fft2(image)
            torch.cuda.synchronize()
            
            # Benchmark
            iterations = 10
            start_time = time.time()
            for _ in range(iterations):
                fft_result = torch.fft.fft2(image)
            torch.cuda.synchronize()
            elapsed_2d = time.time() - start_time
            
            time_per_fft_2d = (elapsed_2d / iterations) * 1000
            print(f"      2D FFT ({image_size}×{image_size}): {time_per_fft_2d:.2f}ms/iter")
            
            # Store results
            cuda_functional_results['cufft_performance'] = {
                '1d_fft_ms': round(time_per_fft_1d, 2),
                '2d_fft_ms': round(time_per_fft_2d, 2)
            }
            
            cuda_functional_results["tests_run"].append("cufft")
            print(f"   ✅ cuFFT benchmark complete")
            print(f"   Status: PASSED")
            
            del signal, image, fft_result
            
        except Exception as e:
            cuda_functional_results["errors"].append(f"cuFFT benchmark failed: {str(e)}")
            print(f"   ❌ cuFFT benchmark failed: {str(e)}")
        
        # ========================================================================
        # TEST 10: cuSOLVER (Linear Algebra Operations)
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 10: cuSOLVER Performance (Linear Algebra)")
        print("─" * 80)
        
        try:
            # Test matrix inversion and eigenvalue decomposition
            matrix_sizes = [512, 1024, 2048]
            cusolver_results = []
            
            print(f"   Testing matrix inversion...")
            
            for size in matrix_sizes:
                # Create random positive definite matrix
                A = torch.randn(size, size, device=device)
                A = torch.mm(A, A.t()) + torch.eye(size, device=device) * 0.1
                
                # Warm-up
                for _ in range(2):
                    A_inv = torch.inverse(A)
                torch.cuda.synchronize()
                
                # Benchmark
                iterations = 5
                start_time = time.time()
                for _ in range(iterations):
                    A_inv = torch.inverse(A)
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                
                time_per_inv = (elapsed / iterations) * 1000
                cusolver_results.append((size, time_per_inv))
                
                print(f"      {size}×{size} inversion: {time_per_inv:.2f}ms")
                
                del A, A_inv
            
            # Store results
            cuda_functional_results['cusolver_performance'] = {
                f'{size}x{size}_inverse_ms': round(time_ms, 2)
                for size, time_ms in cusolver_results
            }
            
            cuda_functional_results["tests_run"].append("cusolver")
            print(f"   ✅ cuSOLVER benchmark complete")
            print(f"   Status: PASSED")
            
        except Exception as e:
            cuda_functional_results["errors"].append(f"cuSOLVER benchmark failed: {str(e)}")
            print(f"   ❌ cuSOLVER benchmark failed: {str(e)}")
        
        # ========================================================================
        # TEST 11: Tensor Cores Performance (if available)
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 11: Tensor Cores Performance (Ampere+ GPUs)")
        print("─" * 80)
        
        try:
            # Tensor Cores available on compute capability 7.0+ (Volta, Turing, Ampere)
            if gpu_capability[0] >= 7:
                print(f"   GPU supports Tensor Cores (compute {gpu_capability[0]}.{gpu_capability[1]})")
                
                # Test with FP16 which uses Tensor Cores
                size = 8192
                iterations = 20
                
                # FP16 matmul (uses Tensor Cores)
                A_fp16 = torch.randn(size, size, device=device, dtype=torch.float16)
                B_fp16 = torch.randn(size, size, device=device, dtype=torch.float16)
                
                # Warm-up
                for _ in range(5):
                    C_fp16 = torch.matmul(A_fp16, B_fp16)
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(iterations):
                    C_fp16 = torch.matmul(A_fp16, B_fp16)
                torch.cuda.synchronize()
                elapsed_fp16 = time.time() - start_time
                
                # Calculate GFLOPS
                flops = 2 * (size ** 3) * iterations
                gflops_fp16 = (flops / elapsed_fp16) / 1e9
                
                print(f"   ✅ FP16 Tensor Cores: {gflops_fp16:.2f} GFLOPS")
                
                # Compare with FP32 (no Tensor Cores)
                A_fp32 = torch.randn(size, size, device=device, dtype=torch.float32)
                B_fp32 = torch.randn(size, size, device=device, dtype=torch.float32)
                
                # Warm-up
                for _ in range(5):
                    C_fp32 = torch.matmul(A_fp32, B_fp32)
                torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(iterations):
                    C_fp32 = torch.matmul(A_fp32, B_fp32)
                torch.cuda.synchronize()
                elapsed_fp32 = time.time() - start_time
                
                gflops_fp32 = (flops / elapsed_fp32) / 1e9
                
                print(f"   ✅ FP32 CUDA Cores: {gflops_fp32:.2f} GFLOPS")
                
                speedup = gflops_fp16 / gflops_fp32
                print(f"   ✅ Tensor Core speedup: {speedup:.2f}x")
                
                cuda_functional_results['tensor_cores_performance'] = {
                    'fp16_gflops': round(gflops_fp16, 2),
                    'fp32_gflops': round(gflops_fp32, 2),
                    'speedup': round(speedup, 2)
                }
                
                cuda_functional_results["tests_run"].append("tensor_cores")
                print(f"   Status: PASSED")
                
                del A_fp16, B_fp16, C_fp16, A_fp32, B_fp32, C_fp32
                
            else:
                print(f"   ⚠️  GPU does not support Tensor Cores (compute {gpu_capability[0]}.{gpu_capability[1]} < 7.0)")
                print(f"   ℹ️  Tensor Cores available on Volta (V100), Turing (T4), Ampere (A100) and newer")
                
        except Exception as e:
            cuda_functional_results["errors"].append(f"Tensor Cores benchmark failed: {str(e)}")
            print(f"   ❌ Tensor Cores benchmark failed: {str(e)}")
        
        # ========================================================================
        # TEST 12: Memory Bandwidth Test
        # ========================================================================
        print("\n" + "─" * 80)
        print("TEST 12: GPU Memory Bandwidth")
        print("─" * 80)
        
        try:
            # Test memory bandwidth with large tensor copies
            size_mb = 512  # 512MB tensor
            num_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
            iterations = 20
            
            print(f"   Testing memory bandwidth with {size_mb}MB transfers...")
            
            # Device to Device copy
            A = torch.randn(num_elements, device=device)
            B = torch.empty_like(A)
            
            # Warm-up
            for _ in range(5):
                B.copy_(A)
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                B.copy_(A)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            # Calculate bandwidth in GB/s
            bytes_transferred = size_mb * 1024 * 1024 * iterations
            bandwidth_gb_s = (bytes_transferred / elapsed) / 1e9
            
            print(f"   ✅ Memory bandwidth: {bandwidth_gb_s:.2f} GB/s")
            
            # Get GPU memory bandwidth spec (theoretical)
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"   ℹ️  GPU: {gpu_props.name}")
            print(f"   ℹ️  Total memory: {gpu_props.total_memory / 1024**3:.2f} GB")
            
            cuda_functional_results['memory_bandwidth_gb_s'] = round(bandwidth_gb_s, 2)
            cuda_functional_results["tests_run"].append("memory_bandwidth")
            print(f"   Status: PASSED")
            
            del A, B
            
        except Exception as e:
            cuda_functional_results["errors"].append(f"Memory bandwidth test failed: {str(e)}")
            print(f"   ❌ Memory bandwidth test failed: {str(e)}")
        
        # Determine overall functional status
        if len(cuda_functional_results["tests_run"]) >= 8:
            cuda_functional_results["cuda_functional"] = True
            cuda_functional_results["status"] = "PASSED"
        elif len(cuda_functional_results["tests_run"]) >= 5:
            cuda_functional_results["status"] = "PARTIAL"
        else:
            cuda_functional_results["status"] = "FAILED"

except ImportError:
    cuda_functional_results["errors"].append("PyTorch not installed")
    cuda_functional_results["status"] = "BLOCKED"
    print("\n❌ PyTorch not installed - cannot run CUDA functional tests")
except Exception as e:
    cuda_functional_results["errors"].append(f"Unexpected error: {str(e)}")
    cuda_functional_results["status"] = "ERROR"
    print(f"\n❌ Unexpected error: {str(e)}")

# Final summary
print("\n" + "=" * 80)
print(f"CUDA FUNCTIONAL TEST STATUS: {cuda_functional_results['status']}")
print("=" * 80)

if cuda_functional_results["status"] == "PASSED":
    print(f"\n✅ ALL FUNCTIONAL TESTS PASSED")
    print(f"   Tests run: {len(cuda_functional_results['tests_run'])}")
    print(f"   GFLOPS: {cuda_functional_results['tensor_ops_speed_gflops']}")
    print(f"   Memory test: {'PASSED' if cuda_functional_results['memory_test_passed'] else 'FAILED'}")
    print(f"   Mixed precision: {sum(cuda_functional_results['mixed_precision_support'].values())} types supported")
elif cuda_functional_results["status"] == "PARTIAL":
    print(f"\n⚠️  PARTIAL SUCCESS")
    print(f"   Tests run: {len(cuda_functional_results['tests_run'])}")
    print(f"   Some tests passed, but not all features available")
elif cuda_functional_results["errors"]:
    print(f"\n❌ ERRORS DETECTED:")
    for i, error in enumerate(cuda_functional_results["errors"], 1):
        print(f"   {i}. {error}")

print("=" * 80)

# Display results as DataFrame
summary_data = {
    "Test": [
        "Tensor Creation",
        "Matrix Multiplication",
        "Memory Management",
        "CUDA Streams",
        "Mixed Precision",
        "cuDNN",
        "NCCL (Distributed)",
        "cuBLAS GEMM",
        "cuFFT",
        "cuSOLVER",
        "Tensor Cores",
        "Memory Bandwidth"
    ],
    "Status": [
        "✅ PASS" if "tensor_creation" in cuda_functional_results["tests_run"] else "❌ FAIL",
        f"✅ PASS ({cuda_functional_results['tensor_ops_speed_gflops']} GFLOPS)" if cuda_functional_results['tensor_ops_speed_gflops'] > 0 else "❌ FAIL",
        "✅ PASS" if cuda_functional_results["memory_test_passed"] else "❌ FAIL",
        "✅ PASS" if "cuda_streams" in cuda_functional_results["tests_run"] else "❌ FAIL",
        f"✅ PASS ({sum(cuda_functional_results['mixed_precision_support'].values())}/3)" if "mixed_precision" in cuda_functional_results["tests_run"] else "❌ FAIL",
        f"✅ PASS (v{cuda_functional_results['cudnn_version']})" if cuda_functional_results["cudnn_available"] else "❌ FAIL",
        f"✅ PASS (v{cuda_functional_results['nccl_version']})" if cuda_functional_results["nccl_available"] else "⚠️ N/A",
        f"✅ PASS (8192: {cuda_functional_results.get('cublas_gflops', {}).get('8192x8192', 0)} GFLOPS)" if "cublas_gemm" in cuda_functional_results["tests_run"] else "⚠️ N/A",
        f"✅ PASS ({cuda_functional_results.get('cufft_performance', {}).get('2d_fft_ms', 0)}ms 2D)" if "cufft" in cuda_functional_results["tests_run"] else "⚠️ N/A",
        f"✅ PASS (2048: {cuda_functional_results.get('cusolver_performance', {}).get('2048x2048_inverse_ms', 0)}ms)" if "cusolver" in cuda_functional_results["tests_run"] else "⚠️ N/A",
        f"✅ PASS ({cuda_functional_results.get('tensor_cores_performance', {}).get('speedup', 0)}x speedup)" if "tensor_cores" in cuda_functional_results["tests_run"] else "⚠️ N/A",
        f"✅ PASS ({cuda_functional_results.get('memory_bandwidth_gb_s', 0)} GB/s)" if "memory_bandwidth" in cuda_functional_results["tests_run"] else "⚠️ N/A"
    ],
    "Details": [
        "Tensor creation on GPU device",
        f"4096×4096 matmul performance",
        "Allocate/free 800MB GPU memory",
        "4 concurrent CUDA streams",
        f"FP16: {cuda_functional_results['mixed_precision_support']['float16']}, BF16: {cuda_functional_results['mixed_precision_support']['bfloat16']}, TF32: {cuda_functional_results['mixed_precision_support']['tf32']}",
        "Deep learning primitives library",
        "Multi-GPU communication (optional)",
        "General Matrix Multiply benchmark",
        "Fast Fourier Transform performance",
        "Linear algebra operations",
        "FP16 vs FP32 performance (Volta+)",
        "Device-to-device copy speed"
    ]
}

df_cuda_functional = pd.DataFrame(summary_data)
display(df_cuda_functional)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🔗 Cell 5: BioNeMo Dependency Stack Validation (NEW!)
# MAGIC
# MAGIC Comprehensive validation of the complete BioNeMo dependency stack for pip installation.
# MAGIC This cell tests the entire dependency chain with automatic installation and version compatibility.
# MAGIC
# MAGIC **Import Chain Testing:**
# MAGIC - Attempts to import bionemo.core (auto-installs if missing)
# MAGIC - Validates bionemo.core.utils.dtype functionality
# MAGIC - Tests get_autocast_dtype('bfloat16') function
# MAGIC - Attempts optional model imports (llm, esm2, evo2) - non-fatal
# MAGIC
# MAGIC **Dependency Chain Validation:**
# MAGIC - NeMo Toolkit installation and version (>= 1.22.0)
# MAGIC - Megatron-Core availability
# MAGIC - PyTorch Lightning version constraint (< 2.5.0)
# MAGIC - PyTorch Lightning GPU strategy initialization
# MAGIC
# MAGIC **PyTorch Integration:**
# MAGIC - Verify torch version >= 2.2.0
# MAGIC - Validate torch.cuda functional testing
# MAGIC - Test PyTorch Lightning Trainer with GPU
# MAGIC - Validate FSDP support for distributed training
# MAGIC
# MAGIC **Autocast Support:**
# MAGIC - Test torch.autocast context manager
# MAGIC - Validate bfloat16 support in autocast
# MAGIC - Test mixed precision type conversions
# MAGIC
# MAGIC **Critical for BioNeMo:**
# MAGIC - BioNeMo requires specific version combinations
# MAGIC - Dependency conflicts can cause silent failures
# MAGIC - Autocast support is essential for mixed precision training
# MAGIC - FSDP enables distributed training on large models

# COMMAND ----------
print("=" * 80)
print("🔗 BIONEMO DEPENDENCY STACK VALIDATION")
print("=" * 80)

# Initialize comprehensive results dictionary
dependency_validation_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "bionemo_core_installed": False,
    "optional_models_available": {
        "llm": False,
        "esm2": False,
        "evo2": False
    },
    "dependency_versions": {
        "nemo": None,
        "megatron": None,
        "lightning": None,
        "torch": None
    },
    "version_compatibility": {
        "nemo_version": "UNKNOWN",
        "lightning_version": "UNKNOWN",
        "torch_version": "UNKNOWN",
        "all_compatible": False
    },
    "autocast_functional": False,
    "distributed_ready": False,
    "critical_errors": [],
    "warnings": [],
    "installation_commands": [],
    "status": "PENDING"
}

try:
    import torch
    import subprocess
    import importlib
    from packaging import version
    
    # ========================================================================
    # SECTION 1: IMPORT CHAIN TESTING
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: IMPORT CHAIN TESTING")
    print("=" * 80)
    
    # Test 1.1: Import bionemo.core (auto-install if missing)
    print("\n📦 Test 1.1: bionemo.core Import")
    print("─" * 80)
    
    try:
        import bionemo.core
        dependency_validation_results["bionemo_core_installed"] = True
        print(f"   ✅ bionemo.core imported successfully")
        
        # Get version if available
        if hasattr(bionemo.core, '__version__'):
            core_version = bionemo.core.__version__
            print(f"   ℹ️  Version: {core_version}")
        
    except ImportError as e:
        print(f"   ⚠️  bionemo.core not found: {str(e)}")
        print(f"   🔧 Installing bionemo-core...")
        
        try:
            # Attempt installation
            install_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "bionemo-core"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if install_result.returncode == 0:
                # Try importing again
                import bionemo.core
                dependency_validation_results["bionemo_core_installed"] = True
                print(f"   ✅ bionemo-core installed and imported successfully")
            else:
                dependency_validation_results["critical_errors"].append(
                    f"Failed to install bionemo-core: {install_result.stderr}"
                )
                print(f"   ❌ Installation failed: {install_result.stderr}")
                
        except subprocess.TimeoutExpired:
            dependency_validation_results["critical_errors"].append("bionemo-core installation timed out")
            print(f"   ❌ Installation timed out (>300s)")
        except Exception as install_error:
            dependency_validation_results["critical_errors"].append(f"Installation error: {str(install_error)}")
            print(f"   ❌ Installation error: {str(install_error)}")
    
    # Test 1.2: Validate bionemo.core.utils.dtype functionality
    if dependency_validation_results["bionemo_core_installed"]:
        print("\n📦 Test 1.2: bionemo.core.utils.dtype Functionality")
        print("─" * 80)
        
        try:
            from bionemo.core.utils import dtype as bionemo_dtype
            print(f"   ✅ bionemo.core.utils.dtype imported")
            
            # Test 1.3: Test get_autocast_dtype function
            print("\n📦 Test 1.3: get_autocast_dtype('bfloat16') Function")
            print("─" * 80)
            
            try:
                # Test get_autocast_dtype with bfloat16
                autocast_dtype = bionemo_dtype.get_autocast_dtype('bfloat16')
                print(f"   ✅ get_autocast_dtype('bfloat16') returned: {autocast_dtype}")
                
                # Verify it returns a PyTorch dtype
                if hasattr(torch, str(autocast_dtype).split('.')[-1]):
                    print(f"   ✅ Returned dtype is valid PyTorch dtype")
                else:
                    dependency_validation_results["warnings"].append(
                        "get_autocast_dtype returned unexpected dtype"
                    )
                    print(f"   ⚠️  Unexpected dtype returned")
                    
            except Exception as e:
                dependency_validation_results["warnings"].append(
                    f"get_autocast_dtype failed: {str(e)}"
                )
                print(f"   ⚠️  get_autocast_dtype test failed: {str(e)}")
                
        except ImportError as e:
            dependency_validation_results["warnings"].append(
                f"bionemo.core.utils.dtype not available: {str(e)}"
            )
            print(f"   ⚠️  Cannot import bionemo.core.utils.dtype: {str(e)}")
    
    # Test 1.4: Optional model imports (non-fatal)
    print("\n📦 Test 1.4: Optional BioNeMo Models (Non-Fatal)")
    print("─" * 80)
    
    optional_models = {
        "llm": "bionemo_llm",
        "esm2": "bionemo_esm2",
        "evo2": "bionemo_evo2"
    }
    
    for model_key, model_module in optional_models.items():
        try:
            importlib.import_module(model_module)
            dependency_validation_results["optional_models_available"][model_key] = True
            print(f"   ✅ {model_module} available")
        except ImportError:
            print(f"   ℹ️  {model_module} not installed (optional)")
    
    # ========================================================================
    # SECTION 2: DEPENDENCY CHAIN VALIDATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: DEPENDENCY CHAIN VALIDATION")
    print("=" * 80)
    
    # Test 2.1: NeMo Toolkit
    print("\n📦 Test 2.1: NeMo Toolkit")
    print("─" * 80)
    
    try:
        import nemo
        nemo_version = nemo.__version__
        dependency_validation_results["dependency_versions"]["nemo"] = nemo_version
        
        print(f"   ✅ NeMo Toolkit installed: v{nemo_version}")
        
        # Validate version >= 1.22.0
        try:
            if version.parse(nemo_version) >= version.parse("1.22.0"):
                dependency_validation_results["version_compatibility"]["nemo_version"] = "PASS"
                print(f"   ✅ Version check: {nemo_version} >= 1.22.0")
            else:
                dependency_validation_results["version_compatibility"]["nemo_version"] = "FAIL"
                dependency_validation_results["critical_errors"].append(
                    f"NeMo version {nemo_version} < 1.22.0 (required)"
                )
                print(f"   ❌ Version check: {nemo_version} < 1.22.0 (required)")
                dependency_validation_results["installation_commands"].append(
                    "pip install 'nemo-toolkit[all]>=1.22.0'"
                )
        except Exception as e:
            dependency_validation_results["warnings"].append(f"NeMo version parse error: {str(e)}")
            print(f"   ⚠️  Could not parse NeMo version: {str(e)}")
            
    except ImportError:
        dependency_validation_results["critical_errors"].append("NeMo Toolkit not installed")
        dependency_validation_results["version_compatibility"]["nemo_version"] = "NOT_INSTALLED"
        print(f"   ❌ NeMo Toolkit not installed")
        dependency_validation_results["installation_commands"].append(
            "pip install 'nemo-toolkit[all]>=1.22.0'"
        )
    
    # Test 2.2: Megatron-Core
    print("\n📦 Test 2.2: Megatron-Core")
    print("─" * 80)
    
    try:
        import megatron
        # Try to get version from megatron.core
        megatron_version = "Unknown"
        if hasattr(megatron, '__version__'):
            megatron_version = megatron.__version__
        elif hasattr(megatron, 'core') and hasattr(megatron.core, '__version__'):
            megatron_version = megatron.core.__version__
        
        dependency_validation_results["dependency_versions"]["megatron"] = megatron_version
        print(f"   ✅ Megatron-Core installed: v{megatron_version}")
        
    except ImportError:
        dependency_validation_results["warnings"].append("Megatron-Core not installed")
        print(f"   ⚠️  Megatron-Core not installed (may be bundled with NeMo)")
    
    # Test 2.3: PyTorch Lightning version constraint
    print("\n📦 Test 2.3: PyTorch Lightning Version Constraint")
    print("─" * 80)
    
    try:
        import pytorch_lightning as pl
        lightning_version = pl.__version__
        dependency_validation_results["dependency_versions"]["lightning"] = lightning_version
        
        print(f"   ✅ PyTorch Lightning installed: v{lightning_version}")
        
        # Validate version < 2.5.0
        try:
            if version.parse(lightning_version) < version.parse("2.5.0"):
                dependency_validation_results["version_compatibility"]["lightning_version"] = "PASS"
                print(f"   ✅ Version check: {lightning_version} < 2.5.0")
            else:
                dependency_validation_results["version_compatibility"]["lightning_version"] = "WARN"
                dependency_validation_results["warnings"].append(
                    f"PyTorch Lightning {lightning_version} >= 2.5.0 (may have compatibility issues)"
                )
                print(f"   ⚠️  Version check: {lightning_version} >= 2.5.0 (may cause issues)")
                dependency_validation_results["installation_commands"].append(
                    "pip install 'pytorch-lightning<2.5.0'"
                )
        except Exception as e:
            dependency_validation_results["warnings"].append(f"Lightning version parse error: {str(e)}")
            print(f"   ⚠️  Could not parse Lightning version: {str(e)}")
            
    except ImportError:
        dependency_validation_results["critical_errors"].append("PyTorch Lightning not installed")
        dependency_validation_results["version_compatibility"]["lightning_version"] = "NOT_INSTALLED"
        print(f"   ❌ PyTorch Lightning not installed")
        dependency_validation_results["installation_commands"].append(
            "pip install 'pytorch-lightning<2.5.0'"
        )
    
    # Test 2.4: PyTorch Lightning GPU Strategy
    print("\n📦 Test 2.4: PyTorch Lightning GPU Strategy Initialization")
    print("─" * 80)
    
    if dependency_validation_results["dependency_versions"]["lightning"]:
        try:
            from pytorch_lightning.strategies import DDPStrategy
            
            # Test strategy initialization (doesn't require GPU to instantiate)
            strategy = DDPStrategy(find_unused_parameters=True)
            print(f"   ✅ DDPStrategy initialized successfully")
            
            # Check for FSDP strategy (for distributed training)
            try:
                from pytorch_lightning.strategies import FSDPStrategy
                print(f"   ✅ FSDPStrategy available (distributed training ready)")
                dependency_validation_results["distributed_ready"] = True
            except ImportError:
                dependency_validation_results["warnings"].append("FSDP strategy not available")
                print(f"   ⚠️  FSDP strategy not available (Lightning >= 1.9.0 required)")
                
        except Exception as e:
            dependency_validation_results["warnings"].append(f"Lightning strategy test failed: {str(e)}")
            print(f"   ⚠️  Strategy initialization failed: {str(e)}")
    
    # ========================================================================
    # SECTION 3: PYTORCH INTEGRATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: PYTORCH INTEGRATION")
    print("=" * 80)
    
    # Test 3.1: PyTorch version
    print("\n📦 Test 3.1: PyTorch Version >= 2.2.0")
    print("─" * 80)
    
    torch_version = torch.__version__.split('+')[0]  # Remove +cuXXX suffix
    dependency_validation_results["dependency_versions"]["torch"] = torch_version
    
    print(f"   ✅ PyTorch installed: v{torch_version}")
    
    try:
        if version.parse(torch_version) >= version.parse("2.2.0"):
            dependency_validation_results["version_compatibility"]["torch_version"] = "PASS"
            print(f"   ✅ Version check: {torch_version} >= 2.2.0")
        else:
            dependency_validation_results["version_compatibility"]["torch_version"] = "FAIL"
            dependency_validation_results["critical_errors"].append(
                f"PyTorch version {torch_version} < 2.2.0 (required)"
            )
            print(f"   ❌ Version check: {torch_version} < 2.2.0 (required)")
            dependency_validation_results["installation_commands"].append(
                "pip install 'torch>=2.2.0'"
            )
    except Exception as e:
        dependency_validation_results["warnings"].append(f"Torch version parse error: {str(e)}")
        print(f"   ⚠️  Could not parse PyTorch version: {str(e)}")
    
    # Test 3.2: torch.cuda.is_available() AND functional testing
    print("\n📦 Test 3.2: CUDA Availability and Functional Test")
    print("─" * 80)
    
    if torch.cuda.is_available():
        print(f"   ✅ torch.cuda.is_available() = True")
        
        # Perform a simple functional test
        try:
            test_tensor = torch.randn(100, 100, device='cuda')
            result = torch.matmul(test_tensor, test_tensor)
            torch.cuda.synchronize()
            
            print(f"   ✅ CUDA functional test passed (100×100 matmul)")
            del test_tensor, result
            
        except Exception as e:
            dependency_validation_results["warnings"].append(f"CUDA functional test failed: {str(e)}")
            print(f"   ⚠️  CUDA functional test failed: {str(e)}")
    else:
        dependency_validation_results["critical_errors"].append("CUDA not available")
        print(f"   ❌ torch.cuda.is_available() = False")
    
    # Test 3.3: PyTorch Lightning Trainer with GPU
    print("\n📦 Test 3.3: PyTorch Lightning Trainer with GPU")
    print("─" * 80)
    
    if dependency_validation_results["dependency_versions"]["lightning"] and torch.cuda.is_available():
        try:
            import pytorch_lightning as pl
            
            # Create a minimal trainer with GPU
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                max_epochs=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                enable_checkpointing=False
            )
            
            print(f"   ✅ Trainer instantiated with GPU accelerator")
            print(f"      Accelerator: {trainer.accelerator.__class__.__name__}")
            print(f"      Devices: {trainer.num_devices}")
            
        except Exception as e:
            dependency_validation_results["warnings"].append(f"Trainer instantiation failed: {str(e)}")
            print(f"   ⚠️  Trainer instantiation failed: {str(e)}")
    elif not torch.cuda.is_available():
        print(f"   ⏭️  Skipped (CUDA not available)")
    else:
        print(f"   ⏭️  Skipped (PyTorch Lightning not installed)")
    
    # Test 3.4: FSDP support for distributed training
    print("\n📦 Test 3.4: FSDP Support for Distributed Training")
    print("─" * 80)
    
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        print(f"   ✅ torch.distributed.fsdp available")
        
        # Check for ShardingStrategy
        try:
            from torch.distributed.fsdp import ShardingStrategy
            print(f"   ✅ ShardingStrategy available")
            dependency_validation_results["distributed_ready"] = True
        except ImportError:
            dependency_validation_results["warnings"].append("ShardingStrategy not available")
            print(f"   ⚠️  ShardingStrategy not available")
            
    except ImportError:
        dependency_validation_results["warnings"].append("FSDP not available (PyTorch >= 2.0.0 required)")
        print(f"   ⚠️  FSDP not available (requires PyTorch >= 2.0.0)")
    
    # ========================================================================
    # SECTION 4: AUTOCAST SUPPORT
    # ========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: AUTOCAST SUPPORT")
    print("=" * 80)
    
    # Test 4.1: torch.autocast context manager
    print("\n📦 Test 4.1: torch.autocast Context Manager")
    print("─" * 80)
    
    if torch.cuda.is_available():
        try:
            # Test autocast with CUDA
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                A = torch.randn(100, 100, device='cuda')
                B = torch.randn(100, 100, device='cuda')
                C = torch.matmul(A, B)
                
                # Verify dtype
                if C.dtype == torch.float16:
                    print(f"   ✅ torch.autocast working (output dtype: {C.dtype})")
                else:
                    dependency_validation_results["warnings"].append(
                        f"Autocast produced unexpected dtype: {C.dtype}"
                    )
                    print(f"   ⚠️  Unexpected dtype: {C.dtype} (expected float16)")
                
                del A, B, C
                
        except Exception as e:
            dependency_validation_results["warnings"].append(f"torch.autocast test failed: {str(e)}")
            print(f"   ⚠️  torch.autocast test failed: {str(e)}")
    else:
        print(f"   ⏭️  Skipped (CUDA not available)")
    
    # Test 4.2: bfloat16 support in autocast
    print("\n📦 Test 4.2: bfloat16 Support in Autocast")
    print("─" * 80)
    
    if torch.cuda.is_available():
        try:
            # Check GPU compute capability for bfloat16 support
            gpu_capability = torch.cuda.get_device_capability(0)
            
            if gpu_capability[0] >= 8:  # Ampere or newer
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    A = torch.randn(100, 100, device='cuda')
                    B = torch.randn(100, 100, device='cuda')
                    C = torch.matmul(A, B)
                    
                    if C.dtype == torch.bfloat16:
                        print(f"   ✅ bfloat16 autocast working (output dtype: {C.dtype})")
                        dependency_validation_results["autocast_functional"] = True
                    else:
                        dependency_validation_results["warnings"].append(
                            f"bfloat16 autocast produced unexpected dtype: {C.dtype}"
                        )
                        print(f"   ⚠️  Unexpected dtype: {C.dtype} (expected bfloat16)")
                    
                    del A, B, C
            else:
                print(f"   ⚠️  GPU compute capability {gpu_capability[0]}.{gpu_capability[1]} < 8.0")
                print(f"      bfloat16 requires Ampere+ GPU")
                
        except Exception as e:
            dependency_validation_results["warnings"].append(f"bfloat16 autocast test failed: {str(e)}")
            print(f"   ⚠️  bfloat16 autocast test failed: {str(e)}")
    else:
        print(f"   ⏭️  Skipped (CUDA not available)")
    
    # Test 4.3: Mixed precision type conversions
    print("\n📦 Test 4.3: Mixed Precision Type Conversions")
    print("─" * 80)
    
    if torch.cuda.is_available():
        try:
            # Test type conversions
            tensor_fp32 = torch.randn(100, 100, device='cuda', dtype=torch.float32)
            tensor_fp16 = tensor_fp32.half()  # to float16
            tensor_fp32_back = tensor_fp16.float()  # back to float32
            
            print(f"   ✅ FP32 → FP16: {tensor_fp32.dtype} → {tensor_fp16.dtype}")
            print(f"   ✅ FP16 → FP32: {tensor_fp16.dtype} → {tensor_fp32_back.dtype}")
            
            # Test bfloat16 if available
            gpu_capability = torch.cuda.get_device_capability(0)
            if gpu_capability[0] >= 8:
                tensor_bf16 = tensor_fp32.to(torch.bfloat16)
                print(f"   ✅ FP32 → BF16: {tensor_fp32.dtype} → {tensor_bf16.dtype}")
                del tensor_bf16
            
            dependency_validation_results["autocast_functional"] = True
            
            del tensor_fp32, tensor_fp16, tensor_fp32_back
            
        except Exception as e:
            dependency_validation_results["warnings"].append(f"Type conversion test failed: {str(e)}")
            print(f"   ⚠️  Type conversion test failed: {str(e)}")
    else:
        print(f"   ⏭️  Skipped (CUDA not available)")
    
    # ========================================================================
    # DETERMINE OVERALL STATUS
    # ========================================================================
    
    # Check version compatibility
    all_compatible = (
        dependency_validation_results["version_compatibility"]["nemo_version"] == "PASS" and
        dependency_validation_results["version_compatibility"]["lightning_version"] in ["PASS", "WARN"] and
        dependency_validation_results["version_compatibility"]["torch_version"] == "PASS"
    )
    dependency_validation_results["version_compatibility"]["all_compatible"] = all_compatible
    
    # Determine status
    if dependency_validation_results["critical_errors"]:
        dependency_validation_results["status"] = "BLOCKED"
    elif dependency_validation_results["bionemo_core_installed"] and all_compatible:
        dependency_validation_results["status"] = "READY"
    elif dependency_validation_results["bionemo_core_installed"]:
        dependency_validation_results["status"] = "PARTIAL"
    else:
        dependency_validation_results["status"] = "NOT_READY"

except ImportError as e:
    dependency_validation_results["critical_errors"].append(f"Import error: {str(e)}")
    dependency_validation_results["status"] = "BLOCKED"
    print(f"\n❌ Critical import error: {str(e)}")
except Exception as e:
    dependency_validation_results["critical_errors"].append(f"Unexpected error: {str(e)}")
    dependency_validation_results["status"] = "ERROR"
    print(f"\n❌ Unexpected error: {str(e)}")

# Final Summary
print("\n" + "=" * 80)
print(f"DEPENDENCY VALIDATION STATUS: {dependency_validation_results['status']}")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"   BioNeMo Core: {'✅ Installed' if dependency_validation_results['bionemo_core_installed'] else '❌ Not Installed'}")
print(f"   Dependencies: NeMo={dependency_validation_results['dependency_versions']['nemo']}, "
      f"Lightning={dependency_validation_results['dependency_versions']['lightning']}, "
      f"PyTorch={dependency_validation_results['dependency_versions']['torch']}")
print(f"   Version Compatibility: {'✅ All Pass' if dependency_validation_results['version_compatibility']['all_compatible'] else '⚠️ Issues Found'}")
print(f"   Autocast Functional: {'✅ Working' if dependency_validation_results['autocast_functional'] else '⚠️ Not Tested'}")
print(f"   Distributed Ready: {'✅ FSDP Available' if dependency_validation_results['distributed_ready'] else '⚠️ Limited'}")

if dependency_validation_results["critical_errors"]:
    print(f"\n🚨 Critical Errors ({len(dependency_validation_results['critical_errors'])}):")
    for i, error in enumerate(dependency_validation_results["critical_errors"], 1):
        print(f"   {i}. {error}")

if dependency_validation_results["warnings"]:
    print(f"\n⚠️  Warnings ({len(dependency_validation_results['warnings'])}):")
    for i, warning in enumerate(dependency_validation_results["warnings"], 1):
        print(f"   {i}. {warning}")

if dependency_validation_results["installation_commands"]:
    print(f"\n💡 Installation Commands:")
    for i, cmd in enumerate(dependency_validation_results["installation_commands"], 1):
        print(f"   {i}. {cmd}")

print("=" * 80)

# Display results as DataFrame
import pandas as pd

summary_data = {
    "Component": [
        "BioNeMo Core",
        "NeMo Toolkit",
        "Megatron-Core",
        "PyTorch Lightning",
        "PyTorch",
        "Autocast Support",
        "FSDP (Distributed)",
        "Optional Models"
    ],
    "Status": [
        "✅ Installed" if dependency_validation_results["bionemo_core_installed"] else "❌ Not Installed",
        f"✅ v{dependency_validation_results['dependency_versions']['nemo']}" if dependency_validation_results['dependency_versions']['nemo'] else "❌ Not Installed",
        f"✅ v{dependency_validation_results['dependency_versions']['megatron']}" if dependency_validation_results['dependency_versions']['megatron'] else "⚠️ Not Found",
        f"✅ v{dependency_validation_results['dependency_versions']['lightning']}" if dependency_validation_results['dependency_versions']['lightning'] else "❌ Not Installed",
        f"✅ v{dependency_validation_results['dependency_versions']['torch']}" if dependency_validation_results['dependency_versions']['torch'] else "❌ Not Installed",
        "✅ Working" if dependency_validation_results["autocast_functional"] else "⚠️ Not Tested",
        "✅ Available" if dependency_validation_results["distributed_ready"] else "⚠️ Limited",
        f"{sum(dependency_validation_results['optional_models_available'].values())}/3 Available"
    ],
    "Version Check": [
        "N/A",
        dependency_validation_results["version_compatibility"]["nemo_version"],
        "N/A",
        dependency_validation_results["version_compatibility"]["lightning_version"],
        dependency_validation_results["version_compatibility"]["torch_version"],
        "N/A",
        "N/A",
        "N/A"
    ]
}

df_dependency = pd.DataFrame(summary_data)
display(df_dependency)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🧬 Cell 6: BioNeMo Core Package Availability (NEW!)
# MAGIC
# MAGIC Tests availability and imports of BioNeMo Framework packages.
# MAGIC
# MAGIC **Validates:**
# MAGIC - bionemo-core (configuration, testing utilities)
# MAGIC - bionemo-scdl (single cell data loader)
# MAGIC - bionemo-moco (molecular co-design tools)
# MAGIC - bionemo-noodles (fast FASTA I/O)
# MAGIC
# MAGIC **Note:** BioNeMo recipes and 5D parallelism models require Docker images
# MAGIC and are tested separately.

# COMMAND ----------
print("=" * 80)
print("🧬 BIONEMO CORE PACKAGE AVAILABILITY TEST")
print("=" * 80)

# Initialize results dictionary
bionemo_test_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "packages_tested": {},
    "status": "PENDING",
    "blockers": [],
    "warnings": []
}

# Define BioNeMo packages to test
# Reference: https://github.com/NVIDIA/bionemo-framework
bionemo_packages = {
    "bionemo-core": {
        "description": "Model config and test data utilities",
        "pip_name": "bionemo-core",
        "import_name": "bionemo.core",
        "required": False,
        "support": "Active",
        "category": "5D Parallelism"
    },
    "bionemo-scdl": {
        "description": "Modular Single Cell Data Loader",
        "pip_name": "bionemo-scdl",
        "import_name": "bionemo.scdl",
        "required": False,
        "support": "Active",
        "category": "Tooling"
    },
    "bionemo-moco": {
        "description": "Molecular Co-design tools",
        "pip_name": "bionemo-moco",
        "import_name": "bionemo.moco",
        "required": False,
        "support": "Active",
        "category": "Tooling"
    },
    "bionemo-noodles": {
        "description": "Python API to fast FASTA file I/O",
        "pip_name": "bionemo-noodles",
        "import_name": "bionemo.noodles",
        "required": False,
        "support": "Maintenance",
        "category": "Tooling"
    },
    "bionemo-llm": {
        "description": "5D parallel base model (BioBert)",
        "pip_name": "bionemo-llm",
        "import_name": "bionemo.llm",
        "required": False,
        "support": "Active",
        "category": "5D Parallelism"
    },
    "bionemo-evo2": {
        "description": "5D parallel Evo2 model",
        "pip_name": "bionemo-evo2",
        "import_name": "bionemo.evo2",
        "required": False,
        "support": "Active",
        "category": "5D Parallelism"
    },
    "bionemo-geneformer": {
        "description": "5D parallel Geneformer model",
        "pip_name": "bionemo-geneformer",
        "import_name": "bionemo.geneformer",
        "required": False,
        "support": "Maintenance",
        "category": "5D Parallelism"
    }
}

print("\n📦 Testing BioNeMo Package Availability...")
print(f"   Total packages to test: {len(bionemo_packages)}")

# Test each package
for package_name, package_info in bionemo_packages.items():
    print(f"\n{'─' * 80}")
    print(f"Testing: {package_name}")
    print(f"{'─' * 80}")
    
    result = {
        "description": package_info["description"],
        "pip_name": package_info["pip_name"],
        "import_name": package_info["import_name"],
        "category": package_info["category"],
        "support": package_info["support"],
        "is_installed": False,
        "is_importable": False,
        "version": None,
        "import_error": None,
        "test_passed": False
    }
    
    try:
        # 1. Check if package is installed via pip
        print(f"   Checking installation via pip...")
        pip_result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_info["pip_name"]],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if pip_result.returncode == 0:
            result["is_installed"] = True
            # Extract version
            for line in pip_result.stdout.split('\n'):
                if line.startswith('Version:'):
                    result["version"] = line.split(':')[1].strip()
                    break
            print(f"   ✅ Package installed: v{result['version']}")
        else:
            print(f"   ❌ Package not installed")
            print(f"   💡 Install with: pip install {package_info['pip_name']}")
        
        # 2. Try to import the package
        if result["is_installed"]:
            print(f"   Testing import: {package_info['import_name']}...")
            
            try:
                # Attempt import
                spec = importlib.util.find_spec(package_info["import_name"])
                if spec is not None:
                    module = importlib.import_module(package_info["import_name"])
                    result["is_importable"] = True
                    result["test_passed"] = True
                    
                    # Try to get version from module
                    if hasattr(module, '__version__'):
                        result["version"] = module.__version__
                    
                    print(f"   ✅ Import successful")
                    
                    # Check for common submodules
                    submodules = []
                    if hasattr(module, '__all__'):
                        submodules = module.__all__[:5]  # First 5
                        print(f"   📦 Available submodules: {', '.join(submodules)}")
                else:
                    result["import_error"] = "Module spec not found"
                    print(f"   ❌ Import failed: Module spec not found")
                    
            except ImportError as e:
                result["import_error"] = str(e)
                print(f"   ❌ Import failed: {str(e)}")
            except Exception as e:
                result["import_error"] = f"Unexpected error: {str(e)}"
                print(f"   ❌ Unexpected error: {str(e)}")
    
    except subprocess.TimeoutExpired:
        result["import_error"] = "pip show command timed out"
        print(f"   ❌ pip show command timed out")
    except Exception as e:
        result["import_error"] = f"Test error: {str(e)}"
        print(f"   ❌ Test error: {str(e)}")
    
    # Store result
    bionemo_test_results["packages_tested"][package_name] = result
    
    # Track blockers and warnings
    if package_info["required"] and not result["test_passed"]:
        bionemo_test_results["blockers"].append({
            "check": package_name,
            "message": f"Required package {package_name} not available",
            "severity": "BLOCKER"
        })
    elif not result["test_passed"] and result["is_installed"]:
        bionemo_test_results["warnings"].append({
            "check": package_name,
            "message": f"Package {package_name} installed but import failed: {result['import_error']}",
            "severity": "WARNING"
        })

# Calculate statistics
total_tested = len(bionemo_packages)
installed = sum(1 for r in bionemo_test_results["packages_tested"].values() if r["is_installed"])
importable = sum(1 for r in bionemo_test_results["packages_tested"].values() if r["is_importable"])
passed = sum(1 for r in bionemo_test_results["packages_tested"].values() if r["test_passed"])

print("\n" + "=" * 80)
print("📊 BIONEMO PACKAGE TEST SUMMARY")
print("=" * 80)
print(f"   Total Packages Tested: {total_tested}")
print(f"   Installed: {installed}/{total_tested}")
print(f"   Importable: {importable}/{total_tested}")
print(f"   Tests Passed: {passed}/{total_tested}")

# Determine overall status
if bionemo_test_results["blockers"]:
    bionemo_test_results["status"] = "BLOCKED"
    print(f"\n❌ BIONEMO TEST FAILED")
    print(f"   Blockers: {len(bionemo_test_results['blockers'])}")
elif passed == 0:
    bionemo_test_results["status"] = "NO_PACKAGES"
    print(f"\n⚠️  NO BIONEMO PACKAGES INSTALLED")
    print(f"   This is expected if you haven't installed BioNeMo yet")
else:
    bionemo_test_results["status"] = "PASSED"
    print(f"\n✅ BIONEMO TEST PASSED")
    print(f"   {passed}/{total_tested} packages available")

if bionemo_test_results["warnings"]:
    print(f"\n⚠️  Warnings: {len(bionemo_test_results['warnings'])}")
    for warning in bionemo_test_results["warnings"]:
        print(f"   • {warning['message']}")

print("=" * 80)

# Display detailed results as DataFrame
summary_data = []
for package_name, result in bionemo_test_results["packages_tested"].items():
    summary_data.append({
        "Package": package_name,
        "Category": result["category"],
        "Support": result["support"],
        "Installed": "✅" if result["is_installed"] else "❌",
        "Importable": "✅" if result["is_importable"] else ("❌" if result["is_installed"] else "⏭️"),
        "Version": result["version"] if result["version"] else "N/A",
        "Status": "✅ PASS" if result["test_passed"] else ("⚠️ NOT INSTALLED" if not result["is_installed"] else "❌ IMPORT FAILED")
    })

df_bionemo = pd.DataFrame(summary_data)
display(df_bionemo)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📋 Cell 7: Final Summary Report
# MAGIC
# MAGIC Comprehensive summary of all validation checks with recommendations.

# COMMAND ----------
print("=" * 80)
print("📋 BIONEMO FRAMEWORK VALIDATION - FINAL REPORT")
print("=" * 80)

# Aggregate all results
final_report = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "validation_sections": {
        "cuda_environment": cuda_validation_results,
        "pytorch_lightning": lightning_test_results,
        "cuda_functional": cuda_functional_results,
        "dependency_stack": dependency_validation_results,
        "bionemo_packages": bionemo_test_results
    },
    "overall_status": "PENDING",
    "total_blockers": 0,
    "total_warnings": 0,
    "recommendations": []
}

# Count blockers and warnings
for section in final_report["validation_sections"].values():
    final_report["total_blockers"] += len(section.get("blockers", []))
    final_report["total_warnings"] += len(section.get("warnings", []))

# Determine overall status
if final_report["total_blockers"] > 0:
    final_report["overall_status"] = "BLOCKED"
elif bionemo_test_results["status"] == "NO_PACKAGES":
    final_report["overall_status"] = "READY_FOR_INSTALL"
else:
    final_report["overall_status"] = "READY"

# Generate recommendations
print("\n🎯 VALIDATION SUMMARY:")
print("─" * 80)

# Section 1: CUDA Environment
cuda_status = "✅ PASS" if cuda_validation_results["status"] == "PASSED" else "❌ FAIL"
print(f"   1. CUDA Environment: {cuda_status}")
if cuda_validation_results["status"] == "PASSED":
    print(f"      • Runtime: {cuda_validation_results['databricks_runtime']['runtime_version']}")
    print(f"      • GPUs: {cuda_validation_results['gpu_info']['gpu_count']}")
    print(f"      • PyTorch: v{cuda_validation_results['pytorch_info']['version']}")
else:
    print(f"      • Blockers: {len(cuda_validation_results['blockers'])}")

# Section 2: PyTorch Lightning
lightning_status = "✅ PASS" if lightning_test_results["status"] == "PASSED" else "❌ FAIL"
print(f"\n   2. PyTorch Lightning: {lightning_status}")
if lightning_test_results["status"] == "PASSED":
    print(f"      • Version: v{lightning_test_results['version']}")
    print(f"      • GPU Devices: {lightning_test_results['gpu_devices_detected']}")
    print(f"      • Mixed Precision: {'✅ Supported' if lightning_test_results['mixed_precision_support'] else '⚠️ Limited'}")
    if lightning_test_results["benchmark_results"]:
        bench = lightning_test_results["benchmark_results"]
        print(f"      • Throughput: {bench['throughput_iter_per_s']} iter/s")
else:
    print(f"      • Blockers: {len(lightning_test_results['blockers'])}")

# Section 3: CUDA Functional Tests
cuda_func_status = "✅ PASS" if cuda_functional_results["status"] == "PASSED" else ("⏭️ SKIP" if cuda_functional_results["status"] == "SKIPPED" else "❌ FAIL")
print(f"\n   3. CUDA Functional Tests: {cuda_func_status}")
if cuda_functional_results["status"] == "PASSED":
    print(f"      • Tests Run: {len(cuda_functional_results['tests_run'])}")
    print(f"      • Performance: {cuda_functional_results['tensor_ops_speed_gflops']} GFLOPS")
    print(f"      • Memory Test: {'✅ PASSED' if cuda_functional_results['memory_test_passed'] else '❌ FAILED'}")
    mixed_p_count = sum(cuda_functional_results['mixed_precision_support'].values())
    print(f"      • Mixed Precision: {mixed_p_count}/3 types supported")
elif cuda_functional_results["status"] == "SKIPPED":
    print(f"      • Skipped (CUDA not available)")
else:
    print(f"      • Errors: {len(cuda_functional_results['errors'])}")

# Section 4: Dependency Stack Validation
dep_status = "✅ PASS" if dependency_validation_results["status"] == "READY" else ("⚠️ PARTIAL" if dependency_validation_results["status"] == "PARTIAL" else "❌ FAIL")
print(f"\n   4. Dependency Stack: {dep_status}")
if dependency_validation_results["status"] in ["READY", "PARTIAL"]:
    print(f"      • BioNeMo Core: {'✅ Installed' if dependency_validation_results['bionemo_core_installed'] else '❌ Not Installed'}")
    if dependency_validation_results['dependency_versions']['nemo']:
        print(f"      • NeMo: v{dependency_validation_results['dependency_versions']['nemo']}")
    else:
        print(f"      • NeMo: ❌ Not Installed")
    print(f"      • Version Compatibility: {'✅ All Pass' if dependency_validation_results['version_compatibility']['all_compatible'] else '⚠️ Issues'}")
    print(f"      • Autocast: {'✅ Working' if dependency_validation_results['autocast_functional'] else '⚠️ Not Tested'}")
else:
    print(f"      • Errors: {len(dependency_validation_results['critical_errors'])}")

# Section 5: BioNeMo Packages
if bionemo_test_results["status"] == "NO_PACKAGES":
    bionemo_status = "⚠️ NOT INSTALLED"
elif bionemo_test_results["status"] == "PASSED":
    bionemo_status = "✅ PASS"
else:
    bionemo_status = "❌ FAIL"

print(f"\n   5. BioNeMo Packages: {bionemo_status}")
installed = sum(1 for r in bionemo_test_results["packages_tested"].values() if r["is_installed"])
importable = sum(1 for r in bionemo_test_results["packages_tested"].values() if r["is_importable"])
print(f"      • Installed: {installed}/{len(bionemo_packages)}")
print(f"      • Importable: {importable}/{len(bionemo_packages)}")

# Overall Status
print("\n" + "=" * 80)
print(f"🎯 OVERALL STATUS: {final_report['overall_status']}")
print("=" * 80)

if final_report["total_blockers"] > 0:
    print(f"\n❌ VALIDATION FAILED")
    print(f"   Total Blockers: {final_report['total_blockers']}")
    print(f"   Total Warnings: {final_report['total_warnings']}")
    
    print(f"\n🚨 BLOCKERS FOUND:")
    for section_name, section_data in final_report["validation_sections"].items():
        if section_data.get("blockers"):
            print(f"\n   {section_name.replace('_', ' ').title()}:")
            for blocker in section_data["blockers"]:
                print(f"      • {blocker['message']}")
    
    # Add recommendations
    final_report["recommendations"].append("Fix all blockers before proceeding with BioNeMo installation")
    
elif final_report["overall_status"] == "READY_FOR_INSTALL":
    print(f"\n✅ ENVIRONMENT READY FOR BIONEMO INSTALLATION")
    print(f"\n📦 Next Steps:")
    print(f"\n   1. Choose your BioNeMo installation path:")
    print(f"")
    print(f"      🎯 Option A: BioNeMo Recipes (Recommended for Databricks)")
    print(f"         - Lightweight, pip-installable")
    print(f"         - Supports ESM2, Geneformer, DNABERT, Llama3")
    print(f"         - Install: pip install git+https://github.com/NVIDIA/bionemo-framework.git#subdirectory=bionemo-recipes")
    print(f"")
    print(f"      🎯 Option B: BioNeMo Core + Tooling")
    print(f"         - Install individual packages:")
    print(f"         - pip install bionemo-core")
    print(f"         - pip install bionemo-scdl  # Single cell data loader")
    print(f"         - pip install bionemo-moco  # Molecular co-design")
    print(f"")
    print(f"      🎯 Option C: BioNeMo 5D Parallelism (Requires Docker)")
    print(f"         - Use NVIDIA BioNeMo container:")
    print(f"         - docker run --rm -it --gpus=all nvcr.io/nvidia/clara/bionemo-framework:latest")
    print(f"         - Not recommended for Databricks managed clusters")
    print(f"")
    print(f"   2. After installation, re-run this notebook to validate")
    print(f"")
    print(f"   3. Refer to BioNeMo documentation:")
    print(f"      https://nvidia.github.io/bionemo-framework/")
    
    final_report["recommendations"].extend([
        "Install BioNeMo recipes for Databricks: pip install git+https://github.com/NVIDIA/bionemo-framework.git#subdirectory=bionemo-recipes",
        "Or install individual tools: pip install bionemo-scdl bionemo-moco",
        "Re-run this notebook after installation to validate"
    ])
    
else:
    print(f"\n✅ VALIDATION PASSED - BIONEMO READY")
    print(f"\n🎉 Your environment is configured correctly for BioNeMo workloads!")
    
    # Show installed packages
    installed_packages = [name for name, result in bionemo_test_results["packages_tested"].items() if result["is_importable"]]
    if installed_packages:
        print(f"\n📦 Available BioNeMo packages:")
        for pkg in installed_packages:
            result = bionemo_test_results["packages_tested"][pkg]
            print(f"      • {pkg} v{result['version']} ({result['category']})")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Explore BioNeMo tutorials and examples")
    print(f"   2. Start with small models to validate training pipeline")
    print(f"   3. Monitor GPU utilization during training")
    print(f"   4. Refer to BioNeMo documentation for best practices")
    
    final_report["recommendations"].extend([
        "Environment is ready for BioNeMo workloads",
        "Explore BioNeMo tutorials: https://nvidia.github.io/bionemo-framework/",
        "Start with small models to validate training pipeline"
    ])

if final_report["total_warnings"] > 0:
    print(f"\n⚠️  WARNINGS ({final_report['total_warnings']}):")
    for section_name, section_data in final_report["validation_sections"].items():
        if section_data.get("warnings"):
            for warning in section_data["warnings"]:
                print(f"   • {warning['message']}")

print("\n" + "=" * 80)
print("📚 REFERENCES")
print("=" * 80)
print("   • BioNeMo Framework: https://github.com/NVIDIA/bionemo-framework")
print("   • BioNeMo Documentation: https://nvidia.github.io/bionemo-framework/")
print("   • CUDA Healthcheck Tool: https://github.com/TavnerJC/cuda-healthcheck-on-databricks")
print("   • PyTorch Lightning: https://lightning.ai/docs/pytorch/")
print("=" * 80)

# Export report to JSON
print(f"\n💾 Exporting validation report...")
report_json = json.dumps(final_report, indent=2)

# Save to DBFS (if available)
try:
    report_path = "/dbfs/tmp/bionemo_validation_report.json"
    with open(report_path, 'w') as f:
        f.write(report_json)
    print(f"   ✅ Report saved to: {report_path}")
except Exception as e:
    print(f"   ⚠️  Could not save to DBFS: {str(e)}")
    print(f"   📋 Report JSON:")
    print(report_json[:500] + "..." if len(report_json) > 500 else report_json)

print("\n✅ VALIDATION COMPLETE!")
print("=" * 80)

