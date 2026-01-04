# BioNeMo Framework Validation Notebook - Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  02_bionemo_framework_validation.py                                         │
│  NVIDIA BioNeMo Framework Validation for Databricks                         │
│  Size: 40.7 KB | Cells: 10 | Lines: 1,005                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ CELL 1: Setup and Imports                                                 │
├───────────────────────────────────────────────────────────────────────────┤
│ • Installs CUDA Healthcheck Tool (if needed)                              │
│ • Imports dependencies (sys, json, subprocess, datetime)                  │
│ • Handles ImportError with auto-installation                              │
│ • Output: Installation status + version info                              │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌───────────────────────────────────────────────────────────────────────────┐
│ CELL 2: CUDA Environment Validation (Reuses Existing Functions)          │
├───────────────────────────────────────────────────────────────────────────┤
│ ✅ Reuses: detect_databricks_runtime()                                    │
│ ✅ Reuses: detect_gpu_auto()                                              │
│ ✅ Reuses: CUDADetector()                                                 │
│ ✅ Reuses: PyTorch detection from libraries                               │
│                                                                           │
│ Checks:                                                                   │
│ • Databricks Runtime version (14.3, 15.1, 15.2, 16.4)                    │
│ • GPU hardware detection (Classic ML / Serverless)                       │
│ • CUDA runtime, driver, NVCC versions                                    │
│ • PyTorch installation and CUDA linkage                                  │
│                                                                           │
│ Output:                                                                   │
│ • cuda_validation_results (dict)                                         │
│ • DataFrame summary with Pass/Fail status                                │
│ • Blockers/Warnings list                                                 │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌───────────────────────────────────────────────────────────────────────────┐
│ CELL 3: PyTorch Lightning GPU Test (NEW!)                                │
├───────────────────────────────────────────────────────────────────────────┤
│ Tests:                                                                    │
│ • PyTorch Lightning installation (auto-install if missing)               │
│ • torch.cuda.is_available() = True                                       │
│ • GPU device enumeration via PyTorch                                     │
│ • Lightning Trainer initialization with GPU accelerator                  │
│ • SimpleLightningModule forward pass on GPU                              │
│ • Mixed precision (FP16) support via torch.cuda.amp.autocast             │
│ • GPU performance benchmark (throughput, latency)                        │
│                                                                           │
│ Output:                                                                   │
│ • lightning_test_results (dict)                                          │
│ • DataFrame with 5 checks (Lightning, GPU, Trainer, Forward, FP16)      │
│ • Benchmark results (iter/s, latency ms)                                 │
│ • Blockers/Warnings list                                                 │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌───────────────────────────────────────────────────────────────────────────┐
│ CELL 4: BioNeMo Core Package Availability (NEW!)                         │
├───────────────────────────────────────────────────────────────────────────┤
│ Tests 7 BioNeMo Packages:                                                │
│                                                                           │
│ 5D Parallelism Models:                                                   │
│ • bionemo-core        (Model config, test utilities)                     │
│ • bionemo-llm         (BioBert base model)                               │
│ • bionemo-evo2        (Evo2 model)                                       │
│ • bionemo-geneformer  (Geneformer model)                                 │
│                                                                           │
│ Tooling:                                                                 │
│ • bionemo-scdl        (Single cell data loader)                          │
│ • bionemo-moco        (Molecular co-design)                              │
│ • bionemo-noodles     (Fast FASTA I/O)                                   │
│                                                                           │
│ For each package:                                                        │
│ • Checks pip installation (subprocess: pip show)                         │
│ • Tests import capability (importlib.import_module)                      │
│ • Extracts version, submodules                                           │
│ • Handles missing packages gracefully                                    │
│                                                                           │
│ Output:                                                                   │
│ • bionemo_test_results (dict)                                            │
│ • DataFrame with Package, Category, Support, Install, Import, Version    │
│ • Detailed import errors for debugging                                   │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌───────────────────────────────────────────────────────────────────────────┐
│ CELL 5: Final Summary Report                                             │
├───────────────────────────────────────────────────────────────────────────┤
│ Aggregates:                                                               │
│ • cuda_validation_results                                                │
│ • lightning_test_results                                                 │
│ • bionemo_test_results                                                   │
│                                                                           │
│ Determines Overall Status:                                               │
│ • BLOCKED       → Lists all blockers with fix commands                   │
│ • READY_FOR_INSTALL → Provides 3 installation options:                   │
│   - Option A: BioNeMo Recipes (pip-installable, recommended)            │
│   - Option B: Core + Tooling (individual packages)                      │
│   - Option C: 5D Parallelism (Docker container)                         │
│ • READY         → Confirms BioNeMo-ready environment                     │
│                                                                           │
│ Output:                                                                   │
│ • final_report (dict) with all results                                   │
│ • Actionable recommendations                                             │
│ • JSON export to /dbfs/tmp/bionemo_validation_report.json               │
│ • Links to documentation and GitHub                                      │
└───────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

```
┌──────────────────┐
│  User Executes   │
│  Cell 1          │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐      ┌─────────────────────────────────┐
│  CUDA Healthcheck│      │  Existing Functions (Reused)    │
│  Tool Installed  │ ←────┤  • detect_databricks_runtime()  │
└────────┬─────────┘      │  • detect_gpu_auto()            │
         │                │  • CUDADetector()                │
         ↓                │  • get_cuda_packages_from_pip()  │
┌──────────────────┐      └─────────────────────────────────┘
│  Cell 2 Executes │
│  Reuses existing │
│  CUDA checks     │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│ cuda_validation_ │
│ results (dict)   │──┐
└──────────────────┘  │
         │            │
         ↓            │
┌──────────────────┐  │
│  Cell 3 Executes │  │
│  NEW: Lightning  │  │
│  GPU tests       │  │
└────────┬─────────┘  │
         │            │
         ↓            │
┌──────────────────┐  │
│ lightning_test_  │  │
│ results (dict)   │──┤
└──────────────────┘  │
         │            │
         ↓            │
┌──────────────────┐  │
│  Cell 4 Executes │  │
│  NEW: BioNeMo    │  │
│  package tests   │  │
└────────┬─────────┘  │
         │            │
         ↓            │
┌──────────────────┐  │
│ bionemo_test_    │  │
│ results (dict)   │──┤
└──────────────────┘  │
         │            │
         ↓            │
┌──────────────────┐  │
│  Cell 5 Executes │  │
│  Aggregates all  │ ←┘
│  results         │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  final_report    │
│  (dict)          │
├──────────────────┤
│  • Status        │
│  • Blockers      │
│  • Warnings      │
│  • Recommend.    │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  JSON Export     │
│  to DBFS         │
└──────────────────┘
```

## 🎯 Key Design Decisions

### ✅ No Duplication
- Reuses 4 existing functions from CUDA Healthcheck Tool
- Avoids reimplementing Databricks runtime detection
- Avoids reimplementing GPU detection logic
- Avoids reimplementing CUDA/PyTorch validation

### ✅ Comprehensive Error Handling
- 11 try-except blocks throughout all cells
- Graceful handling of missing packages
- Auto-installation where appropriate (Lightning)
- Clear error messages with context

### ✅ Structured Results
- All results stored in dictionaries for programmatic access
- Consistent status values: PASSED, BLOCKED, ERROR, NO_PACKAGES
- Blockers/Warnings arrays for debugging
- JSON-serializable for export to DBFS

### ✅ User-Friendly Output
- DataFrame summaries for visual inspection
- Human-readable status messages
- Actionable recommendations based on environment state
- Links to documentation and GitHub

### ✅ BioNeMo-Specific Tests
- **NEW:** PyTorch Lightning GPU compatibility (critical for BioNeMo recipes)
- **NEW:** 7 BioNeMo packages tested (5D models + tooling)
- **NEW:** Benchmark GPU performance for training validation
- **NEW:** Mixed precision (FP16) support check

## 📊 Validation Matrix

| Check | Method | Duplication? | Status |
|-------|--------|--------------|--------|
| Databricks Runtime Detection | `detect_databricks_runtime()` | ✅ Reused | ✅ |
| GPU Hardware Detection | `detect_gpu_auto()` | ✅ Reused | ✅ |
| CUDA Environment | `CUDADetector()` | ✅ Reused | ✅ |
| PyTorch Installation | Library detection | ✅ Reused | ✅ |
| PyTorch Lightning GPU | NEW test | ❌ No duplicate | ✅ |
| BioNeMo Packages | NEW test | ❌ No duplicate | ✅ |

## 🚀 Ready for Databricks!

The notebook is now available at:
https://github.com/TavnerJC/cuda-healthcheck-on-databricks/blob/main/notebooks/02_bionemo_framework_validation.py

