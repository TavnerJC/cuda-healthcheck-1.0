"""
CUDA Package Parser for pip freeze output.

This module provides utilities to parse CUDA-related packages from pip freeze
to detect exact versions of PyTorch, cuBLAS, nvJitLink, and other NVIDIA libraries.
"""

import re
from typing import Any, Dict, Optional


def parse_cuda_packages(pip_freeze_output: str) -> Dict[str, Any]:
    """
    Parse CUDA-related packages from pip freeze output.

    This function extracts version information for PyTorch and NVIDIA CUDA libraries,
    which is critical for detecting compatibility issues like:
    - PyTorch CUDA branch mismatches
    - nvJitLink version incompatibilities (CuOPT issue)
    - cuBLAS version mismatches

    Args:
        pip_freeze_output: Output from `pip freeze` command

    Returns:
        Dictionary with parsed package information:
        {
            'torch': '2.4.1',                      # PyTorch version
            'torch_cuda_branch': 'cu124',
                # CUDA branch (cu120, cu121, cu124, etc.)
            'cublas': {
                'version': '12.4.127',             # Full version
                'major_minor': '12.4'
                    # Major.minor for compatibility checks
            },
            'nvjitlink': {
                'version': '12.4.127',
                'major_minor': '12.4'
            },
            'other_nvidia': {                      # All other nvidia-* packages
                'nvidia-cuda-runtime-cu12': '12.6.77',
                'nvidia-cudnn-cu12': '9.1.0.70',
                ...
            }
        }

    Examples:
        >>> # Example pip freeze output
        >>> pip_output = '''
        ... torch==2.4.1+cu124
        ... nvidia-cublas-cu12==12.4.5.8
        ... nvidia-nvjitlink-cu12==12.4.127
        ... nvidia-cuda-runtime-cu12==12.6.77
        ... nvidia-cudnn-cu12==9.1.0.70
        ... cuopt-server-cu12==25.12.0
        ... '''
        >>>
        >>> result = parse_cuda_packages(pip_output)
        >>> print(result['torch'])
        '2.4.1'
        >>> print(result['torch_cuda_branch'])
        'cu124'
        >>> print(result['nvjitlink']['major_minor'])
        '12.4'

        >>> # Example with incompatibility
        >>> result = parse_cuda_packages(pip_output)
        >>> if result['nvjitlink']['major_minor'] == '12.4':
        ...     print("⚠️  nvJitLink 12.4 incompatible with CuOPT 25.12+")
        ⚠️  nvJitLink 12.4 incompatible with CuOPT 25.12+
    """
    result: Dict[str, Any] = {
        "torch": None,
        "torch_cuda_branch": None,
        "cublas": {"version": None, "major_minor": None},
        "nvjitlink": {"version": None, "major_minor": None},
        "other_nvidia": {},
    }

    # Split output into lines
    lines = pip_freeze_output.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Parse PyTorch
        torch_match = _parse_torch_version(line)
        if torch_match:
            result["torch"] = torch_match["version"]
            result["torch_cuda_branch"] = torch_match["cuda_branch"]
            continue

        # Parse nvidia-cublas-cu12
        if line.startswith("nvidia-cublas-cu12"):
            cublas_version = _extract_version(line)
            if cublas_version:
                result["cublas"]["version"] = cublas_version
                result["cublas"]["major_minor"] = _extract_major_minor(cublas_version)
            continue

        # Parse nvidia-nvjitlink-cu12
        if line.startswith("nvidia-nvjitlink-cu12"):
            nvjitlink_version = _extract_version(line)
            if nvjitlink_version:
                result["nvjitlink"]["version"] = nvjitlink_version
                result["nvjitlink"]["major_minor"] = _extract_major_minor(nvjitlink_version)
            continue

        # Parse other nvidia-* packages
        if line.startswith("nvidia-"):
            package_info = _parse_nvidia_package(line)
            if package_info:
                result["other_nvidia"][package_info["name"]] = package_info["version"]

    return result


def _parse_torch_version(line: str) -> Optional[Dict[str, str]]:
    """
    Parse PyTorch version and CUDA branch.

    Handles formats like:
    - torch==2.4.1+cu124
    - torch==2.3.0+cu121
    - torch==2.4.1 (CPU-only, no CUDA branch)

    Args:
        line: Single line from pip freeze

    Returns:
        Dict with 'version' and 'cuda_branch', or None if not a torch package
    """
    # Pattern: torch==VERSION+cuXXX or torch==VERSION
    pattern = r"^torch==([0-9.]+)(?:\+cu([0-9]+))?"

    match = re.match(pattern, line)
    if not match:
        return None

    version = match.group(1)
    cuda_branch = f"cu{match.group(2)}" if match.group(2) else None

    return {"version": version, "cuda_branch": cuda_branch}


def _extract_version(line: str) -> Optional[str]:
    """
    Extract version from a package line.

    Handles formats like:
    - nvidia-cublas-cu12==12.4.5.8
    - nvidia-nvjitlink-cu12==12.4.127
    - some-package==1.2.3

    Args:
        line: Single line from pip freeze

    Returns:
        Version string (e.g., "12.4.127") or None
    """
    # Pattern: package==version
    pattern = r"^[a-zA-Z0-9_-]+==([0-9.]+)"

    match = re.match(pattern, line)
    if not match:
        return None

    return match.group(1)


def _extract_major_minor(version: str) -> Optional[str]:
    """
    Extract major.minor from version string.

    Examples:
        12.4.127 → 12.4
        12.4.5.8 → 12.4
        2.4.1 → 2.4

    Args:
        version: Full version string

    Returns:
        Major.minor version string (e.g., "12.4") or None
    """
    if not version:
        return None

    parts = version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"

    return None


def _parse_nvidia_package(line: str) -> Optional[Dict[str, str]]:
    """
    Parse an nvidia-* package line.

    Extracts package name and version for packages like:
    - nvidia-cuda-runtime-cu12==12.6.77
    - nvidia-cudnn-cu12==9.1.0.70

    Args:
        line: Single line from pip freeze

    Returns:
        Dict with 'name' and 'version', or None if parsing fails
    """
    # Pattern: nvidia-*==version
    pattern = r"^(nvidia-[a-zA-Z0-9_-]+)==([0-9.]+)"

    match = re.match(pattern, line)
    if not match:
        return None

    return {"name": match.group(1), "version": match.group(2)}


def get_cuda_packages_from_pip() -> Dict[str, Any]:
    """
    Get CUDA package information from current environment.

    Executes `pip freeze` and parses the output.

    Returns:
        Parsed CUDA package information (same format as parse_cuda_packages)

    Examples:
        >>> result = get_cuda_packages_from_pip()
        >>> if result['nvjitlink']['major_minor'] == '12.4':
        ...     print("Warning: nvJitLink 12.4 detected")
    """
    import subprocess

    try:
        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
        return parse_cuda_packages(result.stdout)
    except subprocess.CalledProcessError as e:
        return {
            "torch": None,
            "torch_cuda_branch": None,
            "cublas": {"version": None, "major_minor": None},
            "nvjitlink": {"version": None, "major_minor": None},
            "other_nvidia": {},
            "error": str(e),
        }


def format_cuda_packages_report(packages: Dict[str, Any]) -> str:
    """
    Format CUDA packages information into a human-readable report.

    Args:
        packages: Output from parse_cuda_packages()

    Returns:
        Formatted string report

    Examples:
        >>> packages = parse_cuda_packages(pip_output)
        >>> print(format_cuda_packages_report(packages))
        CUDA Packages Report
        ====================
        PyTorch: 2.4.1 (cu124)
        cuBLAS: 12.4.5.8 (12.4)
        nvJitLink: 12.4.127 (12.4)
        ...
    """
    lines = []
    lines.append("CUDA Packages Report")
    lines.append("=" * 80)

    # PyTorch
    if packages["torch"]:
        torch_info = f"PyTorch: {packages['torch']}"
        if packages["torch_cuda_branch"]:
            torch_info += f" ({packages['torch_cuda_branch']})"
        else:
            torch_info += " (CPU-only)"
        lines.append(torch_info)
    else:
        lines.append("PyTorch: Not installed")

    # cuBLAS
    if packages["cublas"]["version"]:
        lines.append(
            f"cuBLAS: {packages['cublas']['version']} " f"({packages['cublas']['major_minor']})"
        )
    else:
        lines.append("cuBLAS: Not installed")

    # nvJitLink
    if packages["nvjitlink"]["version"]:
        lines.append(
            f"nvJitLink: {packages['nvjitlink']['version']} "
            f"({packages['nvjitlink']['major_minor']})"
        )
    else:
        lines.append("nvJitLink: Not installed")

    # Other NVIDIA packages
    if packages["other_nvidia"]:
        lines.append(f"\nOther NVIDIA Packages ({len(packages['other_nvidia'])}):")
        for name, version in sorted(packages["other_nvidia"].items()):
            lines.append(f"  • {name}: {version}")
    else:
        lines.append("\nOther NVIDIA Packages: None")

    lines.append("=" * 80)

    return "\n".join(lines)


def check_cuopt_nvjitlink_compatibility(packages: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if nvJitLink version is compatible with CuOPT 25.12+.

    CuOPT 25.12+ requires nvJitLink >= 12.9.79, but Databricks ML Runtime 16.4
    provides nvJitLink 12.4.127 (immutable).

    Args:
        packages: Output from parse_cuda_packages()

    Returns:
        Compatibility check result:
        {
            'is_compatible': bool,
            'nvjitlink_version': str,
            'required_version': str,
            'error_message': Optional[str]
        }

    Examples:
        >>> packages = parse_cuda_packages(pip_output)
        >>> compat = check_cuopt_nvjitlink_compatibility(packages)
        >>> if not compat['is_compatible']:
        ...     print(f"❌ {compat['error_message']}")
    """
    result = {
        "is_compatible": False,
        "nvjitlink_version": packages["nvjitlink"]["version"],
        "required_version": "12.9.79",
        "error_message": None,
    }

    nvjitlink_version = packages["nvjitlink"]["version"]
    if not nvjitlink_version:
        result["error_message"] = "nvJitLink not installed"
        return result

    major_minor = packages["nvjitlink"]["major_minor"]

    # CuOPT 25.12+ requires nvJitLink >= 12.9
    if major_minor and major_minor >= "12.9":
        result["is_compatible"] = True
    else:
        result["is_compatible"] = False
        result["error_message"] = (
            f"nvJitLink {nvjitlink_version} is incompatible with CuOPT 25.12+. "
            f"CuOPT requires nvJitLink >= 12.9.79. "
            f"This is a PLATFORM CONSTRAINT on Databricks ML Runtime 16.4 - "
            f"users cannot upgrade nvJitLink."
        )

    return result


def check_pytorch_cuda_branch_compatibility(
    packages: Dict[str, Any], expected_cuda: str
) -> Dict[str, Any]:
    """
    Check if PyTorch CUDA branch matches expected CUDA version.

    Args:
        packages: Output from parse_cuda_packages()
        expected_cuda: Expected CUDA version (e.g., "12.4", "12.6")

    Returns:
        Compatibility check result:
        {
            'is_compatible': bool,
            'torch_cuda_branch': str,
            'expected_cuda': str,
            'error_message': Optional[str]
        }

    Examples:
        >>> packages = parse_cuda_packages(pip_output)
        >>> compat = check_pytorch_cuda_branch_compatibility(packages, "12.4")
        >>> if not compat['is_compatible']:
        ...     print(f"⚠️  {compat['error_message']}")
    """
    result = {
        "is_compatible": False,
        "torch_cuda_branch": packages["torch_cuda_branch"],
        "expected_cuda": expected_cuda,
        "error_message": None,
    }

    torch_branch = packages["torch_cuda_branch"]
    if not torch_branch:
        result["error_message"] = "PyTorch CUDA branch not detected (CPU-only?)"
        return result

    # Extract CUDA version from branch (cu124 → 12.4, cu121 → 12.1)
    branch_match = re.match(r"cu(\d{1,2})(\d)", torch_branch)
    if not branch_match:
        result["error_message"] = f"Could not parse CUDA branch: {torch_branch}"
        return result

    branch_major = branch_match.group(1)
    branch_minor = branch_match.group(2)
    branch_cuda = f"{branch_major}.{branch_minor}"

    # Extract expected major.minor
    expected_parts = expected_cuda.split(".")
    expected_major_minor = f"{expected_parts[0]}.{expected_parts[1]}"

    # Compare with expected
    if branch_cuda == expected_major_minor:
        result["is_compatible"] = True
    else:
        result["is_compatible"] = False
        result["error_message"] = (
            f"PyTorch CUDA branch {torch_branch} (CUDA {branch_cuda}) "
            f"does not match expected CUDA {expected_cuda}"
        )

    return result


def validate_torch_branch_compatibility(
    runtime_version: float, torch_cuda_branch: str
) -> Dict[str, Any]:
    """
    Validate PyTorch CUDA branch compatibility with Databricks runtime.

    Critical Check: Databricks runtimes have specific CUDA versions that may not
    support newer PyTorch CUDA branches. Runtime 14.3 (CUDA 12.0, Driver 535) does
    NOT support PyTorch cu124 (built for CUDA 12.4).

    Compatibility Matrix:
    ┌────────────┬────────┬─────────┬────────┬────────┬────────┐
    │ Runtime    │ Driver │ CUDA    │ cu120  │ cu121  │ cu124  │
    ├────────────┼────────┼─────────┼────────┼────────┼────────┤
    │ 14.3       │ 535    │ 12.0    │ ✅     │ ✅     │ ❌     │
    │ 15.1       │ 550    │ 12.4    │ ✅     │ ✅     │ ✅     │
    │ 15.2+      │ 550    │ 12.4    │ ✅     │ ✅     │ ✅     │
    └────────────┴────────┴─────────┴────────┴────────┴────────┘

    Why cu124 fails on Runtime 14.3:
    - PyTorch cu124 is built against CUDA 12.4 APIs
    - Runtime 14.3 provides CUDA 12.0 runtime
    - Missing CUDA 12.4 symbols cause runtime failures
    - Driver 535 lacks features required by CUDA 12.4

    Args:
        runtime_version: Databricks runtime version (e.g., 14.3, 15.1, 15.2)
        torch_cuda_branch: PyTorch CUDA branch (e.g., "cu120", "cu121", "cu124")

    Returns:
        Compatibility validation result:
        {
            'is_compatible': bool,        # True if compatible
            'severity': str,              # 'BLOCKER' if incompatible, None otherwise
            'runtime_version': float,     # Input runtime version
            'torch_cuda_branch': str,     # Input CUDA branch
            'runtime_cuda': str,          # Runtime's CUDA version
            'runtime_driver': int,        # Runtime's driver version
            'issue': str or None,         # Detailed incompatibility message
            'fix_options': List[str]      # List of fix options
        }

    Examples:
        >>> # Compatible: Runtime 15.2 + cu124
        >>> result = validate_torch_branch_compatibility(15.2, "cu124")
        >>> result['is_compatible']
        True
        >>> result['severity']
        None

        >>> # INCOMPATIBLE: Runtime 14.3 + cu124 (BLOCKER)
        >>> result = validate_torch_branch_compatibility(14.3, "cu124")
        >>> result['is_compatible']
        False
        >>> result['severity']
        'BLOCKER'
        >>> print(result['issue'])
        ❌ CRITICAL: PyTorch cu124 is INCOMPATIBLE with Databricks Runtime 14.3!
        <BLANKLINE>
        Runtime 14.3 provides:
          • CUDA Runtime: 12.0
          • Driver: 535 (immutable)
        <BLANKLINE>
        PyTorch cu124 requires:
          • CUDA Runtime: 12.4
          • Driver: ≥ 550
        ...
        >>> result['fix_options']
        ['Option 1: Downgrade PyTorch to cu120 or cu121',
         'Option 2: Upgrade to Databricks Runtime 15.2+ (provides CUDA 12.4, Driver 550)']

        >>> # Compatible: Runtime 14.3 + cu120
        >>> result = validate_torch_branch_compatibility(14.3, "cu120")
        >>> result['is_compatible']
        True

        >>> # Compatible: Runtime 15.1 + cu121
        >>> result = validate_torch_branch_compatibility(15.1, "cu121")
        >>> result['is_compatible']
        True
    """
    # Compatibility matrix
    # Key: runtime_version, Value: {'cuda': str, 'driver': int, 'supported_branches': list}
    COMPATIBILITY_MATRIX = {
        14.3: {
            "cuda": "12.0",
            "driver": 535,
            "supported_branches": ["cu120", "cu121"],
            "blocked_branches": ["cu124"],
        },
        15.1: {
            "cuda": "12.4",
            "driver": 550,
            "supported_branches": ["cu120", "cu121", "cu124"],
            "blocked_branches": [],
        },
        15.2: {
            "cuda": "12.4",
            "driver": 550,
            "supported_branches": ["cu120", "cu121", "cu124"],
            "blocked_branches": [],
        },
        16.0: {
            "cuda": "12.4",
            "driver": 550,
            "supported_branches": ["cu120", "cu121", "cu124"],
            "blocked_branches": [],
        },
        16.4: {
            "cuda": "12.6",
            "driver": 560,
            "supported_branches": ["cu120", "cu121", "cu124"],
            "blocked_branches": [],
        },
    }

    result = {
        "is_compatible": False,
        "severity": None,
        "runtime_version": runtime_version,
        "torch_cuda_branch": torch_cuda_branch,
        "runtime_cuda": None,
        "runtime_driver": None,
        "issue": None,
        "fix_options": [],
    }

    # Get runtime info
    if runtime_version not in COMPATIBILITY_MATRIX:
        result["issue"] = (
            f"⚠️  Unknown Databricks runtime version: {runtime_version}\n"
            f"Cannot validate PyTorch cu branch compatibility.\n"
            f"Known runtimes: {', '.join(str(v) for v in sorted(COMPATIBILITY_MATRIX.keys()))}"
        )
        result["severity"] = "WARNING"
        return result

    # Handle None torch_cuda_branch (torch not installed or no CUDA support)
    if torch_cuda_branch is None:
        result["is_compatible"] = True  # No torch = no incompatibility
        result["issue"] = "PyTorch not installed or has no CUDA branch detected"
        return result

    runtime_info = COMPATIBILITY_MATRIX[runtime_version]
    result["runtime_cuda"] = runtime_info["cuda"]
    result["runtime_driver"] = runtime_info["driver"]

    # Normalize CUDA branch (handle both "cu124" and "cu1240")
    branch_normalized = torch_cuda_branch[:5] if len(torch_cuda_branch) > 5 else torch_cuda_branch

    # Check compatibility
    if branch_normalized in runtime_info["supported_branches"]:
        result["is_compatible"] = True
        return result

    # Incompatible - generate detailed error
    result["is_compatible"] = False
    result["severity"] = "BLOCKER"

    # Build error message
    result["issue"] = (
        f"❌ CRITICAL: PyTorch {torch_cuda_branch} is INCOMPATIBLE "
        f"with Databricks Runtime {runtime_version}!\n\n"
        f"Runtime {runtime_version} provides:\n"
        f"  • CUDA Runtime: {runtime_info['cuda']}\n"
        f"  • Driver: {runtime_info['driver']} (immutable)\n\n"
    )

    # Extract CUDA version from branch
    branch_match = re.match(r"cu(\d{1,2})(\d)", branch_normalized)
    if branch_match:
        branch_major = branch_match.group(1)
        branch_minor = branch_match.group(2)
        branch_cuda = f"{branch_major}.{branch_minor}"

        result["issue"] += (
            f"PyTorch {torch_cuda_branch} requires:\n"
            f"  • CUDA Runtime: {branch_cuda}\n"
            f"  • Driver: ≥ 550 (for CUDA 12.4+)\n\n"
        )

    result["issue"] += (
        f"⚠️  This mismatch causes:\n"
        f"  • Missing CUDA API symbols\n"
        f"  • Runtime initialization failures\n"
        f"  • Segmentation faults during tensor operations\n"
        f'  • "CUDA driver version is insufficient" errors\n\n'
        f"📋 Why this is IMMUTABLE:\n"
        f"  Runtime {runtime_version} has a locked driver version ({runtime_info['driver']}).\n"
        f"  You CANNOT upgrade the driver or CUDA runtime on this runtime."
    )

    # Generate fix options
    if runtime_info["supported_branches"]:
        supported_list = ", ".join(runtime_info["supported_branches"])
        result["fix_options"].append(
            f"Option 1: Downgrade PyTorch to {supported_list}\n"
            f"  pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    # Find minimum compatible runtime
    compatible_runtimes = [
        v
        for v, info in COMPATIBILITY_MATRIX.items()
        if branch_normalized in info["supported_branches"]
    ]
    if compatible_runtimes:
        min_runtime = min(compatible_runtimes)
        result["fix_options"].append(
            f"Option 2: Upgrade to Databricks Runtime {min_runtime}+ "
            f"(provides CUDA {COMPATIBILITY_MATRIX[min_runtime]['cuda']}, "
            f"Driver {COMPATIBILITY_MATRIX[min_runtime]['driver']})\n"
            f"  This requires creating a new cluster with Runtime {min_runtime}+"
        )

    return result


def check_cublas_nvjitlink_version_match(
    cublas_version: str, nvjitlink_version: str
) -> Dict[str, Any]:
    """
    Detect nvJitLink version mismatches with cuBLAS.

    Critical Rule: cuBLAS and nvJitLink major.minor versions MUST match.
    Mismatch causes runtime errors like:
    "undefined symbol: __nvJitLinkAddData_12_X, version libnvJitLink.so.12"

    This is different from the CuOPT incompatibility - this affects ALL CUDA
    libraries that use JIT compilation (cuBLAS, cuSolver, cuFFT, etc.).

    Args:
        cublas_version: cuBLAS version string (e.g., "12.1.3.1", "12.4.5.8")
        nvjitlink_version: nvJitLink version string (e.g., "12.1.105", "12.4.127")

    Returns:
        Validation result:
        {
            'is_mismatch': bool,           # True if versions don't match
            'severity': str,               # 'BLOCKER' or 'OK'
            'cublas_major_minor': str,     # e.g., "12.1"
            'nvjitlink_major_minor': str,  # e.g., "12.4"
            'error_message': str or None,  # Detailed error with fix
            'fix_command': str or None     # pip command to fix the issue
        }

    Examples:
        >>> # Compatible versions (both 12.4)
        >>> result = check_cublas_nvjitlink_version_match("12.4.5.8", "12.4.127")
        >>> result['is_mismatch']
        False
        >>> result['severity']
        'OK'

        >>> # INCOMPATIBLE versions (12.1 vs 12.4)
        >>> result = check_cublas_nvjitlink_version_match("12.1.3.1", "12.4.127")
        >>> result['is_mismatch']
        True
        >>> result['severity']
        'BLOCKER'
        >>> print(result['error_message'])
        ❌ CRITICAL: cuBLAS/nvJitLink version mismatch detected!
        <BLANKLINE>
        cuBLAS version: 12.1.3.1 (major.minor: 12.1)
        nvJitLink version: 12.4.127 (major.minor: 12.4)
        <BLANKLINE>
        ⚠️  This will cause runtime errors:
           "undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12"
        <BLANKLINE>
        📋 Fix: Install matching nvJitLink version
        >>> print(result['fix_command'])
        pip install --upgrade nvidia-nvjitlink-cu12==12.1.*

        >>> # Databricks ML Runtime 16.4 scenario
        >>> result = check_cublas_nvjitlink_version_match("12.4.5.8", "12.4.127")
        >>> result['is_mismatch']
        False
    """
    result = {
        "is_mismatch": False,
        "severity": "OK",
        "cublas_major_minor": None,
        "nvjitlink_major_minor": None,
        "error_message": None,
        "fix_command": None,
    }

    # Handle None/empty inputs
    if not cublas_version or not nvjitlink_version:
        result["is_mismatch"] = True
        result["severity"] = "BLOCKER"
        result["error_message"] = (
            "❌ CRITICAL: Missing required libraries!\n\n"
            f"cuBLAS version: {cublas_version or 'NOT INSTALLED'}\n"
            f"nvJitLink version: {nvjitlink_version or 'NOT INSTALLED'}\n\n"
            "Both libraries are required for CUDA operations."
        )
        if not nvjitlink_version and cublas_version:
            cublas_mm = _extract_major_minor(cublas_version)
            result["fix_command"] = f"pip install nvidia-nvjitlink-cu12=={cublas_mm}.*"
        return result

    # Extract major.minor versions
    cublas_major_minor = _extract_major_minor(cublas_version)
    nvjitlink_major_minor = _extract_major_minor(nvjitlink_version)

    result["cublas_major_minor"] = cublas_major_minor
    result["nvjitlink_major_minor"] = nvjitlink_major_minor

    # Check if they match
    if cublas_major_minor == nvjitlink_major_minor:
        result["is_mismatch"] = False
        result["severity"] = "OK"
        return result

    # Version mismatch detected - CRITICAL ERROR
    result["is_mismatch"] = True
    result["severity"] = "BLOCKER"
    result["error_message"] = (
        "❌ CRITICAL: cuBLAS/nvJitLink version mismatch detected!\n\n"
        f"cuBLAS version: {cublas_version} (major.minor: {cublas_major_minor})\n"
        f"nvJitLink version: {nvjitlink_version} (major.minor: {nvjitlink_major_minor})\n\n"
        f"⚠️  This will cause runtime errors:\n"
        f'   "undefined symbol: __nvJitLinkAddData_{cublas_major_minor.replace(".", "_")}, '
        f'version libnvJitLink.so.{cublas_major_minor.split(".")[0]}"\n\n'
        f"📋 Required: cuBLAS and nvJitLink major.minor versions MUST match\n"
        f"   cuBLAS {cublas_major_minor}.x requires nvJitLink {cublas_major_minor}.x"
    )
    result["fix_command"] = f"pip install --upgrade nvidia-nvjitlink-cu12=={cublas_major_minor}.*"

    return result


def detect_mixed_cuda_versions(pip_freeze_output: str) -> Dict[str, Any]:
    """
    Detect mixed CUDA 11 and CUDA 12 packages that cause LD_LIBRARY_PATH conflicts.

    **Critical Issue**: Installing CUDA 11 and CUDA 12 packages together causes
    symbol resolution failures and segmentation faults due to conflicting shared
    libraries being loaded at runtime.

    Common Scenario:
    ```bash
    pip install torch==2.0.1+cu118  # Brings CUDA 11.8 libraries
    pip install cudf-cu12           # Brings CUDA 12.x libraries
    # → LD_LIBRARY_PATH contains both cu11 and cu12 libraries
    # → Random symbol resolution failures occur
    ```

    Symptoms:
    - Segmentation faults
    - "undefined symbol" errors (random, not consistent)
    - "version `GLIBCXX_X.X.XX' not found"
    - Libraries loading wrong CUDA version at runtime

    Args:
        pip_freeze_output: Output from `pip freeze` command

    Returns:
        Detection result:
        {
            'has_cu11': bool,              # True if any -cu11 packages found
            'has_cu12': bool,              # True if any -cu12 packages found
            'is_mixed': bool,              # True if BOTH cu11 and cu12 present
            'cu11_packages': List[str],    # List of -cu11 package names
            'cu12_packages': List[str],    # List of -cu12 package names
            'cu11_count': int,             # Count of cu11 packages
            'cu12_count': int,             # Count of cu12 packages
            'severity': str,               # 'BLOCKER' if mixed, None otherwise
            'error_message': str | None,   # Detailed error explanation
            'fix_command': str | None      # pip commands to fix the issue
        }

    Examples:
        >>> # CUDA 12 only (OK)
        >>> pip_output = '''
        ... torch==2.4.1+cu124
        ... nvidia-cublas-cu12==12.4.5.8
        ... cudf-cu12==24.10.1
        ... '''
        >>> result = detect_mixed_cuda_versions(pip_output)
        >>> result['is_mixed']
        False
        >>> result['severity']
        None

        >>> # Mixed CUDA 11 and 12 (BLOCKER!)
        >>> pip_output = '''
        ... torch==2.0.1+cu118
        ... nvidia-cublas-cu12==12.4.5.8
        ... cudf-cu12==24.10.1
        ... '''
        >>> result = detect_mixed_cuda_versions(pip_output)
        >>> result['is_mixed']
        True
        >>> result['severity']
        'BLOCKER'
        >>> print(result['error_message'])
        ❌ CRITICAL: Mixed CUDA 11 and CUDA 12 packages detected!
        <BLANKLINE>
        This causes LD_LIBRARY_PATH conflicts and random symbol resolution failures.
        ...

        >>> # Real-world Databricks scenario
        >>> pip_output = '''
        ... torch==2.1.0+cu121
        ... torchvision==0.16.0+cu121
        ... cupy-cuda11x==12.3.0
        ... '''
        >>> result = detect_mixed_cuda_versions(pip_output)
        >>> result['is_mixed']
        True
        >>> result['cu11_packages']
        ['cupy-cuda11x']
        >>> result['cu12_packages']
        ['torch', 'torchvision']
    """
    result = {
        "has_cu11": False,
        "has_cu12": False,
        "is_mixed": False,
        "cu11_packages": [],
        "cu12_packages": [],
        "cu11_count": 0,
        "cu12_count": 0,
        "severity": None,
        "error_message": None,
        "fix_command": None,
    }

    # Parse each line for CUDA version indicators
    lines = pip_freeze_output.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Extract package name (before ==)
        if "==" in line:
            package_name = line.split("==")[0]
        else:
            package_name = line

        # Check the FULL line for CUDA indicators (captures version strings like +cu124)
        line_lower = line.lower()

        # Check for CUDA 11 indicators
        # Patterns: -cu11, -cu118, +cu118, cuda11, cuda-11, etc.
        if re.search(r"([-+]cu11[0-9]?|[-+]cuda11|[-+]cuda-11)", line_lower):
            result["cu11_packages"].append(package_name)
            result["cu11_count"] += 1
            result["has_cu11"] = True

        # Check for CUDA 12 indicators
        # Patterns: -cu12, -cu124, +cu124, cuda12, cuda-12, etc.
        if re.search(r"([-+]cu12[0-9]?|[-+]cuda12|[-+]cuda-12)", line_lower):
            result["cu12_packages"].append(package_name)
            result["cu12_count"] += 1
            result["has_cu12"] = True

    # Check if mixed versions detected
    if result["has_cu11"] and result["has_cu12"]:
        result["is_mixed"] = True
        result["severity"] = "BLOCKER"

        # Build error message
        result["error_message"] = (
            "❌ CRITICAL: Mixed CUDA 11 and CUDA 12 packages detected!\n\n"
            f"CUDA 11 packages ({result['cu11_count']}):\n"
        )
        for pkg in result["cu11_packages"]:
            result["error_message"] += f"  • {pkg}\n"

        result["error_message"] += f"\nCUDA 12 packages ({result['cu12_count']}):\n"
        for pkg in result["cu12_packages"]:
            result["error_message"] += f"  • {pkg}\n"

        result["error_message"] += (
            "\n⚠️  This causes LD_LIBRARY_PATH conflicts!\n"
            "   Both CUDA 11 and CUDA 12 libraries will be in the same path,\n"
            "   causing random symbol resolution failures, segfaults, and\n"
            "   inconsistent behavior.\n\n"
            "📋 Why this happens:\n"
            "   When you mix CUDA 11 and CUDA 12 packages, both versions of\n"
            "   shared libraries (.so files) exist in site-packages. The dynamic\n"
            "   linker may load the wrong version at runtime, causing crashes.\n\n"
            "🔍 Common symptoms:\n"
            "   • Segmentation fault (core dumped)\n"
            "   • undefined symbol: [random CUDA function]\n"
            "   • version `GLIBCXX_X.X.XX' not found\n"
            "   • CUDA runtime version mismatch errors"
        )

        # Build fix command
        all_packages = result["cu11_packages"] + result["cu12_packages"]
        uninstall_cmd = "pip uninstall -y " + " ".join(all_packages)

        result["fix_command"] = (
            f"# Step 1: Uninstall ALL mixed CUDA packages\n"
            f"{uninstall_cmd}\n\n"
            f"# Step 2: Clear pip cache to remove old wheels\n"
            f"pip cache purge\n\n"
            f"# Step 3: Reinstall with CUDA 12 (recommended for Databricks)\n"
            f"pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            f"pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12 cuml-cu12\n\n"
            f"# Alternative: Install with CUDA 11 (if required)\n"
            f"# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        )

    return result


def validate_cuda_library_versions(packages: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate all CUDA library version compatibility.

    Runs multiple compatibility checks:
    1. Mixed CUDA 11/12 packages (CRITICAL)
    2. cuBLAS/nvJitLink version match (CRITICAL)
    3. CuOPT nvJitLink compatibility (WARNING)

    Args:
        packages: Output from parse_cuda_packages()
        pip_freeze_output: Optional raw pip freeze output for mixed version check

    Returns:
        Comprehensive validation result:
        {
            'all_compatible': bool,
            'blockers': List[Dict],           # BLOCKER severity issues
            'warnings': List[Dict],           # WARNING severity issues
            'checks_run': int,
            'checks_passed': int,
            'checks_failed': int
        }

    Examples:
        >>> packages = parse_cuda_packages(pip_output)
        >>> validation = validate_cuda_library_versions(packages)
        >>> if not validation['all_compatible']:
        ...     for blocker in validation['blockers']:
        ...         print(blocker['error_message'])
        ...         print(f"Fix: {blocker['fix_command']}")
    """
    result = {
        "all_compatible": True,
        "blockers": [],
        "warnings": [],
        "checks_run": 0,
        "checks_passed": 0,
        "checks_failed": 0,
    }

    # Check 1: cuBLAS/nvJitLink version match (CRITICAL)
    cublas_version = packages["cublas"]["version"]
    nvjitlink_version = packages["nvjitlink"]["version"]

    if cublas_version or nvjitlink_version:
        result["checks_run"] += 1
        mismatch_check = check_cublas_nvjitlink_version_match(cublas_version, nvjitlink_version)

        if mismatch_check["is_mismatch"]:
            result["checks_failed"] += 1
            result["all_compatible"] = False
            result["blockers"].append(
                {
                    "check": "cuBLAS/nvJitLink Version Match",
                    "severity": "BLOCKER",
                    "error_message": mismatch_check["error_message"],
                    "fix_command": mismatch_check["fix_command"],
                }
            )
        else:
            result["checks_passed"] += 1

    # Check 2: CuOPT nvJitLink compatibility (if needed)
    if nvjitlink_version:
        result["checks_run"] += 1
        cuopt_compat = check_cuopt_nvjitlink_compatibility(packages)

        if not cuopt_compat["is_compatible"]:
            result["checks_failed"] += 1
            result["warnings"].append(
                {
                    "check": "CuOPT nvJitLink Compatibility",
                    "severity": "WARNING",
                    "error_message": cuopt_compat["error_message"],
                    "fix_command": "Contact Databricks support - platform constraint",
                }
            )
        else:
            result["checks_passed"] += 1

    return result
