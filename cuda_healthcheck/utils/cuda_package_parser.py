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


def validate_cuda_library_versions(packages: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate all CUDA library version compatibility.

    Runs multiple compatibility checks:
    1. cuBLAS/nvJitLink version match (CRITICAL)
    2. CuOPT nvJitLink compatibility
    3. PyTorch CUDA branch compatibility (if runtime version provided)

    Args:
        packages: Output from parse_cuda_packages()

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
