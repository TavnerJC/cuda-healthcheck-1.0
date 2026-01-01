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
                result["nvjitlink"]["major_minor"] = _extract_major_minor(
                    nvjitlink_version
                )
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
        result = subprocess.run(
            ["pip", "freeze"], capture_output=True, text=True, check=True
        )
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
            f"cuBLAS: {packages['cublas']['version']} "
            f"({packages['cublas']['major_minor']})"
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
