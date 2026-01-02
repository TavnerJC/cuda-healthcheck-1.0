"""
User-friendly recommendation generator for CUDA healthcheck blockers.

This module converts technical error messages into clear, actionable
recommendations for users.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_recommendations(
    blockers: List[Dict[str, Any]], runtime_version: Optional[float] = None
) -> str:
    """
    Convert technical blockers into clear user recommendations.

    Takes a list of technical blockers and converts them into plain English
    recommendations with actionable fix commands. Provides context-aware
    guidance based on Databricks runtime version.

    Args:
        blockers: List of blocker dictionaries with keys:
            - 'issue' or 'message': Technical description
            - 'root_cause': Category of issue (optional)
            - 'fix_command': Technical fix command (optional)
            - 'fix_commands' or 'fix_options': List of fixes (optional)
        runtime_version: Databricks runtime version for context-aware recommendations

    Returns:
        Markdown-formatted text with user-friendly recommendations

    Example:
        >>> blockers = [
        ...     {
        ...         'issue': 'Driver 535 (too old) for cu124 (requires 550+)',
        ...         'root_cause': 'driver_too_old',
        ...         'fix_options': ['Downgrade to cu120', 'Upgrade runtime to 15.2+']
        ...     }
        ... ]
        >>> recommendations = generate_recommendations(blockers, runtime_version=14.3)
        >>> print(recommendations)
    """
    if not blockers:
        return (
            "## ✅ No Blockers Detected!\n\n"
            "Your environment is configured correctly. You can proceed with your workload.\n"
        )

    md = []
    md.append("# 🚨 Action Required: Critical Issues Detected\n")
    md.append(
        "Your environment has issues that will prevent GPU workloads from running. "
        "Follow the recommendations below to fix them.\n"
    )
    md.append("---\n")

    for i, blocker in enumerate(blockers, 1):
        # Extract blocker info
        issue = blocker.get("issue") or blocker.get("message", "Unknown issue")
        root_cause = blocker.get("root_cause", "")
        feature = blocker.get("feature", "")
        fix_command = blocker.get("fix_command", "")
        fix_options = blocker.get("fix_options") or blocker.get("fix_commands", [])

        md.append(f"\n## Issue #{i}: {_get_user_friendly_title(root_cause, issue, feature)}\n")

        # Generate user-friendly explanation
        explanation = _generate_explanation(root_cause, issue, feature, runtime_version)
        md.append(f"{explanation}\n")

        # Generate fix recommendations
        if fix_options:
            md.append("\n### 🔧 How to Fix:\n")
            for j, option in enumerate(fix_options, 1):
                clean_option = _clean_fix_command(option)
                md.append(f"{j}. {clean_option}\n")
        elif fix_command:
            md.append("\n### 🔧 How to Fix:\n")
            clean_command = _clean_fix_command(fix_command)
            md.append(f"```bash\n{clean_command}\n```\n")

        md.append("\n---\n")

    # Add helpful tips section
    md.append("\n## 💡 General Tips\n")
    md.append(
        "- Always restart Python after installing packages: `dbutils.library.restartPython()`\n"
    )
    md.append("- Use `--no-cache-dir` to ensure fresh package downloads\n")
    md.append("- Check your Databricks runtime version: Some drivers are immutable\n")
    md.append("- For persistent issues, contact Databricks support or report to our GitHub\n")

    return "".join(md)


def _get_user_friendly_title(root_cause: str, issue: str, feature: str = "") -> str:
    """Generate a user-friendly title for the blocker."""
    titles = {
        "driver_too_old": "GPU Driver Too Old",
        "torch_not_installed": "PyTorch Not Installed",
        "torch_no_cuda_support": "PyTorch Missing CUDA Support",
        "cuda_libraries_missing": "CUDA Libraries Missing",
        "no_gpu_device": "No GPU Detected",
        "driver_version_mismatch": "Driver Version Incompatible",
        "nvjitlink_mismatch": "CUDA Library Version Mismatch",
        "mixed_cuda_versions": "Mixed CUDA 11 and CUDA 12 Packages",
        "torch_branch_incompatible": "PyTorch CUDA Branch Incompatible",
    }

    if root_cause in titles:
        return titles[root_cause]

    # Try to infer from issue text
    issue_lower = issue.lower()
    if "nvjitlink" in issue_lower or "cublas" in issue_lower:
        return "CUDA Library Version Mismatch"
    elif "driver" in issue_lower and "old" in issue_lower:
        return "GPU Driver Too Old"
    elif "mixed" in issue_lower and ("cu11" in issue_lower or "cu12" in issue_lower):
        return "Mixed CUDA Versions Detected"
    elif "torch" in issue_lower and "not installed" in issue_lower:
        return "PyTorch Not Installed"
    elif feature:
        return f"Issue with {feature}"

    return "Configuration Issue"


def _generate_explanation(
    root_cause: str, issue: str, feature: str, runtime_version: Optional[float]
) -> str:
    """Generate user-friendly explanation for the blocker."""
    explanations = {
        "nvjitlink_mismatch": (
            "**What's wrong:** Your CUDA libraries don't match. The cuBLAS and nvJitLink "
            "libraries must have the same major.minor version (e.g., both 12.4.x), but they don't.\n\n"
            "**Why it matters:** This causes `undefined symbol` errors when running GPU code. "
            "Your programs will crash immediately.\n\n"
            "**Technical details:** {issue}"
        ),
        "driver_too_old": (
            "**What's wrong:** Your NVIDIA GPU driver is too old for the version of PyTorch you're using.\n\n"
            "**Why it matters:** PyTorch cu124 requires Driver 550+, but you have an older driver. "
            "This prevents CUDA from working.\n\n"
            f"**Platform constraint:** {_get_runtime_constraint_note(runtime_version)}\n\n"
            "**Technical details:** {issue}"
        ),
        "mixed_cuda_versions": (
            "**What's wrong:** You have both CUDA 11 and CUDA 12 packages installed at the same time. "
            "This creates library conflicts.\n\n"
            "**Why it matters:** This causes `LD_LIBRARY_PATH` conflicts, segfaults, and symbol resolution errors. "
            "Your GPU code will be unstable.\n\n"
            "**Technical details:** {issue}"
        ),
        "torch_not_installed": (
            "**What's wrong:** PyTorch is not installed, but you're trying to use features that need it "
            "(like local AI model inference).\n\n"
            "**Why it matters:** Without PyTorch, you can't run GPU-accelerated AI models locally.\n\n"
            "**Note:** If you only need cloud-based inference (API calls), you can ignore this and disable "
            "local inference features.\n\n"
            "**Technical details:** {issue}"
        ),
        "torch_no_cuda_support": (
            "**What's wrong:** PyTorch is installed, but it was built for CPU-only (no CUDA support). "
            "This usually happens when installing from the default PyPI index.\n\n"
            "**Why it matters:** You can't use GPUs without the CUDA-enabled version of PyTorch.\n\n"
            "**How this happened:** You likely ran `pip install torch` without specifying the "
            "PyTorch index URL for CUDA builds.\n\n"
            "**Technical details:** {issue}"
        ),
        "cuda_libraries_missing": (
            "**What's wrong:** CUDA libraries are missing or incompatible. PyTorch can't find "
            "the CUDA libraries it needs.\n\n"
            "**Why it matters:** Even though PyTorch is installed with CUDA support, it can't "
            "access the GPU without these libraries.\n\n"
            "**Technical details:** {issue}"
        ),
        "no_gpu_device": (
            "**What's wrong:** No GPU was detected on your cluster. You're running on a CPU-only node.\n\n"
            "**Why it matters:** You can't run GPU workloads without a GPU!\n\n"
            "**How to check:** In Databricks, check your cluster configuration → "
            "Worker Type → Make sure it says 'GPU' or lists a GPU instance type (A100, A10G, T4).\n\n"
            "**Technical details:** {issue}"
        ),
        "torch_branch_incompatible": (
            "**What's wrong:** The PyTorch CUDA branch you're using doesn't match your Databricks runtime's CUDA version.\n\n"
            "**Why it matters:** Runtime 14.3 has CUDA 12.0 and cannot use PyTorch cu124 (which needs CUDA 12.4).\n\n"
            f"**Platform constraint:** {_get_runtime_constraint_note(runtime_version)}\n\n"
            "**Technical details:** {issue}"
        ),
    }

    template = explanations.get(
        root_cause,
        (
            "**What's wrong:** {issue}\n\n"
            "**Why it matters:** This will prevent your GPU workloads from running correctly."
        ),
    )

    return template.format(issue=issue, feature=feature or "your workload")


def _get_runtime_constraint_note(runtime_version: Optional[float]) -> str:
    """Get runtime-specific constraint note."""
    if not runtime_version:
        return "Check your Databricks runtime version for driver compatibility"

    if runtime_version == 14.3:
        return (
            "⚠️ **Runtime 14.3 has an IMMUTABLE Driver 535.** You CANNOT upgrade the driver. "
            "Your only options are: (1) downgrade PyTorch to cu120, or (2) upgrade to Runtime 15.2+"
        )
    elif runtime_version == 15.1:
        return "⚠️ **Runtime 15.1 has an IMMUTABLE Driver 550.** You CANNOT upgrade the driver."
    elif runtime_version >= 15.2:
        return (
            f"✅ Runtime {runtime_version} supports CUDA 12.4 with Driver 550. "
            "You can use PyTorch cu124."
        )
    else:
        return f"Runtime {runtime_version} detected. Check driver compatibility."


def _clean_fix_command(command: str) -> str:
    """Clean and format fix command for user display."""
    # Remove "Option 1:", "Option 2:", etc.
    command = command.strip()
    if command.startswith("Option"):
        parts = command.split(":", 1)
        if len(parts) > 1:
            command = parts[1].strip()

    # Format as code block if it's a command
    if any(
        cmd in command.lower() for cmd in ["pip install", "pip uninstall", "pip cache", "export"]
    ):
        # Already a command, add code formatting context
        if "pip install" in command:
            return f"**Install command:**\n   ```bash\n   {command}\n   ```"
        elif "pip uninstall" in command:
            return f"**Uninstall command:**\n   ```bash\n   {command}\n   ```"
        else:
            return f"**Run this command:**\n   ```bash\n   {command}\n   ```"
    else:
        # It's a description, return as-is
        return f"**{command}**"


def format_recommendations_for_notebook(
    blockers: List[Dict[str, Any]],
    runtime_version: Optional[float] = None,
    show_technical_details: bool = False,
) -> str:
    """
    Format recommendations specifically for Databricks notebook display.

    Similar to generate_recommendations but optimized for notebook output
    with better formatting and optional technical details.

    Args:
        blockers: List of blocker dictionaries
        runtime_version: Databricks runtime version
        show_technical_details: Whether to include technical error details

    Returns:
        Formatted text optimized for notebook display
    """
    if not blockers:
        return (
            "=" * 80
            + "\n✅ NO BLOCKERS DETECTED!\n"
            + "=" * 80
            + "\n\nYour environment is configured correctly.\n"
            "You can proceed with your GPU workloads.\n"
        )

    lines = []
    lines.append("=" * 80)
    lines.append("🚨 ACTION REQUIRED: CRITICAL ISSUES DETECTED")
    lines.append("=" * 80)
    lines.append("")

    for i, blocker in enumerate(blockers, 1):
        issue = blocker.get("issue") or blocker.get("message", "Unknown issue")
        root_cause = blocker.get("root_cause", "")
        feature = blocker.get("feature", "")
        fix_options = blocker.get("fix_options") or blocker.get("fix_commands", [])

        # Title
        title = _get_user_friendly_title(root_cause, issue, feature)
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Issue #{i}: {title}")
        lines.append(f"{'─' * 80}")

        # Simplified explanation for notebook
        if root_cause == "driver_too_old":
            lines.append("\n❌ Your GPU driver is too old for this PyTorch version.")
            if runtime_version == 14.3:
                lines.append("\n⚠️  Runtime 14.3 has IMMUTABLE Driver 535 - you CANNOT upgrade it.")
        elif root_cause == "torch_not_installed":
            lines.append("\n❌ PyTorch is not installed.")
        elif root_cause == "mixed_cuda_versions":
            lines.append("\n❌ You have both CUDA 11 and CUDA 12 packages - they conflict!")
        elif root_cause == "nvjitlink_mismatch":
            lines.append("\n❌ CUDA libraries don't match (cuBLAS/nvJitLink version mismatch).")
        elif root_cause == "torch_branch_incompatible":
            lines.append("\n❌ PyTorch CUDA branch incompatible with your runtime.")
        else:
            lines.append(f"\n❌ {issue}")

        # Fix options
        if fix_options:
            lines.append("\n🔧 How to Fix:")
            for j, option in enumerate(fix_options, 1):
                # Clean option text
                clean_opt = option.strip()
                if clean_opt.startswith("Option"):
                    parts = clean_opt.split(":", 1)
                    clean_opt = parts[1].strip() if len(parts) > 1 else clean_opt

                lines.append(f"\n   {j}. {clean_opt}")

        # Show technical details if requested
        if show_technical_details:
            lines.append(f"\n   📝 Technical: {issue}")

        lines.append("")

    lines.append("=" * 80)
    lines.append("\n💡 After fixing, restart Python: dbutils.library.restartPython()")
    lines.append("")

    return "\n".join(lines)
