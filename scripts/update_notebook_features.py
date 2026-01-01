#!/usr/bin/env python3
"""
Notebook Feature Sync Tool

Automatically updates the enhanced notebook when new features are added to the codebase.

Usage:
    python scripts/update_notebook_features.py

This script:
1. Detects new public API functions in cuda_healthcheck.databricks
2. Checks if they're used in the enhanced notebook
3. Suggests code snippets to add
4. Updates feature list in notebook header
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Set


def get_public_api_functions(module_path: Path) -> Set[str]:
    """Extract all public API functions from __init__.py."""
    init_file = module_path / "__init__.py"
    
    with open(init_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract __all__ list
    all_match = re.search(r'__all__\s*=\s*\[([^\]]+)\]', content, re.DOTALL)
    if not all_match:
        return set()
    
    # Parse exported names
    exports = re.findall(r'"([^"]+)"', all_match.group(1))
    return set(exports)


def check_notebook_usage(notebook_path: Path, functions: Set[str]) -> Dict[str, bool]:
    """Check which functions are used in the notebook."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_content = f.read()
    
    usage = {}
    for func in functions:
        # Check if function is imported or called
        usage[func] = (
            f"import {func}" in notebook_content or
            f"from cuda_healthcheck.databricks import {func}" in notebook_content or
            f"{func}(" in notebook_content
        )
    
    return usage


def get_feature_suggestions(unused_functions: List[str]) -> Dict[str, str]:
    """Generate code suggestions for unused features."""
    suggestions = {}
    
    # Define feature categories and suggested usage
    feature_docs = {
        "detect_databricks_runtime": """
# Detect Databricks runtime version
from cuda_healthcheck.databricks import detect_databricks_runtime

runtime_info = detect_databricks_runtime()
print(f"Runtime: {runtime_info['runtime_version']}")
print(f"CUDA: {runtime_info['cuda_version']}")
print(f"Is Immutable: {runtime_info.get('is_immutable', False)}")
""",
        "get_driver_version_for_runtime": """
# Get driver requirements for runtime
from cuda_healthcheck.databricks import get_driver_version_for_runtime

driver_info = get_driver_version_for_runtime(14.3)
print(f"Driver Range: {driver_info['driver_min_version']}-{driver_info['driver_max_version']}")
print(f"Immutable: {driver_info['is_immutable']}")
""",
        "check_driver_compatibility": """
# Check driver compatibility
from cuda_healthcheck.databricks import check_driver_compatibility

compatibility = check_driver_compatibility(14.3, 535)
if not compatibility['is_compatible']:
    print(f"❌ {compatibility['error_message']}")
else:
    print("✅ Driver compatible")
""",
        "get_runtime_info_summary": """
# Get human-readable runtime summary
from cuda_healthcheck.databricks import get_runtime_info_summary

summary = get_runtime_info_summary()
print(summary)
""",
    }
    
    for func in unused_functions:
        if func in feature_docs:
            suggestions[func] = feature_docs[func]
    
    return suggestions


def generate_report(
    all_functions: Set[str],
    usage: Dict[str, bool],
    suggestions: Dict[str, str]
) -> str:
    """Generate a markdown report."""
    used_functions = [f for f, used in usage.items() if used]
    unused_functions = [f for f, used in usage.items() if not used]
    
    report = []
    report.append("# 📊 Notebook Feature Sync Report\n")
    report.append(f"**Total Public API Functions:** {len(all_functions)}\n")
    report.append(f"**Used in Notebook:** {len(used_functions)}\n")
    report.append(f"**Unused in Notebook:** {len(unused_functions)}\n")
    report.append("")
    
    if used_functions:
        report.append("## ✅ Features Currently Used\n")
        for func in sorted(used_functions):
            report.append(f"- ✅ `{func}`")
        report.append("")
    
    if unused_functions:
        report.append("## ⚠️  Features NOT Used in Notebook\n")
        for func in sorted(unused_functions):
            report.append(f"- ⚠️  `{func}`")
        report.append("")
        
        report.append("## 💡 Suggested Code to Add\n")
        for func in sorted(unused_functions):
            if func in suggestions:
                report.append(f"### `{func}`\n")
                report.append("```python")
                report.append(suggestions[func].strip())
                report.append("```\n")
    else:
        report.append("## 🎉 All Features Integrated!\n")
        report.append("The notebook is up-to-date with all public API functions.\n")
    
    return "\n".join(report)


def main():
    """Main function."""
    # Paths
    project_root = Path(__file__).parent.parent
    databricks_module = project_root / "cuda_healthcheck" / "databricks"
    notebook_path = project_root / "notebooks" / "01_cuda_environment_validation_enhanced.py"
    
    print("Scanning for public API functions...")
    all_functions = get_public_api_functions(databricks_module)
    print(f"   Found {len(all_functions)} functions")
    
    print("\nChecking notebook usage...")
    usage = check_notebook_usage(notebook_path, all_functions)
    used_count = sum(1 for used in usage.values() if used)
    unused_count = len(usage) - used_count
    print(f"   Used: {used_count}, Unused: {unused_count}")
    
    # Generate suggestions
    unused_functions = [f for f, used in usage.items() if not used]
    suggestions = get_feature_suggestions(unused_functions)
    
    # Generate report
    report = generate_report(all_functions, usage, suggestions)
    
    # Save report
    report_path = project_root / "NOTEBOOK_FEATURE_SYNC_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Used Functions: {used_count}")
    print(f"Unused Functions: {unused_count}")
    
    if unused_count > 0:
        print(f"\nConsider adding these features to the notebook:")
        for func in sorted(unused_functions[:5]):  # Show first 5
            print(f"   - {func}")
    else:
        print(f"\nAll features are integrated into the notebook!")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

