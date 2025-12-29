#!/usr/bin/env python3
"""
Quick Start Example for CUDA Healthcheck Tool

This script demonstrates all the main features of the CUDA healthcheck tool.
Run this to test the tool after installation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def example_1_local_detection():
    """Example 1: Detect CUDA environment on local machine."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Local CUDA Detection")
    print("=" * 80)

    from cuda_healthcheck.cuda_detector import detect_cuda_environment

    print("\n🔍 Detecting CUDA environment...")
    env = detect_cuda_environment()

    print(f"\nCUDA Driver Version: {env.get('cuda_driver_version', 'Not detected')}")
    print(f"CUDA Runtime Version: {env.get('cuda_runtime_version', 'Not detected')}")
    print(f"NVCC Version: {env.get('nvcc_version', 'Not detected')}")
    print(f"Number of GPUs: {len(env.get('gpus', []))}")

    for gpu in env.get("gpus", []):
        print(
            f"  - {gpu['name']} (Compute {gpu['compute_capability']}, {gpu['memory_total_mb']} MB)"
        )

    print("\n✅ Example 1 complete!")


def example_2_complete_healthcheck():
    """Example 2: Run complete healthcheck with breaking change analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Complete Healthcheck")
    print("=" * 80)

    from cuda_healthcheck.healthcheck import run_complete_healthcheck

    print("\n🏥 Running complete healthcheck...")
    results = run_complete_healthcheck()

    analysis = results["compatibility_analysis"]

    print(f"\n📊 Compatibility Score: {analysis['compatibility_score']}/100")
    print(f"   Critical Issues: {analysis['critical_issues']}")
    print(f"   Warnings: {analysis['warning_issues']}")
    print(f"   Info: {analysis['info_issues']}")

    print(f"\n💡 Recommendation:")
    print(f"   {analysis['recommendation']}")

    if analysis["critical_issues"] > 0:
        print(f"\n🚨 CRITICAL ISSUES FOUND:")
        for change in analysis["breaking_changes"]["CRITICAL"][:3]:  # Show first 3
            print(f"   - {change['title']}")

    print("\n✅ Example 2 complete!")


def example_3_breaking_changes():
    """Example 3: Query breaking changes database."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Breaking Changes Database")
    print("=" * 80)

    from cuda_healthcheck.data import get_breaking_changes

    print("\n📋 Querying breaking changes for PyTorch...")
    pytorch_changes = get_breaking_changes(library="pytorch")

    print(f"\nFound {len(pytorch_changes)} PyTorch-related breaking changes:")

    for change in pytorch_changes:
        severity_icon = {"CRITICAL": "🚨", "WARNING": "⚠️", "INFO": "ℹ️"}.get(change["severity"], "")
        print(f"\n{severity_icon} [{change['severity']}] {change['title']}")
        print(f"   CUDA: {change['cuda_version_from']} → {change['cuda_version_to']}")
        print(f"   {change['description'][:100]}...")

    print("\n✅ Example 3 complete!")


def example_4_compatibility_scoring():
    """Example 4: Score compatibility for a specific scenario."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Compatibility Scoring")
    print("=" * 80)

    from cuda_healthcheck.data import score_compatibility

    # Scenario: PyTorch 2.1.0 with CUDA 12.4 trying to upgrade to CUDA 13.0
    print("\n🎯 Scenario: Upgrading from CUDA 12.4 to 13.0 with PyTorch 2.1.0")

    libraries = [
        {"name": "pytorch", "version": "2.1.0", "cuda_version": "12.1"},
        {"name": "tensorflow", "version": "2.15.0", "cuda_version": "12.2"},
    ]

    score = score_compatibility(
        detected_libraries=libraries, cuda_version="13.0", compute_capability="8.0"
    )

    print(f"\n📊 Compatibility Score: {score['compatibility_score']}/100")
    print(f"   Total Issues: {score['total_issues']}")
    print(f"   Critical: {score['critical_issues']}")
    print(f"   Warnings: {score['warning_issues']}")

    print(f"\n💡 {score['recommendation']}")

    print("\n✅ Example 4 complete!")


def example_5_databricks_info():
    """Example 5: Information about Databricks integration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Databricks Integration (Info Only)")
    print("=" * 80)

    print("\n📝 To scan Databricks clusters, you need:")
    print("   1. DATABRICKS_HOST environment variable")
    print("   2. DATABRICKS_TOKEN (Personal Access Token)")
    print("   3. DATABRICKS_WAREHOUSE_ID (for Delta table operations)")

    print("\n🔧 Setup:")
    print("   export DATABRICKS_HOST='https://your-workspace.cloud.databricks.com'")
    print("   export DATABRICKS_TOKEN='dapi...'")
    print("   export DATABRICKS_WAREHOUSE_ID='abc123...'")

    print("\n🚀 Then run:")
    print("   python main.py scan")

    print("\n📊 Results will be:")
    print("   - Displayed in console")
    print("   - Saved to cluster-scan-results.json")
    print("   - Stored in Delta table: main.cuda_healthcheck.healthcheck_results")

    print("\n✅ Example 5 complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("CUDA HEALTHCHECK TOOL - QUICK START EXAMPLES")
    print("=" * 80)
    print("\nThis script demonstrates the main features of the CUDA Healthcheck Tool.")
    print("Note: Some examples require NVIDIA GPU and drivers to be installed.")

    try:
        example_1_local_detection()
    except Exception as e:
        print(f"\n⚠️  Example 1 failed: {str(e)}")
        print("   (This is expected if NVIDIA GPU/drivers are not installed)")

    try:
        example_2_complete_healthcheck()
    except Exception as e:
        print(f"\n⚠️  Example 2 failed: {str(e)}")
        print("   (This is expected if NVIDIA GPU/drivers are not installed)")

    try:
        example_3_breaking_changes()
    except Exception as e:
        print(f"\n⚠️  Example 3 failed: {str(e)}")

    try:
        example_4_compatibility_scoring()
    except Exception as e:
        print(f"\n⚠️  Example 4 failed: {str(e)}")

    try:
        example_5_databricks_info()
    except Exception as e:
        print(f"\n⚠️  Example 5 failed: {str(e)}")

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 80)
    print("\n📚 Next steps:")
    print("   - Read README.md for full documentation")
    print("   - Check docs/SETUP.md for detailed setup instructions")
    print("   - Review docs/MIGRATION_GUIDE.md for CUDA migration guidance")
    print("   - Run 'python main.py --help' to see all CLI commands")
    print("\n🚀 Happy healthchecking!")


if __name__ == "__main__":
    main()
