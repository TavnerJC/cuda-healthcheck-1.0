#!/usr/bin/env python3
"""
CUDA Healthcheck Tool - Command Line Interface

Usage:
  python main.py detect          # Detect local CUDA environment
  python main.py healthcheck     # Run complete healthcheck with breaking changes
  python main.py scan            # Scan all Databricks GPU clusters
  python main.py breaking-changes [--library LIBRARY]  # View breaking changes
  python main.py export          # Export breaking changes to JSON
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cuda_detector import detect_cuda_environment
from src.healthcheck import run_complete_healthcheck
from src.databricks_api import scan_clusters
from src.data import get_breaking_changes, BreakingChangesDatabase


def cmd_detect(args):
    """Detect local CUDA environment."""
    print("🔍 Detecting CUDA environment...")
    results = detect_cuda_environment()
    print(json.dumps(results, indent=2))


def cmd_healthcheck(args):
    """Run complete healthcheck."""
    print("🏥 Running complete CUDA healthcheck...")
    results = run_complete_healthcheck()

    print("\n" + "=" * 80)
    print("CUDA HEALTHCHECK RESULTS")
    print("=" * 80)

    # Print CUDA info
    cuda_env = results["cuda_environment"]
    print(f"\n📊 CUDA Environment:")
    print(f"  Runtime Version: {cuda_env['cuda_runtime_version']}")
    print(f"  Driver Version:  {cuda_env['cuda_driver_version']}")
    print(f"  NVCC Version:    {cuda_env['nvcc_version']}")
    print(f"  GPUs:            {len(cuda_env['gpus'])}")

    for gpu in cuda_env["gpus"]:
        print(
            f"    - {gpu['name']} (Compute {gpu['compute_capability']}, {gpu['memory_total_mb']} MB)"
        )

    # Print library info
    print(f"\n📚 Installed Libraries:")
    for lib in results["libraries"]:
        status = "✅" if lib["is_compatible"] else "❌"
        print(f"  {status} {lib['name']}: {lib['version']}")
        if lib["cuda_version"]:
            print(f"      CUDA: {lib['cuda_version']}")
        if lib["warnings"]:
            for warning in lib["warnings"]:
                print(f"      ⚠️  {warning}")

    # Print compatibility analysis
    analysis = results["compatibility_analysis"]
    print(f"\n🎯 Compatibility Analysis:")
    print(f"  Score: {analysis['compatibility_score']}/100")
    print(f"  Critical Issues: {analysis['critical_issues']}")
    print(f"  Warnings: {analysis['warning_issues']}")
    print(f"  Info: {analysis['info_issues']}")
    print(f"\n💡 {analysis['recommendation']}")

    # Print breaking changes
    if analysis["critical_issues"] > 0:
        print(f"\n🚨 CRITICAL BREAKING CHANGES:")
        for change in analysis["breaking_changes"]["CRITICAL"]:
            print(f"\n  [{change['affected_library']}] {change['title']}")
            print(f"  {change['description'][:100]}...")

    if analysis["warning_issues"] > 0:
        print(f"\n⚠️  WARNINGS:")
        for change in analysis["breaking_changes"]["WARNING"]:
            print(f"  - [{change['affected_library']}] {change['title']}")

    print("\n" + "=" * 80)

    # Save to file
    output_file = f"healthcheck-{results['healthcheck_id']}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Full results saved to: {output_file}")


def cmd_scan(args):
    """Scan Databricks clusters."""
    print("🔍 Scanning Databricks GPU clusters...")
    print("This may take several minutes depending on the number of clusters.\n")

    try:
        results = scan_clusters(save_to_delta=not args.no_delta)

        summary = results["summary"]
        print("\n" + "=" * 80)
        print("CLUSTER SCAN SUMMARY")
        print("=" * 80)
        print(f"\nTotal Clusters Scanned: {summary['total_clusters']}")
        print(f"Successful Scans: {summary['successful_scans']}")
        print(f"Failed Scans: {summary['failed_scans']}")
        print(f"\nCUDA Versions Found:")
        for version, count in summary["cuda_versions"].items():
            print(f"  - CUDA {version}: {count} clusters")
        print(f"\nTotal Breaking Changes: {summary['total_breaking_changes']}")
        print(f"Total Warnings: {summary['total_warnings']}")

        if not args.no_delta:
            print("\n✅ Results saved to Delta table: main.cuda_healthcheck.healthcheck_results")

        # Save to file
        output_file = "cluster-scan-results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Full results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Error scanning clusters: {str(e)}")
        print("\nMake sure you have set:")
        print("  - DATABRICKS_HOST")
        print("  - DATABRICKS_TOKEN")
        print("  - DATABRICKS_WAREHOUSE_ID (for Delta table operations)")
        sys.exit(1)


def cmd_breaking_changes(args):
    """View breaking changes."""
    print("📋 CUDA Breaking Changes Database\n")

    changes = get_breaking_changes(library=args.library)

    if args.library:
        print(f"Breaking changes for: {args.library}")
    else:
        print("All breaking changes")

    print(f"Total: {len(changes)}\n")

    for change in changes:
        severity_icon = {"CRITICAL": "🚨", "WARNING": "⚠️", "INFO": "ℹ️"}.get(change["severity"], "")

        print(f"{severity_icon} [{change['severity']}] {change['title']}")
        print(f"   Library: {change['affected_library']}")
        print(f"   CUDA: {change['cuda_version_from']} → {change['cuda_version_to']}")
        print(f"   {change['description'][:150]}...")
        print()


def cmd_export(args):
    """Export breaking changes to JSON."""
    db = BreakingChangesDatabase()
    output_file = args.output or "breaking_changes.json"
    db.export_to_json(output_file)
    print(f"✅ Breaking changes exported to: {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CUDA Healthcheck Tool for Databricks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py detect
  python main.py healthcheck
  python main.py scan
  python main.py breaking-changes --library pytorch
  python main.py export --output my_changes.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # detect command
    parser_detect = subparsers.add_parser("detect", help="Detect local CUDA environment")
    parser_detect.set_defaults(func=cmd_detect)

    # healthcheck command
    parser_healthcheck = subparsers.add_parser("healthcheck", help="Run complete healthcheck")
    parser_healthcheck.set_defaults(func=cmd_healthcheck)

    # scan command
    parser_scan = subparsers.add_parser("scan", help="Scan Databricks clusters")
    parser_scan.add_argument("--no-delta", action="store_true", help="Don't save to Delta table")
    parser_scan.set_defaults(func=cmd_scan)

    # breaking-changes command
    parser_breaking = subparsers.add_parser("breaking-changes", help="View breaking changes")
    parser_breaking.add_argument("--library", help="Filter by library name")
    parser_breaking.set_defaults(func=cmd_breaking_changes)

    # export command
    parser_export = subparsers.add_parser("export", help="Export breaking changes to JSON")
    parser_export.add_argument("--output", "-o", help="Output file path")
    parser_export.set_defaults(func=cmd_export)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()



