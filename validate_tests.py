"""
Test Validation Script

Run this script to validate that all unit tests can run locally
without any dependencies on Databricks or CUDA hardware.

Usage:
    python validate_tests.py
"""

import sys
import subprocess
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print(f"✓ {description} - PASSED")
        return True
    else:
        print(f"✗ {description} - FAILED")
        return False


def main():
    print_header("CUDA HEALTHCHECK - Test Validation")
    
    # Check we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Must run from cuda-healthcheck directory")
        print(f"Current directory: {os.getcwd()}")
        return 1
    
    print("✓ Running from correct directory")
    print(f"  Location: {os.getcwd()}\n")
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    if sys.version_info < (3, 10):
        print("⚠️ Warning: Python 3.10+ recommended")
    print()
    
    # Check pytest is installed
    try:
        import pytest
        print(f"✓ pytest version: {pytest.__version__}")
    except ImportError:
        print("❌ pytest not installed")
        print("   Install with: pip install pytest pytest-cov")
        return 1
    
    results = []
    
    # Test 1: Run utils tests
    print_header("Test Suite 1: Utilities (logging, retry, exceptions)")
    results.append(run_command(
        ["pytest", "tests/test_logging.py", "-v"],
        "Logging tests"
    ))
    results.append(run_command(
        ["pytest", "tests/test_retry.py", "-v"],
        "Retry tests"
    ))
    results.append(run_command(
        ["pytest", "tests/test_exceptions.py", "-v"],
        "Exception tests"
    ))
    
    # Test 2: Run orchestrator tests
    print_header("Test Suite 2: HealthcheckOrchestrator")
    results.append(run_command(
        ["pytest", "tests/test_orchestrator.py", "-v"],
        "Orchestrator tests"
    ))
    
    # Test 3: Run breaking changes tests
    print_header("Test Suite 3: Breaking Changes Database")
    results.append(run_command(
        ["pytest", "tests/test_breaking_changes.py", "-v"],
        "Breaking changes tests"
    ))
    
    # Test 4: Run databricks integration tests (with mocks)
    print_header("Test Suite 4: Databricks Integration (Mocked)")
    results.append(run_command(
        ["pytest", "tests/databricks/", "-v"],
        "Databricks integration tests"
    ))
    
    # Test 5: Run all tests with coverage
    print_header("Test Suite 5: Complete Test Run with Coverage")
    results.append(run_command(
        ["pytest", "tests/", "-v", "--cov=src", "--cov-report=term-missing"],
        "All tests with coverage"
    ))
    
    # Summary
    print_header("TEST VALIDATION SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(results)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print()
    
    if failed_tests == 0:
        print("🎉 ALL TESTS PASSED!")
        print()
        print("✓ All unit tests can run locally without Databricks or CUDA")
        print("✓ Tests use mocks and fixtures for external dependencies")
        print("✓ Code is ready for development and CI/CD")
        return 0
    else:
        print(f"❌ {failed_tests} test suite(s) failed")
        print()
        print("Please review the test output above and fix any failures.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


