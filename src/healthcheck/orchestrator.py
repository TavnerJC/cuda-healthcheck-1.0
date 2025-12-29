"""
Healthcheck Orchestrator.

Coordinates CUDA detection, breaking change analysis, and reporting.
Provides both a class-based orchestrator and a simple function interface.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..cuda_detector.detector import CUDADetector, CUDAEnvironment
from ..data.breaking_changes import BreakingChangesDatabase
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HealthcheckReport:
    """Complete healthcheck report with all analysis results."""

    healthcheck_id: str
    timestamp: str
    cuda_environment: Dict[str, Any]
    compatibility_analysis: Dict[str, Any]
    status: str  # "healthy", "warning", "critical"
    recommendations: List[str]


class HealthcheckOrchestrator:
    """
    Orchestrates CUDA healthcheck workflow.

    Coordinates CUDA detection, library compatibility analysis,
    breaking change detection, and report generation.

    Example:
        ```python
        orchestrator = HealthcheckOrchestrator()
        report = orchestrator.generate_report()
        print(f"Status: {report.status}")
        print(f"Score: {report.compatibility_analysis['compatibility_score']}")
        ```
    """

    def __init__(self) -> None:
        """Initialize the healthcheck orchestrator."""
        self.detector = CUDADetector()
        self.breaking_changes_db = BreakingChangesDatabase()
        self.last_environment: Optional[CUDAEnvironment] = None
        self.last_report: Optional[HealthcheckReport] = None
        logger.info("HealthcheckOrchestrator initialized")

    def check_compatibility(
        self,
        local_version: str,
        cluster_version: str,
    ) -> Dict[str, Any]:
        """
        Check compatibility between local and cluster CUDA versions.

        Args:
            local_version: Local CUDA version (e.g., "12.4")
            cluster_version: Cluster CUDA version (e.g., "12.6")

        Returns:
            Dictionary with compatibility information and breaking changes.

        Example:
            ```python
            result = orchestrator.check_compatibility("12.4", "12.6")
            if result['has_breaking_changes']:
                print("Warning: Breaking changes detected!")
            ```
        """
        logger.info(f"Checking compatibility: {local_version} -> {cluster_version}")

        # Get breaking changes for this transition
        breaking_changes = self.breaking_changes_db.get_changes_by_cuda_transition(
            local_version, cluster_version
        )

        # Categorize by severity
        critical = [c for c in breaking_changes if c.severity == "CRITICAL"]
        warnings = [c for c in breaking_changes if c.severity == "WARNING"]
        info = [c for c in breaking_changes if c.severity == "INFO"]

        compatible = len(critical) == 0

        return {
            "local_version": local_version,
            "cluster_version": cluster_version,
            "compatible": compatible,
            "has_breaking_changes": len(breaking_changes) > 0,
            "breaking_changes": {
                "critical": [asdict(c) for c in critical],
                "warning": [asdict(c) for c in warnings],
                "info": [asdict(c) for c in info],
            },
            "total_changes": len(breaking_changes),
            "recommendation": (
                "Compatible - no critical issues"
                if compatible
                else "Incompatible - resolve critical issues before deployment"
            ),
        }

    def analyze_breaking_changes(
        self,
        detected_libraries: List[Dict[str, Any]],
        cuda_version: str,
        compute_capability: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze breaking changes for detected environment.

        Args:
            detected_libraries: List of library information dictionaries
            cuda_version: Detected CUDA version
            compute_capability: GPU compute capability

        Returns:
            Dictionary with breaking change analysis and compatibility score.

        Example:
            ```python
            libraries = [{"name": "pytorch", "version": "2.0", ...}]
            analysis = orchestrator.analyze_breaking_changes(libraries, "13.0", "9.0")
            ```
        """
        logger.info(f"Analyzing breaking changes for CUDA {cuda_version}")

        return self.breaking_changes_db.score_compatibility(
            detected_libraries=detected_libraries,
            cuda_version=cuda_version,
            compute_capability=compute_capability,
        )

    def generate_report(self) -> HealthcheckReport:
        """
        Generate a complete healthcheck report.

        Performs full CUDA detection, compatibility analysis,
        and generates recommendations.

        Returns:
            HealthcheckReport object with all results.

        Example:
            ```python
            report = orchestrator.generate_report()
            orchestrator.save_report_json(report, "healthcheck_report.json")
            ```
        """
        logger.info("Generating healthcheck report...")

        # Step 1: Detect CUDA environment
        logger.info("Detecting CUDA environment...")
        environment = self.detector.detect_environment()
        self.last_environment = environment

        # Step 2: Prepare library information
        detected_libraries = [
            {
                "name": lib.name,
                "version": lib.version,
                "cuda_version": lib.cuda_version,
                "is_compatible": lib.is_compatible,
                "warnings": lib.warnings,
            }
            for lib in environment.libraries
        ]

        # Step 3: Analyze compatibility
        logger.info("Analyzing compatibility...")
        cuda_version = (
            environment.cuda_driver_version or environment.cuda_runtime_version or "Unknown"
        )

        compute_capability = None
        if environment.gpus:
            compute_capability = environment.gpus[0].compute_capability

        compatibility = self.analyze_breaking_changes(
            detected_libraries=detected_libraries,
            cuda_version=cuda_version,
            compute_capability=compute_capability,
        )

        # Step 4: Determine status
        if compatibility["critical_issues"] > 0:
            status = "critical"
        elif compatibility["warning_issues"] > 0:
            status = "warning"
        else:
            status = "healthy"

        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(environment, compatibility)

        # Step 6: Create report
        healthcheck_id = f"healthcheck-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        report = HealthcheckReport(
            healthcheck_id=healthcheck_id,
            timestamp=environment.timestamp,
            cuda_environment={
                "cuda_runtime_version": environment.cuda_runtime_version,
                "cuda_driver_version": environment.cuda_driver_version,
                "nvcc_version": environment.nvcc_version,
                "gpus": [
                    {
                        "name": gpu.name,
                        "driver_version": gpu.driver_version,
                        "cuda_version": gpu.cuda_version,
                        "compute_capability": gpu.compute_capability,
                        "memory_total_mb": gpu.memory_total_mb,
                        "gpu_index": gpu.gpu_index,
                    }
                    for gpu in environment.gpus
                ],
                "libraries": detected_libraries,
            },
            compatibility_analysis=compatibility,
            status=status,
            recommendations=recommendations,
        )

        self.last_report = report
        logger.info(f"Report generated. Status: {status}")

        return report

    def _generate_recommendations(
        self, environment: CUDAEnvironment, compatibility: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Check for critical issues
        if compatibility["critical_issues"] > 0:
            recommendations.append(
                "⚠️ CRITICAL: Address breaking changes before deploying to production"
            )

        # Check CUDA version
        cuda_version = environment.cuda_driver_version or "Unknown"
        if "13." in cuda_version:
            recommendations.append(
                "📌 CUDA 13.x detected - ensure all libraries support this version"
            )
        elif "12.4" in cuda_version:
            recommendations.append("✓ CUDA 12.4 - stable and well-supported version")

        # Check each library
        for lib in environment.libraries:
            if not lib.is_compatible:
                recommendations.append(f"❌ {lib.name} is not CUDA-compatible - check installation")
            if lib.warnings:
                for warning in lib.warnings:
                    recommendations.append(f"⚠️ {lib.name}: {warning}")

        # Check GPU compute capability
        if environment.gpus:
            gpu = environment.gpus[0]
            try:
                cc = float(gpu.compute_capability)
                if cc < 7.0:
                    recommendations.append(
                        f"⚠️ GPU compute capability {gpu.compute_capability} is outdated"
                    )
                elif cc >= 9.0:
                    recommendations.append(
                        f"✓ Latest GPU detected ({gpu.name}, CC {gpu.compute_capability})"
                    )
            except ValueError:
                logger.warning(f"Could not parse compute capability: {gpu.compute_capability}")

        # General recommendations
        score = compatibility["compatibility_score"]
        if score >= 90:
            recommendations.append("✓ Environment is healthy and well-configured")
        elif score >= 70:
            recommendations.append("📋 Review warnings and test thoroughly before production")
        else:
            recommendations.append("🔧 Significant compatibility issues - migration recommended")

        return recommendations

    def save_report_json(
        self,
        report: Optional[HealthcheckReport] = None,
        filepath: str = "healthcheck_report.json",
    ) -> None:
        """
        Save healthcheck report to JSON file.

        Args:
            report: Optional report (uses last report if not provided)
            filepath: Path to output file
        """
        if report is None:
            if self.last_report is None:
                logger.error("No report to save")
                return
            report = self.last_report

        try:
            with open(filepath, "w") as f:
                json.dump(asdict(report), f, indent=2)
            logger.info(f"Report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def print_report_summary(self, report: Optional[HealthcheckReport] = None) -> None:
        """
        Print a summary of the healthcheck report.

        Args:
            report: Optional report (uses last report if not provided)
        """
        if report is None:
            if self.last_report is None:
                print("No report available")
                return
            report = self.last_report

        print("\n" + "=" * 80)
        print("CUDA HEALTHCHECK SUMMARY")
        print("=" * 80)
        print(f"Healthcheck ID: {report.healthcheck_id}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Status: {report.status.upper()}")
        print()

        cuda_env = report.cuda_environment
        print("CUDA Environment:")
        print(f"  Runtime: {cuda_env.get('cuda_runtime_version', 'N/A')}")
        print(f"  Driver: {cuda_env.get('cuda_driver_version', 'N/A')}")
        print(f"  NVCC: {cuda_env.get('nvcc_version', 'N/A')}")
        print()

        compat = report.compatibility_analysis
        print("Compatibility:")
        print(f"  Score: {compat.get('compatibility_score', 0)}/100")
        print(f"  Critical: {compat.get('critical_issues', 0)}")
        print(f"  Warnings: {compat.get('warning_issues', 0)}")
        print()

        print(f"Recommendations ({len(report.recommendations)}):")
        for rec in report.recommendations:
            print(f"  • {rec}")
        print("=" * 80 + "\n")


def run_complete_healthcheck() -> Dict[str, Any]:
    """
    Convenience function to run a complete healthcheck.

    Returns:
        Dictionary with complete healthcheck results.

    Example:
        ```python
        from src.healthcheck import run_complete_healthcheck

        result = run_complete_healthcheck()
        print(json.dumps(result, indent=2))
        ```
    """
    orchestrator = HealthcheckOrchestrator()
    report = orchestrator.generate_report()
    return asdict(report)


if __name__ == "__main__":
    # Run healthcheck and print results
    orchestrator = HealthcheckOrchestrator()
    report = orchestrator.generate_report()
    orchestrator.print_report_summary(report)
    orchestrator.save_report_json(report, "healthcheck_report.json")
