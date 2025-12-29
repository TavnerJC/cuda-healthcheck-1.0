"""
Databricks Healthchecker - Wrapper for CUDA healthcheck in Databricks notebooks.

Provides a high-level interface for running CUDA healthchecks on Databricks
clusters and storing results in Delta tables.
"""

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..cuda_detector.detector import CUDADetector, CUDAEnvironment
from ..data.breaking_changes import BreakingChangesDatabase
from ..utils.logging_config import get_databricks_logger
from .connector import DatabricksConnector, is_databricks_environment

logger = get_databricks_logger(__name__)

# Try to import dbutils (only available in Databricks)
try:
    # In Databricks, dbutils is automatically available
    dbutils = dbutils  # type: ignore
    HAS_DBUTILS = True
except NameError:
    # Mock for local development
    from unittest.mock import MagicMock

    dbutils = MagicMock()
    HAS_DBUTILS = False
    logger.warning("dbutils not available - using mock (local development mode)")


@dataclass
class HealthcheckResult:
    """Results from a complete healthcheck run."""

    healthcheck_id: str
    cluster_id: Optional[str]
    cluster_name: Optional[str]
    timestamp: str
    cuda_environment: Dict[str, Any]
    compatibility_analysis: Dict[str, Any]
    status: str  # "healthy", "warning", "critical"
    recommendations: List[str]


class DatabricksHealthchecker:
    """
    High-level wrapper for running CUDA healthchecks in Databricks.

    Combines CUDA detection, breaking change analysis, and Delta table
    storage into a single easy-to-use interface.

    Example:
        ```python
        # In a Databricks notebook
        from src.databricks import DatabricksHealthchecker

        checker = DatabricksHealthchecker()
        result = checker.run_healthcheck()
        checker.export_results_to_delta("main.cuda.healthcheck_results")
        ```
    """

    def __init__(
        self,
        cluster_id: Optional[str] = None,
        connector: Optional[DatabricksConnector] = None,
    ):
        """
        Initialize the Databricks healthchecker.

        Args:
            cluster_id: Specific cluster ID to check (optional)
            connector: DatabricksConnector instance (optional, will create if not provided)
        """
        self.cluster_id = cluster_id
        self.connector = connector
        self.cuda_detector = CUDADetector()
        self.breaking_changes_db = BreakingChangesDatabase()
        self.last_result: Optional[HealthcheckResult] = None

        logger.info("DatabricksHealthchecker initialized")

    def get_cluster_cuda_version(self) -> Optional[str]:
        """
        Get CUDA version from the current cluster.

        Returns:
            CUDA version string or None if not detected.

        Example:
            ```python
            cuda_version = checker.get_cluster_cuda_version()
            print(f"Cluster CUDA version: {cuda_version}")
            ```
        """
        try:
            environment = self.cuda_detector.detect_environment()
            return environment.cuda_driver_version or environment.cuda_runtime_version
        except Exception as e:
            logger.error(f"Failed to detect CUDA version: {e}")
            return None

    def get_cluster_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the current Databricks cluster.

        Returns:
            Dictionary with cluster information.

        Example:
            ```python
            metadata = checker.get_cluster_metadata()
            print(f"Running on: {metadata['cluster_name']}")
            ```
        """
        metadata: Dict[str, Any] = {
            "is_databricks": is_databricks_environment(),
            "cluster_id": None,
            "cluster_name": None,
            "spark_version": None,
            "runtime_version": os.getenv("DATABRICKS_RUNTIME_VERSION"),
        }

        # Try to get cluster info from connector
        if self.connector and self.cluster_id:
            try:
                cluster_info = self.connector.get_cluster_info(self.cluster_id)
                if cluster_info is not None:
                    metadata.update(
                        {
                            "cluster_id": cluster_info.cluster_id,
                            "cluster_name": cluster_info.cluster_name,
                            "spark_version": cluster_info.spark_version,
                            "node_type": cluster_info.node_type_id,
                            "num_workers": cluster_info.num_workers,
                        }
                    )
            except Exception as e:
                logger.warning(f"Could not get cluster info: {e}")

        # Try to get cluster ID from Spark context
        if HAS_DBUTILS:
            try:
                notebook_context = (
                    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
                )
                metadata["cluster_id"] = notebook_context.tags().get("clusterId")
                metadata["cluster_name"] = notebook_context.tags().get("clusterName")
            except Exception as e:
                logger.debug(f"Could not get cluster ID from context: {e}")

        return metadata

    def run_healthcheck(self) -> Dict[str, Any]:
        """
        Run a complete CUDA healthcheck on the current cluster.

        Performs:
        1. CUDA environment detection
        2. Breaking change analysis
        3. Compatibility scoring
        4. Recommendation generation

        Returns:
            Dictionary with complete healthcheck results.

        Example:
            ```python
            result = checker.run_healthcheck()
            print(f"Status: {result['status']}")
            print(f"Compatibility Score: {result['compatibility_analysis']['compatibility_score']}")
            ```
        """
        logger.info("Starting CUDA healthcheck...")

        try:
            # Step 1: Detect CUDA environment
            logger.info("Detecting CUDA environment...")
            environment = self.cuda_detector.detect_environment()

            # Step 2: Get cluster metadata
            logger.info("Gathering cluster metadata...")
            cluster_metadata = self.get_cluster_metadata()

            # Step 3: Analyze compatibility
            logger.info("Analyzing compatibility...")
            cuda_version = (
                environment.cuda_driver_version
                or environment.cuda_runtime_version
                or "Unknown"
            )

            compute_capability = None
            if environment.gpus:
                compute_capability = environment.gpus[0].compute_capability

            compatibility = self.breaking_changes_db.score_compatibility(
                detected_libraries=[
                    {
                        "name": lib.name,
                        "version": lib.version,
                        "cuda_version": lib.cuda_version,
                        "is_compatible": lib.is_compatible,
                        "warnings": lib.warnings,
                    }
                    for lib in environment.libraries
                ],
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

            # Step 6: Create result
            healthcheck_id = (
                f"healthcheck-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            )

            result = HealthcheckResult(
                healthcheck_id=healthcheck_id,
                cluster_id=cluster_metadata.get("cluster_id"),
                cluster_name=cluster_metadata.get("cluster_name"),
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
                    "libraries": [
                        {
                            "name": lib.name,
                            "version": lib.version,
                            "cuda_version": lib.cuda_version,
                            "is_compatible": lib.is_compatible,
                            "warnings": lib.warnings,
                        }
                        for lib in environment.libraries
                    ],
                },
                compatibility_analysis=compatibility,
                status=status,
                recommendations=recommendations,
            )

            self.last_result = result
            logger.info(f"Healthcheck complete. Status: {status}")

            return asdict(result)

        except Exception as e:
            logger.error(f"Healthcheck failed: {e}", exc_info=True)
            return {
                "healthcheck_id": (
                    f"healthcheck-error-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
                ),
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _generate_recommendations(
        self, environment: CUDAEnvironment, compatibility: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on healthcheck results."""
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

        # Check each library
        for lib in environment.libraries:
            if not lib.is_compatible:
                recommendations.append(
                    f"❌ {lib.name} is not CUDA-compatible - check installation"
                )
            if lib.warnings:
                for warning in lib.warnings:
                    recommendations.append(f"⚠️ {lib.name}: {warning}")

        # Check GPU compute capability
        if environment.gpus:
            gpu = environment.gpus[0]
            cc = float(gpu.compute_capability)
            if cc < 7.0:
                recommendations.append(
                    f"⚠️ GPU compute capability {gpu.compute_capability} is outdated - "
                    "consider upgrading to newer GPU"
                )
            elif cc >= 9.0:
                recommendations.append(
                    f"✓ Latest GPU detected ({gpu.name}, CC {gpu.compute_capability})"
                )

        # General recommendations
        if compatibility["compatibility_score"] >= 90:
            recommendations.append("✓ Environment is healthy and well-configured")
        elif compatibility["compatibility_score"] >= 70:
            recommendations.append(
                "📋 Review warnings and test thoroughly before production"
            )
        else:
            recommendations.append(
                "🔧 Significant compatibility issues detected - migration recommended"
            )

        return recommendations

    def export_results_to_delta(
        self,
        table_path: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Export healthcheck results to a Delta table.

        Args:
            table_path: Full table path (e.g., "main.cuda.healthcheck_results")
            result: Optional result dict (uses last result if not provided)

        Returns:
            True if export successful, False otherwise.

        Example:
            ```python
            result = checker.run_healthcheck()
            success = checker.export_results_to_delta("main.cuda.healthcheck")
            ```
        """
        if result is None:
            if self.last_result is None:
                logger.error("No healthcheck results to export")
                return False
            result = asdict(self.last_result)

        try:
            # In Databricks, we can use Spark directly
            if HAS_DBUTILS and is_databricks_environment():
                from pyspark.sql import SparkSession

                spark = SparkSession.builder.getOrCreate()

                # Convert result to Spark DataFrame
                df = spark.createDataFrame([result])

                # Write to Delta table
                df.write.format("delta").mode("append").saveAsTable(table_path)

                logger.info(f"Results exported to Delta table: {table_path}")
                return True
            else:
                # Use connector for non-Databricks environments
                if not self.connector:
                    logger.warning("No connector available for Delta export")
                    return False

                self.connector.write_delta_table(table_path, [result])
                return True

        except Exception as e:
            logger.error(f"Failed to export to Delta table: {e}", exc_info=True)
            return False

    def display_results(self, result: Optional[Dict[str, Any]] = None) -> None:
        """
        Display healthcheck results in a formatted way (for notebooks).

        Args:
            result: Optional result dict (uses last result if not provided)
        """
        if result is None:
            if self.last_result is None:
                print("No healthcheck results to display")
                return
            result = asdict(self.last_result)

        print("=" * 80)
        print("CUDA HEALTHCHECK RESULTS")
        print("=" * 80)
        print(f"Healthcheck ID: {result.get('healthcheck_id')}")
        print(f"Timestamp: {result.get('timestamp')}")
        print(f"Status: {result.get('status', 'unknown').upper()}")
        print()

        # Display CUDA environment
        cuda_env = result.get("cuda_environment", {})
        print("CUDA Environment:")
        print(f"  Runtime Version: {cuda_env.get('cuda_runtime_version', 'N/A')}")
        print(f"  Driver Version: {cuda_env.get('cuda_driver_version', 'N/A')}")
        print(f"  NVCC Version: {cuda_env.get('nvcc_version', 'N/A')}")
        print()

        # Display GPUs
        gpus = cuda_env.get("gpus", [])
        print(f"GPUs ({len(gpus)}):")
        for gpu in gpus:
            print(f"  [{gpu.get('gpu_index')}] {gpu.get('name')}")
            print(f"      Compute Capability: {gpu.get('compute_capability')}")
            print(f"      Memory: {gpu.get('memory_total_mb')} MB")
        print()

        # Display libraries
        libraries = cuda_env.get("libraries", [])
        print(f"Libraries ({len(libraries)}):")
        for lib in libraries:
            status_icon = "✓" if lib.get("is_compatible") else "✗"
            print(f"  {status_icon} {lib.get('name')}: {lib.get('version')}")
            if lib.get("cuda_version"):
                print(f"      CUDA Version: {lib.get('cuda_version')}")
        print()

        # Display compatibility
        compat = result.get("compatibility_analysis", {})
        print("Compatibility Analysis:")
        print(f"  Score: {compat.get('compatibility_score', 0)}/100")
        print(f"  Critical Issues: {compat.get('critical_issues', 0)}")
        print(f"  Warnings: {compat.get('warning_issues', 0)}")
        print()

        # Display recommendations
        recommendations = result.get("recommendations", [])
        print(f"Recommendations ({len(recommendations)}):")
        for rec in recommendations:
            print(f"  • {rec}")
        print("=" * 80)


def get_healthchecker(
    cluster_id: Optional[str] = None,
) -> DatabricksHealthchecker:
    """
    Factory function to get a configured healthchecker.

    Args:
        cluster_id: Optional cluster ID to check

    Returns:
        DatabricksHealthchecker instance.

    Example:
        ```python
        checker = get_healthchecker()
        result = checker.run_healthcheck()
        ```
    """
    connector = None

    # Only create connector if we have credentials
    if os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"):
        try:
            connector = DatabricksConnector()
        except Exception as e:
            logger.warning(f"Could not create connector: {e}")

    return DatabricksHealthchecker(cluster_id=cluster_id, connector=connector)
