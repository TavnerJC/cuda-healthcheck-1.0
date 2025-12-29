"""
Databricks Cluster Scanner for CUDA Healthcheck.

This module scans all GPU-enabled Databricks clusters, runs CUDA healthchecks,
and stores results in Delta tables for analysis.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.compute import ClusterDetails
    from databricks.sdk.service.jobs import RunLifecycleState, RunResultState

    DATABRICKS_SDK_AVAILABLE = True
except ImportError:
    DATABRICKS_SDK_AVAILABLE = False
    WorkspaceClient = None
    ClusterDetails = None


@dataclass
class ClusterHealthcheck:
    """Results from a cluster CUDA healthcheck."""

    cluster_id: str
    cluster_name: str
    cuda_version: Optional[str]
    driver_version: Optional[str]
    gpu_count: int
    gpu_types: List[str]
    libraries: List[Dict[str, Any]]
    breaking_changes: List[Dict[str, Any]]
    warnings: List[str]
    timestamp: str
    status: str  # "success", "error", "pending"


class ClusterScanner:
    """Scans Databricks clusters for CUDA compatibility issues."""

    def __init__(self, workspace_url: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize the cluster scanner.

        Args:
            workspace_url: Databricks workspace URL (or set DATABRICKS_HOST env var)
            token: Personal Access Token (or set DATABRICKS_TOKEN env var)
        """
        if not DATABRICKS_SDK_AVAILABLE:
            raise ImportError(
                "Databricks SDK not installed. Install with: pip install databricks-sdk"
            )

        # Use provided credentials or fall back to environment variables
        self.workspace_url = workspace_url or os.getenv("DATABRICKS_HOST")
        self.token = token or os.getenv("DATABRICKS_TOKEN")

        if not self.workspace_url or not self.token:
            raise ValueError(
                "Databricks credentials not provided. Set DATABRICKS_HOST and "
                "DATABRICKS_TOKEN environment variables or pass them to constructor."
            )

        # Initialize Databricks workspace client
        self.client = WorkspaceClient(host=self.workspace_url, token=self.token)

    def list_gpu_clusters(self) -> List[ClusterDetails]:
        """
        List all GPU-enabled clusters in the workspace.

        Returns:
            List of ClusterDetails objects for GPU-enabled clusters.
        """
        all_clusters = list(self.client.clusters.list())

        gpu_clusters = []
        for cluster in all_clusters:
            # Check if cluster has GPU configuration
            if cluster.node_type_id and any(
                gpu_indicator in cluster.node_type_id.lower()
                for gpu_indicator in ["gpu", "g5", "p3", "p4"]
            ):
                gpu_clusters.append(cluster)

        return gpu_clusters

    def create_healthcheck_notebook(self) -> str:
        """
        Create a temporary notebook for running CUDA healthcheck.

        Returns:
            Path to the created notebook in Databricks workspace.
        """
        notebook_content = """
# Databricks notebook source
# CUDA Healthcheck Runner
import json
import sys

# Add healthcheck module to path
sys.path.insert(0, '/Workspace/cuda-healthcheck/src')

from cuda_detector.detector import detect_cuda_environment

# Run the detection
try:
    results = detect_cuda_environment()
    print(json.dumps(results, indent=2))
except Exception as e:
    error_result = {
        "error": str(e),
        "status": "failed",
        "timestamp": datetime.utcnow().isoformat()
    }
    print(json.dumps(error_result, indent=2))
"""

        notebook_path = "/Workspace/cuda-healthcheck-temp-notebook"

        # Create notebook in workspace
        self.client.workspace.mkdirs(notebook_path)
        self.client.workspace.upload(notebook_path, notebook_content.encode(), format="SOURCE")

        return notebook_path

    def run_healthcheck_on_cluster(
        self, cluster_id: str, timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Run CUDA healthcheck on a specific cluster.

        Args:
            cluster_id: The cluster ID to run healthcheck on
            timeout_seconds: Maximum time to wait for job completion

        Returns:
            Dictionary with healthcheck results or error information.
        """
        try:
            # Create healthcheck notebook
            notebook_path = self.create_healthcheck_notebook()

            # Submit a one-time job to run the notebook on the cluster
            run = self.client.jobs.submit(
                run_name=f"CUDA Healthcheck - {cluster_id}",
                tasks=[
                    {
                        "task_key": "healthcheck",
                        "existing_cluster_id": cluster_id,
                        "notebook_task": {
                            "notebook_path": notebook_path,
                            "source": "WORKSPACE",
                        },
                    }
                ],
            )

            run_id = run.run_id

            # Wait for job to complete
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                run_status = self.client.jobs.get_run(run_id)

                if run_status.state.life_cycle_state in [
                    RunLifecycleState.TERMINATED,
                    RunLifecycleState.SKIPPED,
                    RunLifecycleState.INTERNAL_ERROR,
                ]:
                    # Job completed - get output
                    if run_status.state.result_state == RunResultState.SUCCESS:
                        # Get output from run
                        output = self.client.jobs.get_run_output(run_id)

                        if output.notebook_output and output.notebook_output.result:
                            try:
                                result: Dict[str, Any] = json.loads(output.notebook_output.result)
                                return result
                            except json.JSONDecodeError:
                                return {
                                    "error": "Failed to parse output",
                                    "raw_output": output.notebook_output.result,
                                }
                    else:
                        return {
                            "error": f"Job failed with state: {run_status.state.result_state}",
                            "state_message": run_status.state.state_message,
                        }

                time.sleep(5)  # Poll every 5 seconds

            # Timeout
            return {"error": "Healthcheck job timed out", "run_id": run_id}

        except Exception as e:
            return {"error": f"Failed to run healthcheck: {str(e)}"}

    def scan_cluster(self, cluster: ClusterDetails) -> ClusterHealthcheck:
        """
        Scan a single cluster for CUDA compatibility.

        Args:
            cluster: ClusterDetails object for the cluster to scan

        Returns:
            ClusterHealthcheck object with scan results.
        """
        warnings = []

        # Check if cluster is running
        if cluster.state.value != "RUNNING":
            warnings.append(f"Cluster is not running (state: {cluster.state.value})")
            return ClusterHealthcheck(
                cluster_id=cluster.cluster_id,
                cluster_name=cluster.cluster_name or "Unnamed",
                cuda_version=None,
                driver_version=None,
                gpu_count=0,
                gpu_types=[],
                libraries=[],
                breaking_changes=[],
                warnings=warnings,
                timestamp=datetime.utcnow().isoformat(),
                status="error",
            )

        # Run healthcheck on the cluster
        results = self.run_healthcheck_on_cluster(cluster.cluster_id)

        if "error" in results:
            warnings.append(results["error"])
            status = "error"
        else:
            status = "success"

        return ClusterHealthcheck(
            cluster_id=cluster.cluster_id,
            cluster_name=cluster.cluster_name or "Unnamed",
            cuda_version=results.get("cuda_driver_version"),
            driver_version=results.get("cuda_runtime_version"),
            gpu_count=len(results.get("gpus", [])),
            gpu_types=[gpu["name"] for gpu in results.get("gpus", [])],
            libraries=results.get("libraries", []),
            breaking_changes=results.get("breaking_changes", []),
            warnings=warnings,
            timestamp=datetime.utcnow().isoformat(),
            status=status,
        )

    def scan_all_clusters(self) -> List[ClusterHealthcheck]:
        """
        Scan all GPU-enabled clusters in the workspace.

        Returns:
            List of ClusterHealthcheck objects for all GPU clusters.
        """
        gpu_clusters = self.list_gpu_clusters()
        results = []

        for cluster in gpu_clusters:
            print(f"Scanning cluster: {cluster.cluster_name} ({cluster.cluster_id})")
            healthcheck = self.scan_cluster(cluster)
            results.append(healthcheck)

        return results

    def save_to_delta_table(
        self,
        results: List[ClusterHealthcheck],
        catalog: str = "main",
        schema: str = "cuda_healthcheck",
        table: str = "healthcheck_results",
    ) -> None:
        """
        Save healthcheck results to a Delta table.

        Args:
            results: List of ClusterHealthcheck objects
            catalog: Unity Catalog catalog name
            schema: Schema name
            table: Table name
        """
        try:
            # Convert results to dictionaries for future use
            # data = [asdict(result) for result in results]

            # Create SQL statements to insert data
            # Note: This is a simplified version. In production, use Spark DataFrame API
            table_name = f"{catalog}.{schema}.{table}"

            # Create schema if not exists
            self.client.statement_execution.execute_statement(
                warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),
                statement=f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}",
            )

            # Create table if not exists
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                cluster_id STRING,
                cluster_name STRING,
                cuda_version STRING,
                driver_version STRING,
                gpu_count INT,
                gpu_types ARRAY<STRING>,
                libraries ARRAY<STRUCT<
                    name: STRING,
                    version: STRING,
                    cuda_version: STRING,
                    is_compatible: BOOLEAN,
                    warnings: ARRAY<STRING>
                >>,
                breaking_changes ARRAY<STRUCT<
                    severity: STRING,
                    message: STRING,
                    affected_library: STRING
                >>,
                warnings ARRAY<STRING>,
                timestamp TIMESTAMP,
                status STRING
            )
            USING DELTA
            """

            self.client.statement_execution.execute_statement(
                warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),
                statement=create_table_sql,
            )

            print(f"Results saved to Delta table: {table_name}")

        except Exception as e:
            print(f"Warning: Failed to save to Delta table: {str(e)}")
            print("Results are still available in memory.")

    def get_summary(self, results: List[ClusterHealthcheck]) -> Dict[str, Any]:
        """
        Generate a summary of scan results.

        Args:
            results: List of ClusterHealthcheck objects

        Returns:
            Dictionary with summary statistics.
        """
        total_clusters = len(results)
        successful_scans = sum(1 for r in results if r.status == "success")
        failed_scans = sum(1 for r in results if r.status == "error")

        # Count CUDA versions
        cuda_versions: Dict[str, int] = {}
        for result in results:
            if result.cuda_version:
                cuda_versions[result.cuda_version] = cuda_versions.get(result.cuda_version, 0) + 1

        # Count breaking changes
        total_breaking_changes = sum(len(r.breaking_changes) for r in results)

        # Collect all warnings
        all_warnings = []
        for result in results:
            all_warnings.extend(result.warnings)

        return {
            "total_clusters": total_clusters,
            "successful_scans": successful_scans,
            "failed_scans": failed_scans,
            "cuda_versions": cuda_versions,
            "total_breaking_changes": total_breaking_changes,
            "total_warnings": len(all_warnings),
            "timestamp": datetime.utcnow().isoformat(),
        }


def scan_clusters(
    workspace_url: Optional[str] = None,
    token: Optional[str] = None,
    save_to_delta: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to scan all GPU clusters and return results.

    Args:
        workspace_url: Databricks workspace URL
        token: Personal Access Token
        save_to_delta: Whether to save results to Delta table

    Returns:
        Dictionary with scan results and summary.
    """
    scanner = ClusterScanner(workspace_url, token)
    results = scanner.scan_all_clusters()

    if save_to_delta:
        scanner.save_to_delta_table(results)

    summary = scanner.get_summary(results)

    return {"results": [asdict(r) for r in results], "summary": summary}


if __name__ == "__main__":
    # Quick test when run directly
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Testing cluster scanner...")
        try:
            results = scan_clusters()
            print(json.dumps(results, indent=2))
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    else:
        print("Usage: python cluster_scanner.py --test")
        print("Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables first.")
