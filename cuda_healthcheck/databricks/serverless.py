"""
Serverless Compute Detection and GPU Discovery.

This module provides serverless-aware GPU detection that works on both
Classic ML Runtime clusters and Serverless GPU Compute.
"""

import os
import socket
import subprocess
from typing import Any, Dict, List, Optional

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def is_serverless_environment() -> bool:
    """
    Detect if running on Databricks Serverless Compute.

    Serverless compute has limitations:
    - No direct SparkContext access
    - Single-user execution model
    - GPUs directly available to process

    Returns:
        True if running on serverless, False otherwise
    """
    # Check runtime version for serverless indicator
    runtime = os.getenv("DATABRICKS_RUNTIME_VERSION", "")
    if "serverless" in runtime.lower():
        logger.info("Detected Databricks Serverless environment")
        return True

    # Check for serverless-specific environment variables
    if os.getenv("DATABRICKS_SERVERLESS_COMPUTE"):
        logger.info("Detected Databricks Serverless via env var")
        return True

    # Try to detect by checking if SparkContext is accessible
    try:
        from pyspark import SparkContext

        if hasattr(SparkContext, "_active_spark_context"):
            sc = SparkContext._active_spark_context
            if sc is None:
                logger.info("No active SparkContext - likely serverless")
                return True
    except Exception as e:
        logger.debug(f"SparkContext check failed: {e}")

    logger.info("Detected Classic cluster environment")
    return False


def detect_gpu_direct() -> Dict[str, Any]:
    """
    Direct GPU detection without Spark (for serverless or local).

    This method runs nvidia-smi directly on the current process
    without using Spark distributed execution.

    Returns:
        Dictionary with GPU information and detection status
    """
    logger.info("Running direct GPU detection (serverless-compatible)")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap,uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for i, line in enumerate(result.stdout.strip().split("\n")):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append(
                        {
                            "gpu_index": i,
                            "name": parts[0],
                            "driver_version": parts[1],
                            "memory_total": parts[2],
                            "compute_capability": parts[3],
                            "uuid": parts[4],
                        }
                    )

            logger.info(f"Detected {len(gpus)} GPU(s) via direct detection")

            return {
                "success": True,
                "method": "direct",
                "hostname": socket.gethostname(),
                "gpu_count": len(gpus),
                "gpus": gpus,
                "error": None,
            }
        else:
            logger.warning("nvidia-smi returned no GPU data")
            return {
                "success": False,
                "method": "direct",
                "hostname": socket.gethostname(),
                "gpu_count": 0,
                "gpus": [],
                "error": "nvidia-smi returned no data",
            }

    except FileNotFoundError:
        logger.error("nvidia-smi command not found")
        return {
            "success": False,
            "method": "direct",
            "hostname": socket.gethostname(),
            "gpu_count": 0,
            "gpus": [],
            "error": "nvidia-smi not found in PATH",
        }
    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi command timed out")
        return {
            "success": False,
            "method": "direct",
            "hostname": socket.gethostname(),
            "gpu_count": 0,
            "gpus": [],
            "error": "nvidia-smi command timed out",
        }
    except Exception as e:
        logger.error(f"GPU detection failed: {e}")
        return {
            "success": False,
            "method": "direct",
            "hostname": socket.gethostname(),
            "gpu_count": 0,
            "gpus": [],
            "error": str(e),
        }


def detect_gpu_distributed() -> Dict[str, Any]:
    """
    Distributed GPU detection using Spark (for classic clusters).

    This method uses Spark to run GPU detection on all worker nodes
    and deduplicates results by GPU UUID.

    Returns:
        Dictionary with aggregated GPU information across cluster
    """
    logger.info("Running distributed GPU detection (classic cluster)")

    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext

        def check_gpu_worker(_):
            """Run GPU check on worker node."""
            import socket
            import subprocess

            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total,compute_cap,uuid",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0 and result.stdout.strip():
                    gpus = []
                    for line in result.stdout.strip().split("\n"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 5:
                            gpus.append(
                                {
                                    "name": parts[0],
                                    "driver": parts[1],
                                    "memory": parts[2],
                                    "compute_cap": parts[3],
                                    "uuid": parts[4],
                                }
                            )

                    return {
                        "hostname": socket.gethostname(),
                        "has_gpu": True,
                        "gpu_count": len(gpus),
                        "gpus": gpus,
                        "error": None,
                    }
                else:
                    return {
                        "hostname": socket.gethostname(),
                        "has_gpu": False,
                        "gpu_count": 0,
                        "gpus": [],
                        "error": "nvidia-smi returned no data",
                    }
            except Exception as e:
                return {
                    "hostname": socket.gethostname(),
                    "has_gpu": False,
                    "gpu_count": 0,
                    "gpus": [],
                    "error": str(e),
                }

        num_partitions = max(sc.defaultParallelism, 2)
        worker_results = (
            sc.parallelize(range(num_partitions), num_partitions).map(check_gpu_worker).collect()
        )

        # Deduplicate by hostname + GPU UUID
        unique_gpus = {}
        worker_nodes = {}

        for result in worker_results:
            hostname = result["hostname"]
            if result["has_gpu"]:
                for gpu in result["gpus"]:
                    key = f"{hostname}_{gpu['uuid']}"
                    if key not in unique_gpus:
                        unique_gpus[key] = {"hostname": hostname, "gpu": gpu}
                        if hostname not in worker_nodes:
                            worker_nodes[hostname] = []
                        worker_nodes[hostname].append(gpu)

        logger.info(f"Detected {len(unique_gpus)} unique GPU(s) across {len(worker_nodes)} node(s)")

        return {
            "success": True,
            "method": "distributed",
            "total_executors": len(worker_results),
            "worker_node_count": len(worker_nodes),
            "physical_gpu_count": len(unique_gpus),
            "worker_nodes": worker_nodes,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Distributed GPU detection failed: {e}")
        return {
            "success": False,
            "method": "distributed",
            "error": str(e),
        }


def detect_gpu_auto() -> Dict[str, Any]:
    """
    Automatically detect GPUs using the appropriate method.

    This function detects the environment (serverless vs classic)
    and uses the correct GPU detection method.

    Returns:
        Dictionary with GPU information and detection metadata.

        Standard keys (present in all return dictionaries):
        - success (bool): True if detection succeeded
        - method (str): 'direct' or 'distributed'
        - environment (str): 'serverless' or 'classic'
        - gpu_count (int): Number of GPUs detected
        - error (str or None): Error message if detection failed

        Additional keys for serverless (direct detection):
        - hostname (str): Current host name
        - gpus (list): List of GPU dictionaries with detailed info

        Additional keys for classic (distributed detection):
        - total_executors (int): Number of Spark executors
        - worker_node_count (int): Number of unique worker nodes
        - physical_gpu_count (int): Deduplicated GPU count
        - worker_nodes (dict): Mapping of hostnames to GPU lists
    """
    serverless = is_serverless_environment()

    logger.info(f"Environment: {'Serverless' if serverless else 'Classic Cluster'}")

    if serverless:
        result = detect_gpu_direct()
        result["environment"] = "serverless"
        # Ensure gpu_count is present (already is, but for consistency)
        if "gpu_count" not in result:
            result["gpu_count"] = len(result.get("gpus", []))
    else:
        result = detect_gpu_distributed()
        result["environment"] = "classic"
        # Ensure gpu_count is present (use physical_gpu_count for classic)
        if "gpu_count" not in result:
            result["gpu_count"] = result.get("physical_gpu_count", 0)

    return result
