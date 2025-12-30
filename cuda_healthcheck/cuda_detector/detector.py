"""
CUDA Environment Detector for Databricks Clusters.

This module detects CUDA versions, GPU properties, and library compatibility
on Databricks GPU-enabled clusters.
"""

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging_config import get_logger
from ..utils.validation import check_command_available, safe_int_conversion

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """GPU information structure."""

    name: str
    driver_version: str
    cuda_version: str
    compute_capability: str
    memory_total_mb: int
    gpu_index: int


@dataclass
class LibraryInfo:
    """Installed library version information."""

    name: str
    version: str
    cuda_version: Optional[str]
    is_compatible: bool
    warnings: List[str]


@dataclass
class CUDAEnvironment:
    """Complete CUDA environment information."""

    cuda_runtime_version: Optional[str]
    cuda_driver_version: Optional[str]
    nvcc_version: Optional[str]
    gpus: List[GPUInfo]
    libraries: List[LibraryInfo]
    breaking_changes: List[Dict[str, Any]]
    timestamp: str


class CUDADetector:
    """Detects CUDA environment and library compatibility on Databricks clusters."""

    def __init__(self) -> None:
        """Initialize the CUDA detector."""
        self.cuda_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-13.0",
        ]
        logger.info("CUDADetector initialized")

    def detect_nvidia_smi(self) -> Dict[str, Any]:
        """
        Detect GPU information using nvidia-smi.

        Returns:
            Dictionary containing driver version, CUDA version, and GPU details.
            Returns error dict if detection fails.

        Raises:
            CudaDetectionError: If critical detection error occurs
        """
        # Check if nvidia-smi is available
        if not check_command_available("nvidia-smi"):
            error_msg = "nvidia-smi command not found in PATH"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "details": "CUDA toolkit may not be installed or not in PATH",
            }

        try:
            # Get driver and CUDA version with timeout
            logger.debug("Running nvidia-smi to detect GPU information...")
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name,memory.total,compute_cap,index",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                error_msg = f"nvidia-smi returned non-zero exit code: {result.returncode}"
                logger.error(f"{error_msg}, stderr: {result.stderr}")
                return {"error": error_msg, "details": result.stderr, "success": False}

            # Parse nvidia-smi version output for CUDA version
            logger.debug("Parsing CUDA version from nvidia-smi...")
            version_result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )

            cuda_version = None
            if version_result.returncode == 0:
                # Extract CUDA version from header (e.g., "CUDA Version: 12.4")
                match = re.search(r"CUDA Version:\s+(\d+\.\d+)", version_result.stdout)
                if match:
                    cuda_version = match.group(1)
                    logger.info(f"Detected CUDA version: {cuda_version}")
                else:
                    logger.warning("Could not parse CUDA version from nvidia-smi output")

            # Parse GPU information with enhanced error handling
            gpus = []
            for line_num, line in enumerate(result.stdout.strip().split("\n"), 1):
                if not line.strip():
                    continue

                try:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 5:
                        logger.warning(
                            f"Line {line_num}: Expected 5 fields, got {len(parts)}. Skipping."
                        )
                        continue

                    # Safely parse memory and index with error handling
                    try:
                        memory_mb = safe_int_conversion(
                            float(parts[2]) if parts[2] else 0, default=0
                        )
                    except ValueError:
                        logger.warning(f"Line {line_num}: Could not parse memory: {parts[2]}")
                        memory_mb = 0

                    try:
                        gpu_index = safe_int_conversion(parts[4], default=line_num - 1)
                    except ValueError:
                        logger.warning(f"Line {line_num}: Could not parse GPU index: {parts[4]}")
                        gpu_index = line_num - 1

                    gpu_info = GPUInfo(
                        name=parts[1] or "Unknown GPU",
                        driver_version=parts[0] or "Unknown",
                        cuda_version=cuda_version or "Unknown",
                        compute_capability=parts[3] or "Unknown",
                        memory_total_mb=memory_mb,
                        gpu_index=gpu_index,
                    )
                    gpus.append(gpu_info)
                    logger.debug(f"Detected GPU: {gpu_info.name} (Index {gpu_index})")

                except Exception as e:
                    logger.error(f"Line {line_num}: Failed to parse GPU info: {e}")
                    continue

            if not gpus:
                logger.warning("No GPUs detected from nvidia-smi output")
                return {
                    "error": "No GPUs detected",
                    "success": False,
                    "details": "nvidia-smi ran but returned no GPU information",
                }

            logger.info(f"Successfully detected {len(gpus)} GPU(s)")
            return {
                "driver_version": gpus[0].driver_version if gpus else None,
                "cuda_version": cuda_version,
                "gpus": gpus,
                "success": True,
            }

        except FileNotFoundError:
            error_msg = "nvidia-smi command not found"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "details": "nvidia-smi executable not found in system PATH",
            }
        except subprocess.TimeoutExpired:
            error_msg = "nvidia-smi command timed out after 10 seconds"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "details": "GPU may be unresponsive or system is overloaded",
            }
        except PermissionError:
            error_msg = "Permission denied when running nvidia-smi"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "details": "Check user permissions for GPU access",
            }
        except Exception as e:
            error_msg = f"Unexpected error during nvidia-smi detection: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg, "success": False, "details": str(e)}

    def detect_cuda_runtime(self) -> Optional[str]:
        """
        Detect CUDA runtime version from /usr/local/cuda.

        Returns:
            CUDA runtime version string or None if not found.
        """
        logger.debug("Attempting to detect CUDA runtime version...")

        for cuda_path in self.cuda_paths:
            try:
                version_file = Path(cuda_path) / "version.json"
                version_txt = Path(cuda_path) / "version.txt"

                # Try version.json first (newer CUDA versions)
                if version_file.exists():
                    try:
                        logger.debug(f"Found {version_file}, attempting to parse...")
                        with open(version_file, "r") as f:
                            version_data = json.load(f)
                            runtime_version = version_data.get("cuda", {}).get("version")
                            if runtime_version:
                                logger.info(
                                    f"Detected CUDA runtime from version.json: {runtime_version}"
                                )
                                return str(runtime_version)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse {version_file}: {e}")
                    except IOError as e:
                        logger.warning(f"Failed to read {version_file}: {e}")

                # Try version.txt (older CUDA versions)
                if version_txt.exists():
                    try:
                        logger.debug(f"Found {version_txt}, attempting to parse...")
                        with open(version_txt, "r") as f:
                            content = f.read().strip()
                            match = re.search(r"CUDA Version\s+(\d+\.\d+\.\d+)", content)
                            if match:
                                runtime_version = match.group(1)
                                logger.info(
                                    f"Detected CUDA runtime from version.txt: {runtime_version}"
                                )
                                return runtime_version
                    except IOError as e:
                        logger.warning(f"Failed to read {version_txt}: {e}")

                # Check if this CUDA path exists and extract version from path
                cuda_path_obj = Path(cuda_path)
                if cuda_path_obj.exists() and cuda_path_obj.is_dir():
                    match = re.search(r"cuda-(\d+\.\d+)", cuda_path)
                    if match:
                        runtime_version = match.group(1)
                        logger.info(f"Detected CUDA runtime from path: {runtime_version}")
                        return runtime_version

            except PermissionError:
                logger.warning(f"Permission denied accessing {cuda_path}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error checking {cuda_path}: {e}")
                continue

        logger.warning("Could not detect CUDA runtime version from any known location")
        return None

    def detect_nvcc_version(self) -> Optional[str]:
        """
        Detect nvcc (CUDA compiler) version.

        Returns:
            nvcc version string or None if not found.
        """
        if not check_command_available("nvcc"):
            logger.debug("nvcc command not available in PATH")
            return None

        try:
            logger.debug("Running nvcc --version...")
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                # Extract version from output (e.g., "release 12.4, V12.4.131")
                match = re.search(r"release (\d+\.\d+)", result.stdout)
                if match:
                    nvcc_version = match.group(1)
                    logger.info(f"Detected nvcc version: {nvcc_version}")
                    return nvcc_version
                else:
                    logger.warning("nvcc command succeeded but could not parse version")
            else:
                logger.warning(f"nvcc command failed with exit code {result.returncode}")

            return None

        except FileNotFoundError:
            logger.debug("nvcc executable not found")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("nvcc command timed out after 5 seconds")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error running nvcc: {e}")
            return None

    def detect_pytorch(self) -> LibraryInfo:
        """
        Detect PyTorch installation and CUDA compatibility.

        Returns:
            LibraryInfo object with PyTorch details.
        """
        warnings = []

        try:
            logger.debug("Attempting to import PyTorch...")
            import torch

            version = torch.__version__
            logger.info(f"PyTorch version {version} detected")

            try:
                cuda_available = torch.cuda.is_available()
                cuda_version = torch.version.cuda if cuda_available else None

                if not cuda_available:
                    warning_msg = "PyTorch CUDA not available - CPU-only build detected"
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)
                else:
                    logger.info(f"PyTorch CUDA version: {cuda_version}")

                # Check for version compatibility issues
                if cuda_version:
                    try:
                        cuda_major = int(cuda_version.split(".")[0])
                        if cuda_major >= 13:
                            warning_msg = "PyTorch may need rebuild for CUDA 13.x compatibility"
                            warnings.append(warning_msg)
                            logger.warning(warning_msg)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse CUDA major version: {e}")

            except Exception as e:
                warnings.append(f"Error checking CUDA availability: {str(e)}")
                logger.error(f"Error checking PyTorch CUDA: {e}", exc_info=True)
                cuda_available = False
                cuda_version = None

            return LibraryInfo(
                name="pytorch",
                version=version,
                cuda_version=cuda_version,
                is_compatible=cuda_available,
                warnings=warnings,
            )

        except ImportError:
            logger.info("PyTorch not installed")
            return LibraryInfo(
                name="pytorch",
                version="Not installed",
                cuda_version=None,
                is_compatible=False,
                warnings=["PyTorch not installed"],
            )
        except Exception as e:
            error_msg = f"Unexpected error detecting PyTorch: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return LibraryInfo(
                name="pytorch",
                version="Error",
                cuda_version=None,
                is_compatible=False,
                warnings=[error_msg],
            )

    def detect_tensorflow(self) -> LibraryInfo:
        """
        Detect TensorFlow installation and CUDA compatibility.

        Returns:
            LibraryInfo object with TensorFlow details.
        """
        warnings = []

        try:
            import tensorflow as tf

            version = tf.__version__

            # Check GPU availability
            gpus = tf.config.list_physical_devices("GPU")
            cuda_available = len(gpus) > 0

            if not cuda_available:
                warnings.append("TensorFlow GPU not available")

            # TensorFlow doesn't expose CUDA version directly
            # Check build info for CUDA version
            cuda_version = None
            try:
                build_info = tf.sysconfig.get_build_info()
                cuda_version = build_info.get("cuda_version")
            except Exception:
                pass

            return LibraryInfo(
                name="tensorflow",
                version=version,
                cuda_version=cuda_version,
                is_compatible=cuda_available,
                warnings=warnings,
            )

        except ImportError:
            return LibraryInfo(
                name="tensorflow",
                version="Not installed",
                cuda_version=None,
                is_compatible=False,
                warnings=["TensorFlow not installed"],
            )

    def detect_cudf(self) -> LibraryInfo:
        """
        Detect cuDF (RAPIDS) installation and CUDA compatibility.

        Returns:
            LibraryInfo object with cuDF details.
        """
        warnings = []

        try:
            import cudf

            version = cudf.__version__

            # cuDF version is tightly coupled to CUDA version
            # RAPIDS 24.12+ requires CUDA 12.x
            major_version = int(version.split(".")[0])

            if major_version >= 24:
                warnings.append("cuDF 24.x+ requires CUDA 12.x or higher")

            return LibraryInfo(
                name="cudf",
                version=version,
                cuda_version=None,  # cuDF doesn't expose this directly
                is_compatible=True,
                warnings=warnings,
            )

        except ImportError:
            return LibraryInfo(
                name="cudf",
                version="Not installed",
                cuda_version=None,
                is_compatible=False,
                warnings=["cuDF not installed"],
            )

    def detect_cuopt(self) -> LibraryInfo:
        """
        Detect NVIDIA CuOPT installation and check nvJitLink compatibility.
        
        CuOPT 25.12+ requires nvidia-nvjitlink-cu12>=12.9.79 but Databricks
        ML Runtime 16.4 provides 12.4.127, causing compatibility issues.

        Returns:
            LibraryInfo object with CuOPT details and compatibility warnings.
        """
        warnings = []

        try:
            import cuopt

            version = cuopt.__version__

            # Check if CuOPT can actually load (test libcuopt.so)
            cuda_version = None
            load_successful = False
            
            try:
                # Try to import routing module (loads libcuopt.so)
                from cuopt import routing
                
                # Try to create a simple DataModel (validates library loading)
                test_model = routing.DataModel(2, 1)
                load_successful = True
                logger.info(f"CuOPT {version} loaded successfully")
                
                # Try to detect CUDA version from dependencies
                try:
                    import nvidia.cuda_runtime
                    cuda_version = nvidia.cuda_runtime.__version__
                except Exception:
                    pass
                    
            except RuntimeError as e:
                error_msg = str(e)
                
                # Check for specific nvJitLink version mismatch
                if "nvJitLink" in error_msg or "undefined symbol" in error_msg:
                    warnings.append(
                        "CRITICAL: CuOPT failed to load due to nvJitLink version mismatch"
                    )
                    warnings.append(
                        "CuOPT 25.12+ requires nvidia-nvjitlink-cu12>=12.9.79"
                    )
                    
                    # Try to detect installed nvJitLink version
                    try:
                        result = subprocess.run(
                            ["pip", "show", "nvidia-nvjitlink-cu12"],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                if line.startswith('Version:'):
                                    nvjitlink_version = line.split(':')[1].strip()
                                    warnings.append(
                                        f"Detected nvidia-nvjitlink-cu12 version: {nvjitlink_version}"
                                    )
                                    
                                    # Check if it's the problematic 12.4 version
                                    if nvjitlink_version.startswith('12.4'):
                                        warnings.append(
                                            "ERROR: Databricks ML Runtime provides nvJitLink 12.4.x"
                                        )
                                        warnings.append(
                                            "This is incompatible with CuOPT 25.12+ (requires 12.9+)"
                                        )
                                        warnings.append(
                                            "Users CANNOT upgrade nvJitLink in managed Databricks runtimes"
                                        )
                                        warnings.append(
                                            "Report to: https://github.com/databricks-industry-solutions/routing/issues"
                                        )
                    except Exception:
                        pass
                        
                    logger.error(f"CuOPT library load failed: {error_msg}")
                else:
                    warnings.append(f"CuOPT library load error: {error_msg}")
                    
            except Exception as e:
                warnings.append(f"CuOPT verification failed: {str(e)}")
                logger.error(f"CuOPT verification failed: {str(e)}")

            return LibraryInfo(
                name="cuopt",
                version=version,
                cuda_version=cuda_version,
                is_compatible=load_successful,
                warnings=warnings,
            )

        except ImportError:
            return LibraryInfo(
                name="cuopt",
                version="Not installed",
                cuda_version=None,
                is_compatible=False,
                warnings=["CuOPT not installed"],
            )
        except Exception as e:
            error_msg = f"Unexpected error detecting CuOPT: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return LibraryInfo(
                name="cuopt",
                version="Error",
                cuda_version=None,
                is_compatible=False,
                warnings=[error_msg],
            )

    def detect_all_libraries(self) -> List[LibraryInfo]:
        """
        Detect all supported libraries and their CUDA compatibility.

        Returns:
            List of LibraryInfo objects for all detected libraries.
        """
        return [
            self.detect_pytorch(),
            self.detect_tensorflow(),
            self.detect_cudf(),
            self.detect_cuopt(),
        ]

    def detect_environment(self) -> CUDAEnvironment:
        """
        Perform complete CUDA environment detection.

        Returns:
            CUDAEnvironment object with all detection results.
        """
        # Detect GPU and driver info
        nvidia_info = self.detect_nvidia_smi()

        # Detect CUDA versions
        cuda_runtime = self.detect_cuda_runtime()
        nvcc_version = self.detect_nvcc_version()

        # Detect libraries
        libraries = self.detect_all_libraries()

        # Identify breaking changes (to be implemented with breaking_changes module)
        breaking_changes: List[Dict[str, Any]] = []

        return CUDAEnvironment(
            cuda_runtime_version=cuda_runtime,
            cuda_driver_version=nvidia_info.get("cuda_version"),
            nvcc_version=nvcc_version,
            gpus=nvidia_info.get("gpus", []),
            libraries=libraries,
            breaking_changes=breaking_changes,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self, env: CUDAEnvironment) -> Dict[str, Any]:
        """
        Convert CUDAEnvironment to dictionary for JSON serialization.

        Args:
            env: CUDAEnvironment object

        Returns:
            Dictionary representation suitable for JSON output.
        """
        return {
            "cuda_runtime_version": env.cuda_runtime_version,
            "cuda_driver_version": env.cuda_driver_version,
            "nvcc_version": env.nvcc_version,
            "gpus": [asdict(gpu) for gpu in env.gpus],
            "libraries": [asdict(lib) for lib in env.libraries],
            "breaking_changes": env.breaking_changes,
            "timestamp": env.timestamp,
        }

    def to_json(self, env: CUDAEnvironment, indent: int = 2) -> str:
        """
        Convert CUDAEnvironment to JSON string.

        Args:
            env: CUDAEnvironment object
            indent: JSON indentation level

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(env), indent=indent)


def detect_cuda_environment() -> Dict[str, Any]:
    """
    Convenience function to detect CUDA environment and return as dictionary.

    Returns:
        Dictionary with complete CUDA environment information.
    """
    detector = CUDADetector()
    environment = detector.detect_environment()
    return detector.to_dict(environment)


if __name__ == "__main__":
    # Quick test when run directly
    detector = CUDADetector()
    environment = detector.detect_environment()
    print(detector.to_json(environment))
