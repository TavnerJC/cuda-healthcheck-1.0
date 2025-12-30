"""
CUDA Breaking Changes Database.

Maintains a structured database of known CUDA version incompatibilities,
library breaking changes, and migration paths for Databricks environments.
"""

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(Enum):
    """Severity levels for breaking changes."""

    CRITICAL = "CRITICAL"  # Will cause failures
    WARNING = "WARNING"  # May cause issues
    INFO = "INFO"  # Informational


@dataclass
class BreakingChange:
    """Represents a single breaking change."""

    id: str
    title: str
    severity: str
    affected_library: str
    cuda_version_from: str
    cuda_version_to: str
    description: str
    affected_apis: List[str]
    migration_path: str
    references: List[str]
    applies_to_compute_capabilities: Optional[List[str]] = None


class BreakingChangesDatabase:
    """Database of CUDA breaking changes and compatibility issues."""

    def __init__(self) -> None:
        """Initialize the breaking changes database."""
        self.breaking_changes = self._initialize_database()

    def _initialize_database(self) -> List[BreakingChange]:
        """
        Initialize the database with known breaking changes.

        Returns:
            List of BreakingChange objects.
        """
        return [
            # PyTorch CUDA 12.x -> 13.x
            BreakingChange(
                id="pytorch-cuda13-rebuild",
                title="PyTorch requires rebuild for CUDA 13.x",
                severity=Severity.CRITICAL.value,
                affected_library="pytorch",
                cuda_version_from="12.x",
                cuda_version_to="13.0",
                description=(
                    "PyTorch compiled for CUDA 12.x will not work with CUDA 13.x. "
                    "You must use PyTorch binaries specifically built for CUDA 13.x."
                ),
                affected_apis=[
                    "torch.cuda.is_available()",
                    "torch.cuda.device_count()",
                    "All CUDA tensor operations",
                ],
                migration_path=(
                    "1. Wait for official PyTorch CUDA 13.x builds\n"
                    "2. Install: pip install torch "
                    "--index-url https://download.pytorch.org/whl/cu130\n"
                    "3. Verify with: python -c 'import torch; print(torch.version.cuda)'"
                ),
                references=[
                    "https://pytorch.org/get-started/locally/",
                    "https://github.com/pytorch/pytorch/issues/cuda-13-support",
                ],
            ),
            # PyTorch CUDA 12.4 -> 12.6
            BreakingChange(
                id="pytorch-cuda126-compatibility",
                title="PyTorch CUDA 12.4 binaries may work with 12.6 but not guaranteed",
                severity=Severity.WARNING.value,
                affected_library="pytorch",
                cuda_version_from="12.4",
                cuda_version_to="12.6",
                description=(
                    "PyTorch built for CUDA 12.4 may work with CUDA 12.6 due to minor version "
                    "compatibility, but this is not guaranteed. Performance issues may occur."
                ),
                affected_apis=["CUDA kernel launches", "CuDNN operations"],
                migration_path=(
                    "1. Test thoroughly in development environment\n"
                    "2. Consider rebuilding PyTorch for CUDA 12.6 if issues occur\n"
                    "3. Monitor for CUDA errors in logs"
                ),
                references=[
                    "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#versioning"
                ],
            ),
            # TensorFlow CUDA 13.x
            BreakingChange(
                id="tensorflow-cuda13-support",
                title="TensorFlow CUDA 13.x support requires TF 2.18+",
                severity=Severity.CRITICAL.value,
                affected_library="tensorflow",
                cuda_version_from="12.x",
                cuda_version_to="13.0",
                description=(
                    "TensorFlow versions below 2.18 do not support CUDA 13.x. "
                    "Attempting to use older TensorFlow with CUDA 13.x will fail."
                ),
                affected_apis=[
                    "tf.config.list_physical_devices('GPU')",
                    "All GPU operations",
                ],
                migration_path=(
                    "1. Upgrade to TensorFlow 2.18 or later\n"
                    "2. pip install tensorflow[and-cuda]==2.18.0\n"
                    "3. Verify GPU detection: python -c 'import tensorflow as tf; "
                    'print(tf.config.list_physical_devices("GPU"))\''
                ),
                references=[
                    "https://www.tensorflow.org/install/source#gpu",
                    "https://github.com/tensorflow/tensorflow/releases",
                ],
            ),
            # TensorFlow Compute Capability
            BreakingChange(
                id="tensorflow-sm90-support",
                title="TensorFlow requires 2.16+ for SM_90 (H100/H200 GPUs)",
                severity=Severity.CRITICAL.value,
                affected_library="tensorflow",
                cuda_version_from="Any",
                cuda_version_to="Any",
                description=(
                    "TensorFlow versions below 2.16 do not support compute capability 9.0 "
                    "(NVIDIA H100, H200, and future Hopper+ GPUs)."
                ),
                affected_apis=["All GPU tensor operations on H100/H200"],
                migration_path=(
                    "1. Upgrade to TensorFlow 2.16 or later\n"
                    "2. Ensure CUDA 12.3+ is installed\n"
                    "3. Verify: nvidia-smi to check GPU compute capability"
                ),
                references=["https://developer.nvidia.com/cuda-gpus"],
                applies_to_compute_capabilities=["9.0"],
            ),
            # cuDF/RAPIDS CUDA 13.x
            BreakingChange(
                id="cudf-cuda13-support",
                title="cuDF/RAPIDS 24.12+ required for CUDA 13.x",
                severity=Severity.CRITICAL.value,
                affected_library="cudf",
                cuda_version_from="12.x",
                cuda_version_to="13.0",
                description=(
                    "RAPIDS libraries (cuDF, cuML, cuGraph) require version 24.12 or later "
                    "for CUDA 13.x support. Earlier versions will not work."
                ),
                affected_apis=[
                    "cudf.DataFrame",
                    "cudf.read_csv",
                    "All cuDF operations",
                ],
                migration_path=(
                    "1. Upgrade RAPIDS to 24.12+\n"
                    "2. conda install -c rapidsai -c conda-forge -c nvidia "
                    "cudf=24.12 python=3.11 cuda-version=13.0\n"
                    "3. Or use pip: pip install cudf-cu13==24.12.*"
                ),
                references=[
                    "https://rapids.ai/start.html",
                    "https://docs.rapids.ai/install",
                ],
            ),
            # cuDF versioning
            BreakingChange(
                id="cudf-cuda-version-matching",
                title="cuDF package name must match CUDA major version",
                severity=Severity.CRITICAL.value,
                affected_library="cudf",
                cuda_version_from="Any",
                cuda_version_to="Any",
                description=(
                    "cuDF package names include CUDA version (e.g., cudf-cu12 for CUDA 12.x, "
                    "cudf-cu13 for CUDA 13.x). Installing wrong package will fail."
                ),
                affected_apis=["All cuDF functionality"],
                migration_path=(
                    "1. Identify your CUDA version: nvcc --version\n"
                    "2. Install matching package:\n"
                    "   - CUDA 12.x: pip install cudf-cu12\n"
                    "   - CUDA 13.x: pip install cudf-cu13\n"
                    "3. Uninstall incorrect versions first"
                ),
                references=["https://docs.rapids.ai/install"],
            ),
            # NVIDIA Containers - Isaac Sim
            BreakingChange(
                id="isaac-sim-cuda-requirements",
                title="NVIDIA Isaac Sim requires CUDA 12.2+ and specific drivers",
                severity=Severity.CRITICAL.value,
                affected_library="isaac-sim",
                cuda_version_from="11.x",
                cuda_version_to="12.2",
                description=(
                    "Isaac Sim (robotics simulation) requires CUDA 12.2 or later and "
                    "driver version 535.104.05+. Older versions will not work."
                ),
                affected_apis=["omni.isaac.core", "All Isaac Sim APIs"],
                migration_path=(
                    "1. Update NVIDIA drivers to 535.104.05+\n"
                    "2. Install CUDA 12.2 or later\n"
                    "3. Use Isaac Sim container: docker pull nvcr.io/nvidia/isaac-sim:2024.1.0\n"
                    "4. Verify driver: nvidia-smi"
                ),
                references=[
                    "https://docs.omniverse.nvidia.com/isaacsim/latest/"
                    "installation/requirements.html",
                    "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim",
                ],
            ),
            # BioNeMo
            BreakingChange(
                id="bionemo-cuda-requirements",
                title="NVIDIA BioNeMo 2.0 requires CUDA 12.4+",
                severity=Severity.CRITICAL.value,
                affected_library="bionemo",
                cuda_version_from="12.2",
                cuda_version_to="12.4",
                description=(
                    "BioNeMo 2.0 (AI for drug discovery) requires CUDA 12.4 or later. "
                    "Minimum driver version: 550.54.15."
                ),
                affected_apis=["bionemo.model", "All BioNeMo training APIs"],
                migration_path=(
                    "1. Update NVIDIA drivers to 550.54.15+\n"
                    "2. Install CUDA 12.4+\n"
                    "3. Use BioNeMo container: docker pull "
                    "nvcr.io/nvidia/clara/bionemo-framework:2.0\n"
                    "4. Check compatibility: nvidia-smi"
                ),
                references=[
                    "https://docs.nvidia.com/bionemo/",
                    "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/"
                    "containers/bionemo-framework",
                ],
            ),
            # Modulus (Physics ML)
            BreakingChange(
                id="modulus-cuda-requirements",
                title="NVIDIA Modulus requires CUDA 12.1+ for latest features",
                severity=Severity.WARNING.value,
                affected_library="modulus",
                cuda_version_from="11.x",
                cuda_version_to="12.1",
                description=(
                    "NVIDIA Modulus (Physics-ML framework) recommends CUDA 12.1+ for "
                    "optimal performance and latest features. CUDA 11.x may have limitations."
                ),
                affected_apis=["modulus.models", "Physics-informed neural networks"],
                migration_path=(
                    "1. Update to CUDA 12.1 or later\n"
                    "2. Install Modulus: pip install nvidia-modulus\n"
                    "3. Or use container: docker pull nvcr.io/nvidia/modulus/modulus:24.01"
                ),
                references=[
                    "https://docs.nvidia.com/deeplearning/modulus/",
                    "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus",
                ],
            ),
            # CuDNN Version Changes
            BreakingChange(
                id="cudnn9-api-changes",
                title="cuDNN 9.x introduces API changes",
                severity=Severity.WARNING.value,
                affected_library="cudnn",
                cuda_version_from="12.4",
                cuda_version_to="13.0",
                description=(
                    "cuDNN 9.x (shipped with CUDA 13.0) introduces API changes that may "
                    "affect custom CUDA kernels and low-level GPU code."
                ),
                affected_apis=[
                    "cudnnCreate",
                    "cudnnSetTensor4dDescriptor",
                    "Custom kernels",
                ],
                migration_path=(
                    "1. Review cuDNN 9.x migration guide\n"
                    "2. Update custom CUDA kernels\n"
                    "3. Test thoroughly with cuDNN 9.x\n"
                    "4. Most high-level frameworks (PyTorch, TF) handle this automatically"
                ),
                references=[
                    "https://docs.nvidia.com/deeplearning/cudnn/release-notes/",
                    "https://docs.nvidia.com/deeplearning/cudnn/developer-guide/",
                ],
            ),
            # Compute Capability Deprecations
            BreakingChange(
                id="cuda13-sm50-deprecation",
                title="CUDA 13.x deprecates compute capability 5.0 (Maxwell GPUs)",
                severity=Severity.WARNING.value,
                affected_library="cuda",
                cuda_version_from="12.x",
                cuda_version_to="13.0",
                description=(
                    "CUDA 13.x deprecates support for compute capability 5.0 "
                    "(Maxwell architecture: GTX 900 series, Quadro M series). "
                    "These GPUs may not work correctly."
                ),
                affected_apis=["All CUDA operations on Maxwell GPUs"],
                migration_path=(
                    "1. Upgrade to newer GPU hardware (Pascal/Turing/Ampere/Hopper)\n"
                    "2. Or stay on CUDA 12.x for Maxwell GPU support\n"
                    "3. Check GPU: nvidia-smi --query-gpu=compute_cap --format=csv"
                ),
                references=["https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/"],
                applies_to_compute_capabilities=["5.0"],
            ),
            # CuOPT nvJitLink incompatibility (Databricks-specific)
            BreakingChange(
                id="cuopt-nvjitlink-databricks-ml-runtime",
                title="CuOPT 25.12+ requires nvJitLink 12.9+ (incompatible with Databricks ML Runtime 16.4)",
                severity=Severity.CRITICAL.value,
                affected_library="cuopt",
                cuda_version_from="12.4",
                cuda_version_to="12.9",
                description=(
                    "NVIDIA CuOPT 25.12.0 requires nvidia-nvjitlink-cu12>=12.9.79, but Databricks ML Runtime 16.4 "
                    "provides nvidia-nvjitlink-cu12 12.4.127. This causes 'undefined symbol: __nvJitLinkGetErrorLogSize_12_9' "
                    "errors when loading libcuopt.so. Users CANNOT upgrade nvJitLink in Databricks managed runtimes as CUDA "
                    "versions are environment-controlled by Databricks. This is a breaking incompatibility between CuOPT "
                    "releases and Databricks runtime versions that prevents GPU-accelerated routing optimization."
                ),
                affected_apis=[
                    "cuopt.routing.DataModel",
                    "cuopt.routing.Solve",
                    "cuopt.routing.SolverSettings",
                    "All CuOPT routing and optimization APIs",
                ],
                migration_path=(
                    "1. CANNOT FIX IN DATABRICKS ML RUNTIME 16.4 - CUDA libraries are runtime-locked\n"
                    "2. Recommended actions:\n"
                    "   a) Report to Databricks: https://github.com/databricks-industry-solutions/routing/issues\n"
                    "   b) Request ML Runtime update with CUDA 12.9+ support\n"
                    "   c) Use alternative solver: Google OR-Tools (pip install ortools)\n"
                    "   d) Wait for Databricks ML Runtime 17.0+ with updated CUDA components\n"
                    "3. Temporary workaround (may not work):\n"
                    "   - pip install --upgrade nvidia-nvjitlink-cu12>=12.9.79 (often fails due to env constraints)\n"
                    "4. Verification:\n"
                    "   - Check nvJitLink: pip show nvidia-nvjitlink-cu12\n"
                    "   - Check runtime: dbutils.notebook.run('/Workspace/...', 0, {})"
                ),
                references=[
                    "https://github.com/databricks-industry-solutions/routing",
                    "https://github.com/NVIDIA/cuopt",
                    "https://docs.databricks.com/en/release-notes/runtime/index.html",
                    "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compatibility",
                ],
            ),
        ]

    def get_all_changes(self) -> List[BreakingChange]:
        """
        Get all breaking changes in the database.

        Returns:
            List of all BreakingChange objects.
        """
        return self.breaking_changes

    def get_changes_by_library(self, library: str) -> List[BreakingChange]:
        """
        Get breaking changes for a specific library.

        Args:
            library: Library name (pytorch, tensorflow, cudf, etc.)

        Returns:
            List of BreakingChange objects for the specified library.
        """
        return [
            change
            for change in self.breaking_changes
            if change.affected_library.lower() == library.lower()
        ]

    def get_changes_by_cuda_transition(
        self, from_version: str, to_version: str
    ) -> List[BreakingChange]:
        """
        Get breaking changes for a CUDA version transition.

        Args:
            from_version: Source CUDA version (e.g., "12.4")
            to_version: Target CUDA version (e.g., "13.0")

        Returns:
            List of applicable BreakingChange objects.
        """
        changes = []

        for change in self.breaking_changes:
            # Match version patterns (e.g., "12.x" matches "12.4")
            from_match = (
                change.cuda_version_from == from_version
                or change.cuda_version_from.replace(".x", "") in from_version
                or change.cuda_version_from == "Any"
            )

            to_match = (
                change.cuda_version_to == to_version
                or change.cuda_version_to.replace(".x", "") in to_version
                or change.cuda_version_to == "Any"
            )

            if from_match and to_match:
                changes.append(change)

        return changes

    def score_compatibility(
        self,
        detected_libraries: List[Dict[str, Any]],
        cuda_version: str,
        compute_capability: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score compatibility for detected environment.

        Args:
            detected_libraries: List of library info dicts from detector
            cuda_version: Detected CUDA version
            compute_capability: GPU compute capability (e.g., "8.0")

        Returns:
            Dictionary with compatibility score and applicable breaking changes.
        """
        applicable_changes: Dict[str, List[Dict[str, Any]]] = {
            "CRITICAL": [],
            "WARNING": [],
            "INFO": [],
        }

        # Check each detected library
        for lib_info in detected_libraries:
            lib_name = lib_info.get("name", "").lower()
            # lib_version and lib_cuda are reserved for future compatibility checks
            # lib_version = lib_info.get("version", "")
            # lib_cuda = lib_info.get("cuda_version")

            if lib_name == "not installed":
                continue

            # Get changes for this library
            lib_changes = self.get_changes_by_library(lib_name)

            for change in lib_changes:
                # Check if this change applies to current CUDA version
                if (
                    change.cuda_version_to.replace(".x", "") in cuda_version
                    or change.cuda_version_to == "Any"
                ):
                    # Check compute capability if specified
                    if change.applies_to_compute_capabilities:
                        if (
                            compute_capability
                            and compute_capability in change.applies_to_compute_capabilities
                        ):
                            applicable_changes[change.severity].append(asdict(change))
                    else:
                        applicable_changes[change.severity].append(asdict(change))

        # Check CUDA-level changes (not library specific)
        cuda_changes = [c for c in self.breaking_changes if c.affected_library == "cuda"]
        for change in cuda_changes:
            if change.applies_to_compute_capabilities:
                if (
                    compute_capability
                    and compute_capability in change.applies_to_compute_capabilities
                ):
                    applicable_changes[change.severity].append(asdict(change))

        # Calculate compatibility score (0-100)
        critical_count = len(applicable_changes["CRITICAL"])
        warning_count = len(applicable_changes["WARNING"])
        info_count = len(applicable_changes["INFO"])

        # Score: 100 - (criticals * 30) - (warnings * 10) - (info * 2)
        score = max(0, 100 - (critical_count * 30) - (warning_count * 10) - (info_count * 2))

        return {
            "compatibility_score": score,
            "total_issues": critical_count + warning_count + info_count,
            "critical_issues": critical_count,
            "warning_issues": warning_count,
            "info_issues": info_count,
            "breaking_changes": applicable_changes,
            "recommendation": self._get_recommendation(score, critical_count),
        }

    def _get_recommendation(self, score: int, critical_count: int) -> str:
        """
        Generate recommendation based on compatibility score.

        Args:
            score: Compatibility score (0-100)
            critical_count: Number of critical issues

        Returns:
            Recommendation string.
        """
        if critical_count > 0:
            return (
                "CRITICAL: Environment has breaking changes that will cause failures. "
                "Immediate action required."
            )
        elif score >= 90:
            return "GOOD: Environment is highly compatible. Minor issues may exist."
        elif score >= 70:
            return "ACCEPTABLE: Environment is mostly compatible. Review warnings."
        elif score >= 50:
            return "CAUTION: Environment has compatibility concerns. Testing recommended."
        else:
            return "HIGH RISK: Environment has significant compatibility issues. Migration needed."

    def export_to_json(self, filepath: str) -> None:
        """
        Export breaking changes database to JSON file.

        Args:
            filepath: Path to output JSON file.
        """
        data = {
            "breaking_changes": [asdict(change) for change in self.breaking_changes],
            "version": "1.0",
            "last_updated": "2024-12",
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, filepath: str) -> None:
        """
        Load breaking changes from JSON file.

        Args:
            filepath: Path to JSON file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self.breaking_changes = [
            BreakingChange(**change) for change in data.get("breaking_changes", [])
        ]


def score_compatibility(
    detected_libraries: List[Dict[str, Any]],
    cuda_version: str,
    compute_capability: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to score compatibility.

    Args:
        detected_libraries: List of library info dicts
        cuda_version: CUDA version string
        compute_capability: GPU compute capability

    Returns:
        Compatibility score dictionary.
    """
    db = BreakingChangesDatabase()
    return db.score_compatibility(detected_libraries, cuda_version, compute_capability)


def get_breaking_changes(library: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get breaking changes, optionally filtered by library.

    Args:
        library: Optional library name to filter by

    Returns:
        List of breaking change dictionaries.
    """
    db = BreakingChangesDatabase()

    if library:
        changes = db.get_changes_by_library(library)
    else:
        changes = db.get_all_changes()

    return [asdict(change) for change in changes]


if __name__ == "__main__":
    # Demo: export database to JSON
    db = BreakingChangesDatabase()

    print("CUDA Breaking Changes Database")
    print("=" * 80)
    print(f"Total changes: {len(db.get_all_changes())}")
    print()

    # Show changes by library
    for library in ["pytorch", "tensorflow", "cudf", "isaac-sim", "bionemo"]:
        changes = db.get_changes_by_library(library)
        print(f"{library}: {len(changes)} changes")

    print()
    print("Exporting to breaking_changes.json...")
    db.export_to_json("breaking_changes.json")
    print("Done!")
