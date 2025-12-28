"""
Unit tests for custom exceptions.

Tests can be run locally without any dependencies on Databricks or CUDA.
"""

import pytest

from src.utils.exceptions import (
    BreakingChangeError,
    ClusterNotFoundError,
    ClusterNotRunningError,
    CompatibilityError,
    ConfigurationError,
    CudaDetectionError,
    CudaHealthcheckError,
    DatabricksConnectionError,
    DeltaTableError,
)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from CudaHealthcheckError."""
        exceptions = [
            CudaDetectionError,
            DatabricksConnectionError,
            ClusterNotRunningError,
            ClusterNotFoundError,
            DeltaTableError,
            CompatibilityError,
            BreakingChangeError,
            ConfigurationError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, CudaHealthcheckError)

    def test_base_exception_inherits_from_exception(self):
        """Test that base exception inherits from built-in Exception."""
        assert issubclass(CudaHealthcheckError, Exception)


class TestExceptionCreation:
    """Test exception instantiation and basic properties."""

    def test_cuda_healthcheck_error_creation(self):
        """Test creating base CudaHealthcheckError."""
        error = CudaHealthcheckError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_cuda_detection_error_creation(self):
        """Test creating CudaDetectionError."""
        error = CudaDetectionError("CUDA detection failed")
        assert str(error) == "CUDA detection failed"
        assert isinstance(error, CudaHealthcheckError)

    def test_databricks_connection_error_creation(self):
        """Test creating DatabricksConnectionError."""
        error = DatabricksConnectionError("Connection failed")
        assert str(error) == "Connection failed"

    def test_cluster_not_running_error_creation(self):
        """Test creating ClusterNotRunningError."""
        error = ClusterNotRunningError("Cluster is terminated")
        assert str(error) == "Cluster is terminated"

    def test_cluster_not_found_error_creation(self):
        """Test creating ClusterNotFoundError."""
        error = ClusterNotFoundError("Cluster ABC not found")
        assert str(error) == "Cluster ABC not found"

    def test_delta_table_error_creation(self):
        """Test creating DeltaTableError."""
        error = DeltaTableError("Table write failed")
        assert str(error) == "Table write failed"

    def test_compatibility_error_creation(self):
        """Test creating CompatibilityError."""
        error = CompatibilityError("Incompatible versions")
        assert str(error) == "Incompatible versions"

    def test_breaking_change_error_creation(self):
        """Test creating BreakingChangeError."""
        error = BreakingChangeError("Breaking change detected")
        assert str(error) == "Breaking change detected"

    def test_configuration_error_creation(self):
        """Test creating ConfigurationError."""
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"


class TestExceptionCatching:
    """Test exception catching behavior."""

    def test_catch_specific_exception(self):
        """Test catching specific exception type."""

        def raises_detection_error():
            raise CudaDetectionError("Detection failed")

        with pytest.raises(CudaDetectionError) as exc_info:
            raises_detection_error()

        assert "Detection failed" in str(exc_info.value)

    def test_catch_base_exception(self):
        """Test that base exception catches all derived exceptions."""

        def raises_derived_error():
            raise ClusterNotFoundError("Cluster missing")

        with pytest.raises(CudaHealthcheckError):
            raises_derived_error()

    def test_catch_multiple_exception_types(self):
        """Test catching multiple exception types."""

        def raises_various_errors(error_type):
            if error_type == "cuda":
                raise CudaDetectionError("CUDA error")
            elif error_type == "databricks":
                raise DatabricksConnectionError("Connection error")
            else:
                raise Exception("Unknown error")

        # Catch CUDA error
        with pytest.raises(CudaDetectionError):
            raises_various_errors("cuda")

        # Catch Databricks error
        with pytest.raises(DatabricksConnectionError):
            raises_various_errors("databricks")

    def test_exception_hierarchy_catching(self):
        """Test that catching base class catches derived exceptions."""
        exceptions_raised = []

        for exc_class in [CudaDetectionError, DatabricksConnectionError, ClusterNotFoundError]:
            try:
                raise exc_class("Test error")
            except CudaHealthcheckError as e:
                exceptions_raised.append(type(e).__name__)

        assert len(exceptions_raised) == 3
        assert "CudaDetectionError" in exceptions_raised
        assert "DatabricksConnectionError" in exceptions_raised
        assert "ClusterNotFoundError" in exceptions_raised


class TestExceptionMessages:
    """Test exception message handling."""

    def test_exception_with_formatted_message(self):
        """Test exception with formatted message."""
        cluster_id = "cluster-123"
        error = ClusterNotFoundError(f"Cluster {cluster_id} not found in workspace")
        assert "cluster-123" in str(error)

    def test_exception_with_multiline_message(self):
        """Test exception with multiline message."""
        message = """
        CUDA detection failed:
        - nvidia-smi not found
        - CUDA toolkit not installed
        """
        error = CudaDetectionError(message)
        assert "nvidia-smi" in str(error)
        assert "CUDA toolkit" in str(error)

    def test_exception_reraise_with_context(self):
        """Test re-raising exception with context."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ConfigurationError("Configuration invalid") from e
        except ConfigurationError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)


class TestExceptionUsageScenarios:
    """Test realistic exception usage scenarios."""

    def test_cuda_detection_failure_scenario(self):
        """Test typical CUDA detection failure scenario."""

        def detect_cuda():
            # Simulate nvidia-smi not found
            raise CudaDetectionError("nvidia-smi command not found")

        with pytest.raises(CudaDetectionError) as exc_info:
            detect_cuda()

        assert "nvidia-smi" in str(exc_info.value)

    def test_databricks_connection_failure_scenario(self):
        """Test typical Databricks connection failure."""

        def connect_to_databricks(host, token):
            if not token:
                raise DatabricksConnectionError("DATABRICKS_TOKEN environment variable not set")

        with pytest.raises(DatabricksConnectionError) as exc_info:
            connect_to_databricks("https://example.com", None)

        assert "DATABRICKS_TOKEN" in str(exc_info.value)

    def test_cluster_state_error_scenario(self):
        """Test cluster state error scenario."""

        def ensure_cluster_running(cluster_id, state):
            if state != "RUNNING":
                raise ClusterNotRunningError(
                    f"Cluster {cluster_id} is in {state} state, expected RUNNING"
                )

        with pytest.raises(ClusterNotRunningError) as exc_info:
            ensure_cluster_running("cluster-123", "TERMINATED")

        assert "TERMINATED" in str(exc_info.value)

    def test_compatibility_check_scenario(self):
        """Test compatibility check failure scenario."""

        def check_compatibility(local_version, cluster_version):
            if local_version != cluster_version:
                raise CompatibilityError(
                    f"Version mismatch: local={local_version}, cluster={cluster_version}"
                )

        with pytest.raises(CompatibilityError) as exc_info:
            check_compatibility("12.4", "13.0")

        assert "12.4" in str(exc_info.value)
        assert "13.0" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
