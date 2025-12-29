"""
Unit tests for Breaking Changes Database.

Tests can be run locally without any dependencies on Databricks or CUDA.
"""

import pytest

from cuda_healthcheck.data.breaking_changes import (
    BreakingChange,
    BreakingChangesDatabase,
    Severity,
    get_breaking_changes,
    score_compatibility,
)


class TestBreakingChangeDataclass:
    """Test BreakingChange dataclass."""

    def test_breaking_change_creation(self):
        """Test creating a BreakingChange instance."""
        change = BreakingChange(
            id="test-1",
            title="Test Change",
            severity="CRITICAL",
            affected_library="pytorch",
            cuda_version_from="12.4",
            cuda_version_to="13.0",
            description="Test description",
            affected_apis=["api1", "api2"],
            migration_path="Step 1\nStep 2",
            references=["https://example.com"],
        )

        assert change.id == "test-1"
        assert change.title == "Test Change"
        assert change.severity == "CRITICAL"
        assert len(change.affected_apis) == 2


class TestBreakingChangesDatabaseInitialization:
    """Test BreakingChangesDatabase initialization."""

    def test_database_initialization(self):
        """Test that database initializes with changes."""
        db = BreakingChangesDatabase()
        assert db is not None
        assert len(db.breaking_changes) > 0

    def test_database_has_pytorch_changes(self):
        """Test that database includes PyTorch changes."""
        db = BreakingChangesDatabase()
        pytorch_changes = [c for c in db.breaking_changes if c.affected_library == "pytorch"]
        assert len(pytorch_changes) > 0

    def test_database_has_tensorflow_changes(self):
        """Test that database includes TensorFlow changes."""
        db = BreakingChangesDatabase()
        tf_changes = [c for c in db.breaking_changes if c.affected_library == "tensorflow"]
        assert len(tf_changes) > 0

    def test_database_has_cudf_changes(self):
        """Test that database includes cuDF changes."""
        db = BreakingChangesDatabase()
        cudf_changes = [c for c in db.breaking_changes if c.affected_library == "cudf"]
        assert len(cudf_changes) > 0


class TestGetAllChanges:
    """Test get_all_changes method."""

    def test_get_all_changes_returns_list(self):
        """Test that get_all_changes returns a list."""
        db = BreakingChangesDatabase()
        changes = db.get_all_changes()
        assert isinstance(changes, list)
        assert len(changes) > 0

    def test_get_all_changes_returns_breaking_change_objects(self):
        """Test that all items are BreakingChange objects."""
        db = BreakingChangesDatabase()
        changes = db.get_all_changes()
        for change in changes:
            assert isinstance(change, BreakingChange)


class TestGetChangesByLibrary:
    """Test get_changes_by_library method."""

    def test_get_changes_by_library_pytorch(self):
        """Test getting PyTorch changes."""
        db = BreakingChangesDatabase()
        changes = db.get_changes_by_library("pytorch")

        assert isinstance(changes, list)
        assert len(changes) > 0
        assert all(c.affected_library == "pytorch" for c in changes)

    def test_get_changes_by_library_tensorflow(self):
        """Test getting TensorFlow changes."""
        db = BreakingChangesDatabase()
        changes = db.get_changes_by_library("tensorflow")

        assert len(changes) > 0
        assert all(c.affected_library == "tensorflow" for c in changes)

    def test_get_changes_by_library_case_insensitive(self):
        """Test that library search is case insensitive."""
        db = BreakingChangesDatabase()
        changes_lower = db.get_changes_by_library("pytorch")
        changes_upper = db.get_changes_by_library("PYTORCH")
        changes_mixed = db.get_changes_by_library("PyTorch")

        assert len(changes_lower) == len(changes_upper)
        assert len(changes_lower) == len(changes_mixed)

    def test_get_changes_by_library_nonexistent(self):
        """Test getting changes for nonexistent library."""
        db = BreakingChangesDatabase()
        changes = db.get_changes_by_library("nonexistent-library")
        assert len(changes) == 0


class TestGetChangesByCudaTransition:
    """Test get_changes_by_cuda_transition method."""

    def test_get_changes_by_cuda_transition_exact_match(self):
        """Test transition with exact version match."""
        db = BreakingChangesDatabase()
        changes = db.get_changes_by_cuda_transition("12.4", "13.0")

        # Should find changes for this transition
        assert isinstance(changes, list)

    def test_get_changes_by_cuda_transition_returns_relevant_changes(self):
        """Test that returned changes are relevant to transition."""
        db = BreakingChangesDatabase()
        changes = db.get_changes_by_cuda_transition("12.4", "13.0")

        # All changes should involve version 13.0 or be general
        for change in changes:
            is_relevant = (
                "13.0" in change.cuda_version_to
                or change.cuda_version_to == "Any"
                or "12" in change.cuda_version_from
                or change.cuda_version_from == "Any"
            )
            assert is_relevant

    def test_get_changes_by_cuda_transition_same_version(self):
        """Test transition with same version."""
        db = BreakingChangesDatabase()
        changes = db.get_changes_by_cuda_transition("12.4", "12.4")

        # May or may not have changes depending on database
        assert isinstance(changes, list)


class TestScoreCompatibility:
    """Test score_compatibility method."""

    def test_score_compatibility_basic(self):
        """Test basic compatibility scoring."""
        db = BreakingChangesDatabase()
        libraries = [
            {
                "name": "pytorch",
                "version": "2.0.0",
                "cuda_version": "12.4",
                "is_compatible": True,
                "warnings": [],
            }
        ]

        result = db.score_compatibility(libraries, "12.4")

        assert isinstance(result, dict)
        assert "compatibility_score" in result
        assert "total_issues" in result
        assert "critical_issues" in result
        assert "warning_issues" in result
        assert "info_issues" in result
        assert "breaking_changes" in result
        assert "recommendation" in result

    def test_score_compatibility_returns_valid_score(self):
        """Test that score is between 0 and 100."""
        db = BreakingChangesDatabase()
        libraries = [
            {
                "name": "pytorch",
                "version": "2.0.0",
                "cuda_version": "12.4",
                "is_compatible": True,
                "warnings": [],
            }
        ]

        result = db.score_compatibility(libraries, "12.4")
        score = result["compatibility_score"]

        assert 0 <= score <= 100

    def test_score_compatibility_with_compute_capability(self):
        """Test scoring with compute capability."""
        db = BreakingChangesDatabase()
        libraries = [
            {
                "name": "tensorflow",
                "version": "2.13.0",
                "cuda_version": "12.4",
                "is_compatible": True,
                "warnings": [],
            }
        ]

        result = db.score_compatibility(libraries, "12.4", compute_capability="9.0")

        assert isinstance(result, dict)
        assert "compatibility_score" in result

    def test_score_compatibility_breaking_changes_structure(self):
        """Test that breaking changes have correct structure."""
        db = BreakingChangesDatabase()
        libraries = [
            {
                "name": "pytorch",
                "version": "2.0.0",
                "cuda_version": "13.0",
                "is_compatible": True,
                "warnings": [],
            }
        ]

        result = db.score_compatibility(libraries, "13.0")
        breaking_changes = result["breaking_changes"]

        assert "CRITICAL" in breaking_changes
        assert "WARNING" in breaking_changes
        assert "INFO" in breaking_changes
        assert isinstance(breaking_changes["CRITICAL"], list)
        assert isinstance(breaking_changes["WARNING"], list)
        assert isinstance(breaking_changes["INFO"], list)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_score_compatibility_function(self):
        """Test score_compatibility convenience function."""
        libraries = [
            {
                "name": "pytorch",
                "version": "2.0.0",
                "cuda_version": "12.4",
                "is_compatible": True,
                "warnings": [],
            }
        ]

        result = score_compatibility(libraries, "12.4")

        assert isinstance(result, dict)
        assert "compatibility_score" in result

    def test_get_breaking_changes_function_all(self):
        """Test get_breaking_changes without filter."""
        changes = get_breaking_changes()

        assert isinstance(changes, list)
        assert len(changes) > 0
        assert all(isinstance(c, dict) for c in changes)

    def test_get_breaking_changes_function_filtered(self):
        """Test get_breaking_changes with library filter."""
        changes = get_breaking_changes(library="pytorch")

        assert isinstance(changes, list)
        # All returned changes should be for pytorch
        for change in changes:
            assert change["affected_library"] == "pytorch"


class TestSeverityEnum:
    """Test Severity enum."""

    def test_severity_enum_values(self):
        """Test that Severity enum has expected values."""
        assert Severity.CRITICAL.value == "CRITICAL"
        assert Severity.WARNING.value == "WARNING"
        assert Severity.INFO.value == "INFO"

    def test_severity_enum_types(self):
        """Test that severity values are strings."""
        assert isinstance(Severity.CRITICAL.value, str)
        assert isinstance(Severity.WARNING.value, str)
        assert isinstance(Severity.INFO.value, str)


class TestRecommendations:
    """Test recommendation generation."""

    def test_recommendation_for_no_issues(self):
        """Test recommendation when no issues found."""
        db = BreakingChangesDatabase()
        libraries = []

        result = db.score_compatibility(libraries, "12.4")

        # High score should give positive recommendation
        if result["compatibility_score"] >= 90:
            assert (
                "GOOD" in result["recommendation"]
                or "compatible" in result["recommendation"].lower()
            )

    def test_recommendation_for_critical_issues(self):
        """Test recommendation when critical issues found."""
        db = BreakingChangesDatabase()
        # Use a version transition known to have critical issues
        libraries = [
            {
                "name": "pytorch",
                "version": "2.0.0",
                "cuda_version": "13.0",
                "is_compatible": True,
                "warnings": [],
            }
        ]

        result = db.score_compatibility(libraries, "13.0")

        # If critical issues found, recommendation should mention them
        if result["critical_issues"] > 0:
            assert "CRITICAL" in result["recommendation"]


class TestDatabaseExportImport:
    """Test database export and import functionality."""

    def test_export_to_json(self, tmp_path):
        """Test exporting database to JSON."""
        db = BreakingChangesDatabase()
        filepath = tmp_path / "test_export.json"

        db.export_to_json(str(filepath))

        assert filepath.exists()

    def test_load_from_json(self, tmp_path):
        """Test loading database from JSON."""
        db = BreakingChangesDatabase()
        filepath = tmp_path / "test_load.json"

        # Export first
        db.export_to_json(str(filepath))

        # Create new database and load
        db2 = BreakingChangesDatabase()
        # original_count = len(db2.breaking_changes)  # Not used
        db2.load_from_json(str(filepath))

        # Should have changes
        assert len(db2.breaking_changes) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_libraries_list(self):
        """Test scoring with empty libraries list."""
        db = BreakingChangesDatabase()
        result = db.score_compatibility([], "12.4")

        # Should not crash and return valid structure
        assert isinstance(result, dict)
        assert "compatibility_score" in result

    def test_unknown_cuda_version(self):
        """Test scoring with unknown CUDA version."""
        db = BreakingChangesDatabase()
        libraries = [
            {
                "name": "pytorch",
                "version": "2.0.0",
                "cuda_version": "99.9",
                "is_compatible": True,
                "warnings": [],
            }
        ]

        result = db.score_compatibility(libraries, "99.9")

        # Should not crash
        assert isinstance(result, dict)

    def test_library_with_warnings(self):
        """Test library with warnings."""
        db = BreakingChangesDatabase()
        libraries = [
            {
                "name": "pytorch",
                "version": "2.0.0",
                "cuda_version": "12.4",
                "is_compatible": True,
                "warnings": ["Warning 1", "Warning 2"],
            }
        ]

        result = db.score_compatibility(libraries, "12.4")

        # Should still return valid result
        assert isinstance(result, dict)
        assert "compatibility_score" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
