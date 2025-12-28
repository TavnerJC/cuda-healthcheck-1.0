"""
Breaking Changes Database Module.

Provides access to CUDA breaking changes database and compatibility
scoring functions.

Example:
    ```python
    from src.data import BreakingChangesDatabase, score_compatibility

    db = BreakingChangesDatabase()
    changes = db.get_changes_by_library("pytorch")
    ```
"""

from .breaking_changes import (
    BreakingChange,
    BreakingChangesDatabase,
    score_compatibility,
    get_breaking_changes,
    Severity,
)

__all__ = [
    "BreakingChange",
    "BreakingChangesDatabase",
    "score_compatibility",
    "get_breaking_changes",
    "Severity",
]

__version__ = "1.0.0"
