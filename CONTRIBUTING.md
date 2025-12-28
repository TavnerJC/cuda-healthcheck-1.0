# Contributing to CUDA Healthcheck Tool

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run tests**: `pytest tests/`
6. **Format code**: `black src/ tests/`
7. **Commit changes**: `git commit -m "Add: your feature description"`
8. **Push to your fork**: `git push origin feature/your-feature-name`
9. **Create a Pull Request**

## Code Style

- **Python**: Follow PEP 8 style guide
- **Formatting**: Use Black with default settings
- **Linting**: Code must pass flake8 checks
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

Example:

```python
def detect_cuda_version() -> Optional[str]:
    """
    Detect CUDA version from nvidia-smi.
    
    Returns:
        CUDA version string or None if not detected.
    """
    pass
```

## Testing

- **Unit tests required** for all new functions
- **Coverage**: Aim for >80% code coverage
- **Test naming**: `test_<function_name>_<scenario>`
- **Use mocks**: Mock external dependencies (subprocess, file I/O)

Example:

```python
def test_detect_cuda_version_success(self, mock_run):
    """Test successful CUDA version detection."""
    mock_run.return_value = Mock(returncode=0, stdout="12.4")
    result = detect_cuda_version()
    assert result == "12.4"
```

## Adding Breaking Changes

To add a new breaking change to the database:

1. Edit `src/data/breaking_changes.py`
2. Add a new `BreakingChange` object to `_initialize_database()`
3. Follow the existing pattern:

```python
BreakingChange(
    id="unique-id",
    title="Short description",
    severity=Severity.CRITICAL.value,
    affected_library="library-name",
    cuda_version_from="12.4",
    cuda_version_to="13.0",
    description="Detailed description...",
    affected_apis=["api1", "api2"],
    migration_path="Step-by-step migration instructions...",
    references=["https://docs.link1", "https://docs.link2"],
    applies_to_compute_capabilities=["9.0"]  # Optional
)
```

4. Add a test case in `tests/test_breaking_changes.py`
5. Update documentation in `docs/MIGRATION_GUIDE.md`

## PR Guidelines

### PR Title Format
- `Add: [description]` - New feature
- `Fix: [description]` - Bug fix
- `Update: [description]` - Update existing feature
- `Docs: [description]` - Documentation changes
- `Test: [description]` - Test updates

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing performed
- [ ] All tests pass

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings
```

## CodeRabbit Integration

PRs are automatically reviewed by CodeRabbit AI. Address any issues it identifies before requesting human review.

## Commit Message Guidelines

Follow conventional commit format:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

Example:
```
feat(detector): add support for CUDA 13.1

- Add CUDA 13.1 detection in detector.py
- Update breaking changes database
- Add migration guide section

Closes #123
```

## Release Process

1. Update version in `src/__init__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. Push tag: `git push origin v1.0.0`
5. GitHub Actions will create release

## Questions?

- Create a GitHub issue
- Tag with `question` label
- We'll respond within 48 hours

Thank you for contributing! 🎉








