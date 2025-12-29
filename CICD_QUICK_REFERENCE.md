# CI/CD Quick Reference Guide

**Quick reference for GitHub Actions workflows in the CUDA Healthcheck Tool**

---

## 🚀 Quick Commands

### Run Tests Locally (CI-equivalent)

```bash
# Navigate to project
cd cuda-healthcheck

# Install test dependencies
pip install pytest pytest-cov pytest-xdist

# Run all tests (parallel)
pytest tests/ -v --tb=short -n auto

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific module
pytest tests/test_orchestrator.py -v
```

### Code Quality Checks

```bash
# Install tools
pip install flake8 black isort mypy bandit radon

# Run linting
flake8 src/ tests/ --max-line-length=100 --max-complexity=10

# Check formatting
black --check --line-length 100 src/ tests/

# Check imports
isort --check-only --profile black --line-length 100 src/ tests/

# Type checking
mypy src/ --ignore-missing-imports

# Security scan
bandit -r src/ -ll

# Complexity
radon cc src/ -a -nb
radon mi src/ -nb
```

### Fix Code Issues

```bash
# Auto-format code
black --line-length 100 src/ tests/

# Auto-fix imports
isort --profile black --line-length 100 src/ tests/
```

---

## 📋 Workflows Overview

| Workflow | Trigger | Duration | Jobs |
|----------|---------|----------|------|
| **Tests** | Push, PR, Manual | ~15 min | 6 |
| **Code Quality** | Push, PR, Manual | ~8 min | 5 |
| **PR Checks** | PR events | ~5 min | 7 |
| **Release** | Tag push | ~10 min | 1 |
| **Nightly** | Daily 2AM UTC | ~20 min | 3 |

---

## 🎯 Test Workflow Jobs

```
┌─────────────────────────────────────┐
│ 1. Multi-version Tests              │
│    Python: 3.10, 3.11, 3.12         │
│    Time: ~5 min                     │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ 2. Coverage Report                  │
│    Target: >80%                     │
│    Time: ~3 min                     │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ 3. Module Tests                     │
│    Utils, Orchestrator, Databricks  │
│    Time: ~4 min                     │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ 4. Compatibility Tests              │
│    CUDA versions, Breaking changes  │
│    Time: ~2 min                     │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ 5. Integration Tests                │
│    Complete workflow                │
│    Time: ~2 min                     │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ 6. Notebook Validation              │
│    Syntax checks                    │
│    Time: ~1 min                     │
└─────────────────────────────────────┘
```

---

## ✨ Code Quality Jobs

```
┌────────────────────────────────────┐
│ Linting: flake8, black, isort      │
│ Status: ✅ MUST PASS               │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│ Type Checking: mypy                │
│ Status: ⚠️ OPTIONAL                │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│ Security: bandit                   │
│ Status: ⚠️ REVIEW                  │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│ Complexity: radon                  │
│ Status: ℹ️ INFO                    │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│ Documentation: docstrings          │
│ Status: ℹ️ INFO                    │
└────────────────────────────────────┘
```

---

## 🔄 Creating a Release

### Method 1: Git Tag (Recommended)

```bash
# Create annotated tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tag to trigger workflow
git push origin v1.0.0

# Wait for workflow to complete
# Check: Actions tab → Release workflow
```

### Method 2: Manual Dispatch

```
1. Go to GitHub → Actions
2. Select "Release" workflow
3. Click "Run workflow"
4. Enter version (e.g., 1.0.0)
5. Click "Run workflow" button
```

### What Happens

```
Tag Push (v1.0.0)
    ↓
Run Full Test Suite ✅
    ↓
Generate Changelog 📝
    ↓
Create .tar.gz Archive 📦
    ↓
Create GitHub Release 🚀
    ↓
Attach Artifacts 📎
```

---

## 🔍 PR Workflow

### What Runs on PR

```
PR Created/Updated
    ↓
┌───────────────────────────────────┐
│ Quick Tests (1-2 min)             │
│ - Exception tests                 │
│ - Logging tests                   │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Full Test Suite (5 min)           │
│ - All Python versions             │
│ - Coverage report                 │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Code Quality (3 min)              │
│ - Linting, typing, security       │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ PR Analysis                       │
│ - Changed files                   │
│ - Size check                      │
│ - Auto-labeling                   │
└───────────────────────────────────┘
```

### PR Labels (Auto-applied)

- `type: src` - Source code changes
- `type: tests` - Test changes
- `type: docs` - Documentation
- `module: *` - Module-specific (databricks, utils, etc.)
- `ci/cd` - CI/CD changes
- `dependencies` - Dependency updates

---

## 🛠️ Manual Workflow Runs

### Via GitHub UI

```
1. Go to repository → Actions tab
2. Select workflow (Tests, Code Quality, etc.)
3. Click "Run workflow" dropdown
4. Select branch
5. Click "Run workflow" button
```

### Via GitHub CLI

```bash
# Install GitHub CLI
# https://cli.github.com/

# List workflows
gh workflow list

# Run specific workflow
gh workflow run test.yml

gh workflow run code-quality.yml

gh workflow run release.yml -f version=1.0.0

# View runs
gh run list --workflow=test.yml

# Watch a run
gh run watch
```

---

## 📊 Monitoring

### Check Status

```bash
# View recent runs
gh run list --limit 10

# View specific run
gh run view <run-id>

# Download artifacts
gh run download <run-id>

# Re-run failed jobs
gh run rerun <run-id>

# Re-run only failed jobs
gh run rerun <run-id> --failed
```

### CI Status Dashboard

**Where to find**:
- Actions tab: All workflow runs
- PR page: Status checks section
- Commit page: Checkmarks/X's next to commits

---

## 🔧 Troubleshooting

### Tests Failing in CI but Pass Locally

```bash
# Check Python version
python --version  # CI uses 3.10, 3.11, 3.12

# Run in clean environment
python -m venv test-env
source test-env/bin/activate  # Windows: test-env\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v

# Check working directory
cd cuda-healthcheck  # CI runs from here
pytest tests/ -v
```

### Linting Failures

```bash
# Auto-fix most issues
black --line-length 100 src/ tests/
isort --profile black --line-length 100 src/ tests/

# Check what would change
black --check --line-length 100 src/ tests/
isort --check-only --profile black --line-length 100 src/ tests/

# Run flake8
flake8 src/ tests/ --max-line-length=100
```

### Coverage Too Low

```bash
# See what's not covered
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View in browser
```

---

## 🎯 Best Practices

### Before Pushing

```bash
# 1. Run tests
pytest tests/ -v

# 2. Check linting
black --check src/ tests/
flake8 src/ tests/

# 3. Check imports
isort --check-only src/ tests/

# 4. (Optional) Run full CI suite locally
pytest tests/ -v --cov=src -n auto
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy src/
```

### Creating PRs

✅ **DO**:
- Keep PRs small (<500 lines)
- Add tests for new features
- Update documentation
- Follow .cursorrules standards
- Write clear PR description

❌ **DON'T**:
- Mix unrelated changes
- Skip tests
- Ignore linter warnings
- Commit commented code
- Push directly to main

---

## 📈 Metrics

### Target Benchmarks

| Metric | Target | Current |
|--------|--------|---------|
| Test Success Rate | >95% | ✅ |
| Code Coverage | >80% | 🎯 |
| Lint Success | 100% | ✅ |
| PR Review Time | <24h | - |
| CI Runtime | <15min | ✅ |

---

## 🚨 Common Errors

### `Module not found: src`

```bash
# Fix: Run from correct directory
cd cuda-healthcheck
pytest tests/
```

### `Import "src" could not be resolved`

```bash
# Fix: Install package in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### `pytest: command not found`

```bash
# Fix: Install pytest
pip install pytest pytest-cov pytest-xdist
```

---

## 📚 Documentation

- **Full CI/CD Guide**: [docs/CICD.md](docs/CICD.md)
- **Implementation Summary**: [CICD_IMPLEMENTATION_SUMMARY.md](CICD_IMPLEMENTATION_SUMMARY.md)
- **Project README**: [README.md](README.md)

---

## 🆘 Getting Help

**Workflow issues**:
1. Check Actions tab for error logs
2. Review [docs/CICD.md](docs/CICD.md) troubleshooting
3. Enable debug logging (add `ACTIONS_STEP_DEBUG=true` secret)

**Test failures**:
1. Run locally to reproduce
2. Check test fixtures in `tests/conftest.py`
3. Review test output for details

**Linting issues**:
1. Run auto-formatters (black, isort)
2. Check .cursorrules for standards
3. Review error messages for specific violations

---

**Last Updated**: December 2024  
**Workflows Version**: 1.0.0



