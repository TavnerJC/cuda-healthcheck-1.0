# CI/CD Pipeline Documentation

**Version**: 1.0.0  
**Last Updated**: December 2024

## Overview

Comprehensive CI/CD setup for the CUDA Healthcheck Tool using GitHub Actions. Provides automated testing, code quality checks, dependency management, and release automation.

---

## Workflows

### 1. 🧪 Test Workflow (`test.yml`)

**Trigger**: Push to main/develop, Pull Requests, Manual dispatch

**Jobs**:

#### Main Test (Multi-version)
- Tests on Python 3.10, 3.11, 3.12
- Runs all unit tests in parallel
- Validates imports
- **Duration**: ~5 minutes

#### Test with Coverage
- Runs on Python 3.11
- Generates coverage reports
- Uploads to Codecov
- Adds coverage to PR summary
- **Duration**: ~3 minutes

#### Module Tests
- Tests individual modules separately
- Utils, Orchestrator, Breaking Changes, Databricks
- Isolates failures
- **Duration**: ~4 minutes

#### Compatibility Tests
- Tests CUDA version compatibility scenarios
- Validates breaking changes detection
- **Duration**: ~2 minutes

#### Integration Tests
- Tests complete workflow
- Breaking changes database validation
- **Duration**: ~2 minutes

#### Notebook Validation
- Checks notebook syntax
- Validates Python compilation
- **Duration**: ~1 minute

**Total Runtime**: ~10-15 minutes (parallel execution)

### 2. ✨ Code Quality Workflow (`code-quality.yml`)

**Trigger**: Push to main/develop, Pull Requests, Manual dispatch

**Jobs**:

#### Linting
- **flake8**: Syntax errors, undefined names
- **black**: Code formatting (line length 100)
- **isort**: Import sorting
- **Duration**: ~2 minutes

#### Type Checking
- **mypy**: Static type checking
- Ignores missing imports
- **Duration**: ~2 minutes

#### Security Scan
- **bandit**: Security vulnerability scanning
- Medium and high severity issues
- JSON report generation
- **Duration**: ~2 minutes

#### Complexity Analysis
- **radon**: Cyclomatic complexity
- Maintainability index
- **Duration**: ~1 minute

#### Documentation Check
- Validates docstrings presence
- Checks for README and docs
- **Duration**: ~1 minute

**Total Runtime**: ~5-8 minutes

### 3. 🔍 PR Checks Workflow (`pr-checks.yml`)

**Trigger**: Pull Request opened, synchronized, reopened

**Jobs**:

#### PR Information
- Displays PR title, author, branches
- Shows files changed count

#### Quick Tests
- Runs fast subset of tests
- Quick validation before full suite

#### Changed Files Analysis
- Lists all changed files
- Checks if tests were added
- Warns if no test changes

#### Requirements Check
- Displays requirements.txt
- Analyzes dependency tree
- Shows first 30 dependencies

#### PR Size Check
- Calculates additions/deletions
- Warns if PR is large (>1000 lines)
- Suggests splitting if needed

#### Auto Labeling
- Automatically labels PRs by file type
- Module-specific labels
- Type labels (src, tests, docs, etc.)

#### PR Checklist
- Provides review checklist
- Ensures standards are followed

**Total Runtime**: ~3-5 minutes

### 4. 📦 Release Workflow (`release.yml`)

**Trigger**: Tag push (v*.*.*), Manual dispatch with version

**Jobs**:

#### Create Release
- Runs full test suite
- Generates changelog
- Creates release archive (.tar.gz)
- Creates GitHub Release
- Attaches artifacts

**Usage**:
```bash
# Create and push tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Or use manual dispatch on GitHub Actions tab
```

### 5. 🌙 Nightly Build Workflow (`nightly.yml`)

**Trigger**: Daily at 2 AM UTC, Manual dispatch

**Jobs**:

#### Nightly Full Test Suite
- Runs complete test suite with coverage
- Generates HTML coverage report
- Uploads coverage artifacts

#### Compatibility Matrix
- Tests on Python 3.10, 3.11, 3.12
- Full matrix validation

#### Nightly Report
- Generates summary report
- Shows pass/fail status

---

## Configuration Files

### Dependabot (`dependabot.yml`)

**Purpose**: Automated dependency updates

**Configuration**:
- **GitHub Actions**: Weekly updates on Monday
- **Python packages**: Weekly updates on Monday
- Groups minor/patch updates
- Limits open PRs (5 for actions, 10 for Python)
- Ignores major version updates

**Labels**: `dependencies`, `github-actions`, `python`

### Labeler (`labeler.yml`)

**Purpose**: Auto-label PRs based on changed files

**Labels**:
- **Type**: src, tests, docs, config, notebooks
- **Module**: cuda-detector, databricks, healthcheck, utils, data
- **Special**: ci/cd, dependencies

---

## Setup Instructions

### 1. Enable GitHub Actions

1. Go to repository Settings → Actions
2. Select "Allow all actions and reusable workflows"
3. Save settings

### 2. Configure Secrets

**Required Secrets** (if needed):
- `CODECOV_TOKEN` - For coverage uploads (optional)
- `GITHUB_TOKEN` - Automatically provided

**Setup**:
1. Go to Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add required secrets

### 3. Enable Dependabot

1. Go to Settings → Code security and analysis
2. Enable "Dependabot alerts"
3. Enable "Dependabot security updates"
4. Enable "Dependabot version updates"

### 4. Configure Branch Protection

**Recommended for `main` branch**:
1. Go to Settings → Branches
2. Add rule for `main`
3. Enable:
   - ✅ Require pull request reviews (1 reviewer)
   - ✅ Require status checks to pass
     - `Test Python 3.11`
     - `Test with Coverage`
     - `Linting (flake8, black)`
   - ✅ Require branches to be up to date
   - ✅ Include administrators (optional)

### 5. Configure Codecov (Optional)

1. Sign up at https://codecov.io
2. Connect GitHub repository
3. Get token from Codecov dashboard
4. Add `CODECOV_TOKEN` to repository secrets

---

## Usage Guide

### Running Workflows Manually

1. Go to Actions tab
2. Select workflow
3. Click "Run workflow"
4. Choose branch
5. Click "Run workflow" button

### Viewing Results

**Test Results**:
- Go to Actions → select workflow run
- View job summaries
- Check test output
- Download artifacts if available

**Coverage Reports**:
- Click on coverage job
- View in summary or download artifact
- Check Codecov for detailed reports

### Creating a Release

**Method 1: Git Tag**
```bash
# Create annotated tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tag
git push origin v1.0.0

# Workflow triggers automatically
```

**Method 2: Manual Dispatch**
1. Go to Actions → Release workflow
2. Click "Run workflow"
3. Enter version number (e.g., 1.0.0)
4. Click "Run workflow"

---

## Monitoring

### Check Status Badges

Add to README.md:

```markdown
![Tests](https://github.com/username/repo/workflows/Tests/badge.svg)
![Code Quality](https://github.com/username/repo/workflows/Code%20Quality/badge.svg)
[![codecov](https://codecov.io/gh/username/repo/branch/main/graph/badge.svg)](https://codecov.io/gh/username/repo)
```

### Email Notifications

Configure in Settings → Notifications:
- Failed workflows
- PR reviews needed
- Dependabot alerts

### Workflow Analytics

View in Actions tab:
- Success rate over time
- Average duration
- Resource usage

---

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "Module not found"
```yaml
# Solution: Ensure working-directory is set
working-directory: cuda-healthcheck
```

**Issue**: Timeout during tests
```yaml
# Solution: Add timeout and run in parallel
- name: Run tests
  timeout-minutes: 15
  run: pytest tests/ -n auto
```

**Issue**: Dependabot PRs fail
```
# Solution: Update requirements.txt with compatible versions
# Or update tests to handle new versions
```

**Issue**: Coverage upload fails
```yaml
# Solution: Make fail_ci_if_error: false
- uses: codecov/codecov-action@v4
  with:
    fail_ci_if_error: false
```

### Debug Workflows

**Enable debug logging**:
1. Go to Settings → Secrets
2. Add `ACTIONS_STEP_DEBUG` = `true`
3. Add `ACTIONS_RUNNER_DEBUG` = `true`
4. Re-run workflow

**Access runner logs**:
1. Go to failed workflow run
2. Click on failed job
3. Click "View raw logs"
4. Download if needed

---

## Best Practices

### 1. Fast Feedback
- Quick tests run first in PRs
- Full suite runs on merge
- Nightly runs for comprehensive checks

### 2. Parallel Execution
- Use `-n auto` for pytest
- Matrix strategy for Python versions
- Independent jobs run in parallel

### 3. Caching
- Pip cache enabled
- Reduces install time by ~50%

### 4. Fail Fast
- `fail-fast: false` in matrix
- See all failures, not just first

### 5. Resource Efficiency
- Use ubuntu-latest (fastest)
- Only run necessary jobs
- Cache dependencies

### 6. Security
- Dependabot for vulnerabilities
- Bandit for code scanning
- No secrets in logs

---

## Maintenance

### Monthly Tasks
- Review Dependabot PRs
- Update Python versions if new release
- Review and merge successful nightly builds

### Quarterly Tasks
- Review workflow efficiency
- Update GitHub Actions versions
- Audit security scan results
- Update documentation

### Yearly Tasks
- Review overall CI/CD strategy
- Consider new tools/actions
- Update best practices

---

## Metrics

**Target Metrics**:
- ✅ Test success rate: >95%
- ✅ Code coverage: >80%
- ✅ PR review time: <24 hours
- ✅ Time to deployment: <1 hour
- ✅ Mean time to recovery: <2 hours

**Tracking**:
- GitHub Actions dashboard
- Codecov dashboard
- Custom analytics (if needed)

---

## Advanced Configuration

### Custom Test Runners

```yaml
# Run specific test categories
- name: Run smoke tests
  run: pytest tests/ -m smoke

- name: Run integration tests
  run: pytest tests/ -m integration
```

### Conditional Workflows

```yaml
# Only run on specific paths
on:
  push:
    paths:
      - 'src/**'
      - 'tests/**'
```

### Artifact Management

```yaml
# Upload test results
- uses: actions/upload-artifact@v4
  with:
    name: test-results
    path: test-results/
    retention-days: 30
```

### Slack Notifications

```yaml
# Add to workflow
- name: Notify Slack
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
  if: always()
```

---

## Summary

✅ **5 GitHub Actions workflows**  
✅ **Automated testing on 3 Python versions**  
✅ **Code quality checks (linting, security, complexity)**  
✅ **Automated dependency updates**  
✅ **Release automation**  
✅ **Nightly builds**  
✅ **Auto PR labeling**  
✅ **Comprehensive coverage reporting**  

**The CI/CD pipeline ensures code quality, reliability, and automated workflows for the entire development lifecycle!** 🚀


