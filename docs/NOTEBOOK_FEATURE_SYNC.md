# Notebook Feature Sync System

## 🎯 Purpose

Automatically detect and integrate new features into the enhanced Databricks notebook (`notebooks/01_cuda_environment_validation_enhanced.py`).

---

## 🚀 Quick Check

### Option 1: Automated Script (Recommended)

```bash
python scripts/update_notebook_features.py
```

**Output:**
```
🔍 Scanning for public API functions...
   Found 12 functions

📓 Checking notebook usage...
   Used: 10, Unused: 2

📝 Report saved to: NOTEBOOK_FEATURE_SYNC_REPORT.md

================================================================================
SUMMARY
================================================================================
✅ Used Functions: 10
⚠️  Unused Functions: 2

💡 Consider adding these features to the notebook:
   - some_new_function
   - another_new_function
================================================================================
```

### Option 2: Manual Check

```bash
# Check what's exported from databricks module
grep -A 20 "__all__" cuda_healthcheck/databricks/__init__.py

# Check what's used in notebook
grep "from cuda_healthcheck.databricks import" notebooks/01_cuda_environment_validation_enhanced.py
```

---

## 📋 Workflow: Adding New Features

### Step 1: Add Feature to Codebase

```python
# Example: Add new function to cuda_healthcheck/databricks/runtime_detector.py
def get_pytorch_compatibility(runtime_version: float) -> Dict[str, Any]:
    """Check PyTorch compatibility for runtime."""
    # ... implementation ...
    pass
```

### Step 2: Export in __init__.py

```python
# cuda_healthcheck/databricks/__init__.py
from .runtime_detector import (
    detect_databricks_runtime,
    get_driver_version_for_runtime,
    get_pytorch_compatibility,  # ← Add new function
)

__all__ = [
    ...
    "get_pytorch_compatibility",  # ← Add to exports
]
```

### Step 3: Run Feature Sync Check

```bash
python scripts/update_notebook_features.py
```

### Step 4: Review Generated Report

Open `NOTEBOOK_FEATURE_SYNC_REPORT.md` to see:
- Which features are used
- Which features are missing
- Suggested code snippets

### Step 5: Update Notebook

Add a new cell to the notebook:

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## 🐍 Step X: PyTorch Compatibility Check (NEW!)
# MAGIC
# MAGIC Check if PyTorch version is compatible with this runtime.

# COMMAND ----------
from cuda_healthcheck.databricks import get_pytorch_compatibility

# Check PyTorch compatibility
pytorch_compat = get_pytorch_compatibility(runtime_info['runtime_version'])

print(f"PyTorch Compatibility: {pytorch_compat['is_compatible']}")
if not pytorch_compat['is_compatible']:
    print(f"⚠️  {pytorch_compat['error_message']}")
```

### Step 6: Update Notebook Header

Update the feature list in the first cell:

```markdown
## What This Notebook Does:

...
9. ✅ **Checks PyTorch compatibility (NEW!)** 🎉
...
```

### Step 7: Re-run Feature Sync Check

```bash
python scripts/update_notebook_features.py
```

Should now show the feature as used!

---

## 🎓 Best Practices

### 1. Add Features Incrementally

**Don't:**
```python
# Adding 10 features at once without updating notebook
```

**Do:**
```python
# Add 1-2 features at a time
# Update notebook immediately
# Test in Databricks
# Commit together
```

### 2. Use Descriptive Section Headers

**Format:**
```markdown
## 🔧 Step X: Feature Name (NEW!)

Brief description of what this feature does and why it matters.
```

### 3. Include Context in Notebook

```python
# Always explain WHY the feature is important
print("This check is CRITICAL because:")
print("  • Users cannot upgrade drivers on Databricks")
print("  • PyTorch 2.4+ requires driver ≥ 550")
print("  • Runtime 14.3 has driver 535 (immutable)")
```

### 4. Link to Documentation

```python
print("\n📚 Learn more:")
print("  • Docs: https://github.com/TavnerJC/cuda-healthcheck-on-databricks/blob/main/docs/DRIVER_VERSION_MAPPING.md")
print("  • GitHub: https://github.com/TavnerJC/cuda-healthcheck-on-databricks")
```

---

## 🔄 CI/CD Integration (Future)

### GitHub Action to Auto-Check

Create `.github/workflows/notebook-sync-check.yml`:

```yaml
name: Notebook Feature Sync Check

on:
  push:
    paths:
      - 'cuda_healthcheck/databricks/__init__.py'
      - 'cuda_healthcheck/databricks/*.py'

jobs:
  check-notebook-sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Run feature sync check
        run: |
          python scripts/update_notebook_features.py
      
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: feature-sync-report
          path: NOTEBOOK_FEATURE_SYNC_REPORT.md
      
      - name: Comment on PR (if applicable)
        if: github.event_name == 'pull_request'
        run: |
          # Post report as PR comment
          cat NOTEBOOK_FEATURE_SYNC_REPORT.md >> $GITHUB_STEP_SUMMARY
```

---

## 📊 Feature Tracking

### Current Features (v0.5.0)

| Feature | Added | In Notebook? | Step |
|---------|-------|--------------|------|
| `detect_databricks_runtime` | v0.5.0 | ✅ Yes | Step 4 |
| `get_driver_version_for_runtime` | v0.5.0 | ✅ Yes | Step 4 |
| `check_driver_compatibility` | v0.5.0 | ✅ Yes | Step 4 |
| `get_runtime_info_summary` | v0.5.0 | ❌ No | - |
| `detect_gpu_auto` | v0.1.0 | ✅ Yes | Step 2 |
| `is_serverless_environment` | v0.1.0 | ❌ No | - |

### Feature Categories

**High Priority (Must Include):**
- Runtime detection
- Driver compatibility
- PyTorch/CuOPT incompatibilities
- Breaking changes

**Medium Priority (Should Include):**
- Convenience functions (summaries)
- Advanced compatibility checks
- Diagnostic helpers

**Low Priority (Optional):**
- Low-level helpers
- Internal utilities

---

## 🎯 Checklist: Adding New Features

### For Developers

- [ ] Implement feature in appropriate module
- [ ] Add unit tests (>= 90% coverage)
- [ ] Add docstring with examples
- [ ] Export in `__init__.py`
- [ ] Run `python scripts/update_notebook_features.py`
- [ ] Review generated report
- [ ] Add feature to notebook (if high/medium priority)
- [ ] Update notebook header/feature list
- [ ] Test in Databricks
- [ ] Update documentation
- [ ] Commit notebook + code together

### For Reviewers

- [ ] Check feature sync report
- [ ] Verify notebook includes new features
- [ ] Test notebook in Databricks
- [ ] Confirm documentation is updated

---

## 🐛 Troubleshooting

### Issue: Feature not detected

**Problem:** New function not showing in report

**Solution:**
```bash
# Check if exported
grep "your_function" cuda_healthcheck/databricks/__init__.py

# Check __all__ list
python -c "from cuda_healthcheck.databricks import __all__; print(__all__)"
```

### Issue: False positive (feature shows as unused)

**Problem:** Function is used but not detected

**Solution:**
```bash
# Check import statement format
grep "from cuda_healthcheck.databricks import" notebooks/01_cuda_environment_validation_enhanced.py

# The script looks for:
# - "import function_name"
# - "from cuda_healthcheck.databricks import function_name"
# - "function_name("
```

---

## 📚 Related Documents

- `DRIVER_MAPPING_IMPLEMENTATION_SUMMARY.md` - Driver mapping feature details
- `CODE_QUALITY_PRE_FLIGHT_CHECKLIST.md` - Quality checks before push
- `QUICK_REFERENCE_PRE_PUSH.md` - Pre-push commands

---

## 🎉 Success Criteria

**Notebook is synced when:**
- ✅ All high-priority features are included
- ✅ Feature sync report shows < 2 unused features
- ✅ Notebook runs successfully in Databricks
- ✅ All features have explanatory text
- ✅ Documentation is updated

---

**Maintained by:** Cursor AI Assistant  
**Last Updated:** January 1, 2026  
**Version:** 0.5.0

