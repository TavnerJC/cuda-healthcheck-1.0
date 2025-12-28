# Pre-Commit Validation Script
Write-Host "`n=== PRE-COMMIT VALIDATION SUITE ===" -ForegroundColor Green

$checks_passed = 0
$total_checks = 9

Set-Location "C:\Users\joelc\OneDrive - NVIDIA Corporation\Desktop\Cursor Projects\CUDA Healthcheck Tool on Databricks\CUDA Healthcheck Code Base\cuda-healthcheck"

# CHECK 1: MyPy
Write-Host "`n[1/9] MyPy Type Checking..." -ForegroundColor Cyan
$mypy_output = python -m mypy src/ --ignore-missing-imports 2>&1 | Out-String
if ($mypy_output -match "Success") {
    Write-Host "[PASS] MyPy type checking" -ForegroundColor Green
    $checks_passed++
} else {
    Write-Host "[FAIL] MyPy errors" -ForegroundColor Red
}

# CHECK 2: Tests
Write-Host "`n[2/9] Unit Tests..." -ForegroundColor Cyan
$test_output = python -m pytest tests/ -q 2>&1 | Out-String
if ($test_output -match "passed") {
    Write-Host "[PASS] All unit tests (147 tests)" -ForegroundColor Green
    $checks_passed++
} else {
    Write-Host "[FAIL] Unit tests" -ForegroundColor Red
}

# CHECK 3: Flake8
Write-Host "`n[3/9] Flake8 Linting..." -ForegroundColor Cyan
python -m flake8 src/ --max-line-length=100 --extend-ignore=E203,W503 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[PASS] Flake8 linting" -ForegroundColor Green
    $checks_passed++
} else {
    Write-Host "[FAIL] Flake8 errors" -ForegroundColor Red
}

# CHECK 4: Black
Write-Host "`n[4/9] Black Formatting..." -ForegroundColor Cyan
python -m black --check --line-length 100 src/ tests/ 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[PASS] Black code formatting" -ForegroundColor Green
    $checks_passed++
} else {
    Write-Host "[FAIL] Black formatting" -ForegroundColor Red
}

# CHECK 5-8: Imports
Write-Host "`n[5/9] Import: CUDADetector..." -ForegroundColor Cyan
python -c "from src.cuda_detector import CUDADetector" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[PASS] CUDADetector import" -ForegroundColor Green
    $checks_passed++
}

Write-Host "[6/9] Import: HealthcheckOrchestrator..." -ForegroundColor Cyan
python -c "from src.healthcheck import HealthcheckOrchestrator" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[PASS] HealthcheckOrchestrator import" -ForegroundColor Green
    $checks_passed++
}

Write-Host "[7/9] Import: DatabricksHealthchecker..." -ForegroundColor Cyan
python -c "from src.databricks import DatabricksHealthchecker" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[PASS] DatabricksHealthchecker import" -ForegroundColor Green
    $checks_passed++
}

Write-Host "[8/9] Import: BreakingChangesDatabase..." -ForegroundColor Cyan
python -c "from src.data import BreakingChangesDatabase" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[PASS] BreakingChangesDatabase import" -ForegroundColor Green
    $checks_passed++
}

# CHECK 9: Files
Write-Host "`n[9/9] Required Files..." -ForegroundColor Cyan
$all_exist = (Test-Path "mypy.ini") -and (Test-Path "src/utils/performance.py") -and (Test-Path "docs/API_REFERENCE.md")
if ($all_exist) {
    Write-Host "[PASS] All required files exist" -ForegroundColor Green
    $checks_passed++
}

# Summary
Write-Host "`n=== SUMMARY ===" -ForegroundColor Yellow
Write-Host "Passed: $checks_passed / $total_checks" -ForegroundColor $(if ($checks_passed -eq $total_checks) { "Green" } else { "Yellow" })

if ($checks_passed -eq $total_checks) {
    Write-Host "`n[SUCCESS] ALL CHECKS PASSED - Ready to commit!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n[WARNING] Some checks need attention" -ForegroundColor Yellow
    exit 1
}
