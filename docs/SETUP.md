# Setup Instructions

## Quick Start

1. **Clone and Install**
```bash
git clone <your-repo>
cd cuda-healthcheck
pip install -r requirements.txt
```

2. **Configure Databricks**
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Run Local Test**
```bash
python main.py detect
```

4. **Run Complete Healthcheck**
```bash
python main.py healthcheck
```

5. **Scan Databricks Clusters**
```bash
python main.py scan
```

## Databricks Setup

### 1. Generate Personal Access Token

1. Log in to your Databricks workspace
2. Click on your username → Settings
3. Go to Developer → Access tokens
4. Click "Generate new token"
5. Copy the token (you won't see it again!)

### 2. Find Warehouse ID

For Delta table operations, you need a SQL warehouse:

1. Go to SQL Warehouses in Databricks
2. Click on your warehouse
3. Copy the ID from the URL: `/sql/warehouses/<warehouse-id>`

### 3. Set Environment Variables

```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi1234567890abcdef"
export DATABRICKS_WAREHOUSE_ID="abc123xyz456"
```

Or create a `.env` file (recommended).

## Development Setup

### Install Development Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_detector.py -v
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/

# Lint
flake8 src/ tests/
```

## Databricks Connect (Optional)

For local development with Databricks clusters:

```bash
pip install databricks-connect

# Configure
databricks-connect configure

# Test connection
databricks-connect test
```

## Troubleshooting

### Issue: `nvidia-smi: command not found`
**Solution**: Tool requires NVIDIA GPU and drivers. Run on GPU-enabled cluster or machine.

### Issue: Import errors for torch/tensorflow/cudf
**Solution**: These are optional. The tool detects if they're installed but doesn't require them.

### Issue: Databricks authentication failed
**Solution**: 
1. Check token is valid
2. Verify workspace URL is correct
3. Ensure token has cluster access permissions

### Issue: Delta table creation fails
**Solution**: 
1. Verify DATABRICKS_WAREHOUSE_ID is set
2. Check Unity Catalog permissions
3. Try running without Delta table: `python main.py scan --no-delta`

## Next Steps

1. Read the [Migration Guide](docs/MIGRATION_GUIDE.md)
2. Understand [Breaking Changes](docs/BREAKING_CHANGES.md)
3. Set up CI/CD with GitHub Actions
4. Configure automated scanning schedule









