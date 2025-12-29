# Environment Variables Configuration

This document describes all environment variables used by the CUDA Healthcheck Tool.

## Required for Databricks Integration

### `DATABRICKS_HOST`
- **Description**: URL of your Databricks workspace
- **Format**: `https://<workspace-id>.cloud.databricks.com`
- **Example**: `https://dbc-1234567-89ab.cloud.databricks.com`
- **Required for**: DatabricksConnector, cluster scanning, Delta table operations
- **Default**: None (must be set)

### `DATABRICKS_TOKEN`
- **Description**: Personal Access Token for Databricks authentication
- **Format**: String token starting with `dapi`
- **Example**: `dapi1234567890abcdef1234567890ab`
- **Required for**: All Databricks API operations
- **Default**: None (must be set)
- **Security**: Store in secrets manager, never commit to version control

**How to generate**:
1. Log into your Databricks workspace
2. Go to Settings → User Settings → Access Tokens
3. Click "Generate New Token"
4. Copy the token immediately (it won't be shown again)

## Optional Configuration

### `DATABRICKS_WAREHOUSE_ID`
- **Description**: SQL Warehouse ID for Delta table operations
- **Format**: Alphanumeric string
- **Example**: `abc123def456gh789`
- **Required for**: Reading/writing Delta tables via SQL
- **Default**: None (Delta operations will fail without this)

**How to find**:
1. Go to SQL → SQL Warehouses in Databricks
2. Click on your warehouse
3. The ID is in the URL: `/sql/warehouses/<warehouse-id>`

### `CUDA_HEALTHCHECK_LOG_LEVEL`
- **Description**: Logging verbosity level
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Default**: `INFO`
- **Example**: `export CUDA_HEALTHCHECK_LOG_LEVEL=DEBUG`

**Recommended settings**:
- Development: `DEBUG` - Shows all details including command outputs
- Testing: `INFO` - Standard information messages
- Production: `WARNING` - Only warnings and errors

### `CUDA_HEALTHCHECK_RETRY_ATTEMPTS`
- **Description**: Number of retry attempts for API calls
- **Type**: Integer
- **Range**: 1-10
- **Default**: `3`
- **Example**: `export CUDA_HEALTHCHECK_RETRY_ATTEMPTS=5`

### `CUDA_HEALTHCHECK_TIMEOUT`
- **Description**: Timeout in seconds for long-running operations
- **Type**: Integer (seconds)
- **Range**: 30-600
- **Default**: `300` (5 minutes)
- **Example**: `export CUDA_HEALTHCHECK_TIMEOUT=600`

## Environment-Specific Variables

### `DATABRICKS_RUNTIME_VERSION`
- **Description**: Databricks Runtime version (automatically set in Databricks)
- **Format**: `X.Y.x-gpu-ml-scala2.12`
- **Example**: `13.3.x-gpu-ml-scala2.12`
- **Set by**: Databricks environment (do not set manually)
- **Used for**: Detecting if running in Databricks

## Configuration Methods

### 1. Environment Variables (Recommended for CI/CD)

```bash
# Linux/Mac
export DATABRICKS_HOST="https://dbc-xxxxx.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi1234567890abcdef"
export DATABRICKS_WAREHOUSE_ID="abc123def456"
export CUDA_HEALTHCHECK_LOG_LEVEL="INFO"
```

```powershell
# Windows PowerShell
$env:DATABRICKS_HOST="https://dbc-xxxxx.cloud.databricks.com"
$env:DATABRICKS_TOKEN="dapi1234567890abcdef"
$env:DATABRICKS_WAREHOUSE_ID="abc123def456"
$env:CUDA_HEALTHCHECK_LOG_LEVEL="INFO"
```

### 2. `.env` File (Recommended for Local Development)

Create a `.env` file in the project root:

```bash
# .env file
DATABRICKS_HOST=https://dbc-xxxxx.cloud.databricks.com
DATABRICKS_TOKEN=dapi1234567890abcdef
DATABRICKS_WAREHOUSE_ID=abc123def456
CUDA_HEALTHCHECK_LOG_LEVEL=DEBUG
CUDA_HEALTHCHECK_RETRY_ATTEMPTS=3
CUDA_HEALTHCHECK_TIMEOUT=300
```

Load with python-dotenv:

```python
from dotenv import load_dotenv
load_dotenv()

from src.databricks import DatabricksConnector
connector = DatabricksConnector()  # Uses .env credentials
```

### 3. Databricks Secrets (Recommended for Production)

Store credentials in Databricks secret scopes:

```python
# In Databricks notebook
token = dbutils.secrets.get(scope="cuda-healthcheck", key="databricks-token")
host = dbutils.secrets.get(scope="cuda-healthcheck", key="databricks-host")

from src.databricks import DatabricksConnector
connector = DatabricksConnector(workspace_url=host, token=token)
```

**Setup Databricks secrets**:
```bash
# Create secret scope
databricks secrets create-scope --scope cuda-healthcheck

# Add secrets
databricks secrets put --scope cuda-healthcheck --key databricks-token
databricks secrets put --scope cuda-healthcheck --key databricks-host
```

### 4. Direct Instantiation (For Testing Only)

```python
from src.databricks import DatabricksConnector

connector = DatabricksConnector(
    workspace_url="https://test.databricks.com",
    token="dapi_test_token"
)
```

**⚠️ Warning**: Never hardcode credentials in production code!

## Security Best Practices

### 1. Never Commit Credentials
```bash
# Add to .gitignore
.env
*.env
.env.local
credentials.json
```

### 2. Use Secrets Management
- **Local development**: `.env` file (gitignored)
- **CI/CD**: GitHub Secrets, Azure Key Vault, AWS Secrets Manager
- **Databricks**: Databricks Secret Scopes

### 3. Rotate Tokens Regularly
- Generate new tokens every 90 days
- Revoke old tokens immediately after rotation
- Use service principals in production (not personal tokens)

### 4. Principle of Least Privilege
Grant only necessary permissions:
- Cluster read access (for healthcheck)
- Delta table read/write (for results storage)
- No admin privileges required

### 5. Token Scope Restriction
When creating tokens, limit scope to:
- Workspace access
- SQL Analytics (if using Delta tables)
- No Jobs or Notebooks execution (unless required)

## Validation

Test your configuration:

```python
# test_config.py
import os
from src.databricks import DatabricksConnector, is_databricks_environment

def validate_config():
    """Validate environment variables are set correctly."""
    required = ["DATABRICKS_HOST", "DATABRICKS_TOKEN"]
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing required variables: {', '.join(missing)}")
        return False
    
    # Test connection
    try:
        connector = DatabricksConnector()
        clusters = connector.list_clusters()
        print(f"✓ Connected successfully! Found {len(clusters)} clusters")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    validate_config()
```

Run validation:
```bash
python test_config.py
```

## Troubleshooting

### Error: "credentials not provided"
- Check that `DATABRICKS_HOST` and `DATABRICKS_TOKEN` are set
- Verify no typos in variable names
- On Windows, restart terminal after setting environment variables

### Error: "SDK not installed"
```bash
pip install databricks-sdk
```

### Error: "Cluster not found"
- Verify cluster ID is correct
- Check token has permission to access the cluster
- Ensure cluster exists in the workspace

### Error: "DATABRICKS_WAREHOUSE_ID environment variable not set"
- Set the variable or specify warehouse ID in code
- Only required for Delta table operations

### Token Authentication Failed
- Token may have expired (check Databricks console)
- Token may have been revoked
- Generate a new token and update environment variable

## Example Configurations

### Development Environment
```bash
# .env.development
DATABRICKS_HOST=https://dev-workspace.databricks.com
DATABRICKS_TOKEN=dapi_dev_token_123
DATABRICKS_WAREHOUSE_ID=dev_warehouse_id
CUDA_HEALTHCHECK_LOG_LEVEL=DEBUG
CUDA_HEALTHCHECK_RETRY_ATTEMPTS=2
CUDA_HEALTHCHECK_TIMEOUT=120
```

### Production Environment
```bash
# .env.production (stored in secrets manager, not in repo)
DATABRICKS_HOST=https://prod-workspace.databricks.com
DATABRICKS_TOKEN=dapi_prod_token_789
DATABRICKS_WAREHOUSE_ID=prod_warehouse_id
CUDA_HEALTHCHECK_LOG_LEVEL=WARNING
CUDA_HEALTHCHECK_RETRY_ATTEMPTS=5
CUDA_HEALTHCHECK_TIMEOUT=600
```

### CI/CD Pipeline (GitHub Actions)
```yaml
# .github/workflows/healthcheck.yml
env:
  DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
  DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
  DATABRICKS_WAREHOUSE_ID: ${{ secrets.DATABRICKS_WAREHOUSE_ID }}
  CUDA_HEALTHCHECK_LOG_LEVEL: INFO
```

## Quick Reference

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DATABRICKS_HOST` | Yes | None | Workspace URL |
| `DATABRICKS_TOKEN` | Yes | None | Authentication |
| `DATABRICKS_WAREHOUSE_ID` | No* | None | Delta operations |
| `CUDA_HEALTHCHECK_LOG_LEVEL` | No | INFO | Logging level |
| `CUDA_HEALTHCHECK_RETRY_ATTEMPTS` | No | 3 | API retries |
| `CUDA_HEALTHCHECK_TIMEOUT` | No | 300 | Operation timeout |

\* Required only for Delta table read/write operations

---

For more information, see:
- [SETUP.md](SETUP.md) - Initial setup instructions
- [DATABRICKS_INTEGRATION.md](DATABRICKS_INTEGRATION.md) - Databricks-specific guide
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guidelines



