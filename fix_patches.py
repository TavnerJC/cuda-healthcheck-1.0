import os
import re

test_files = [
    'tests/test_detector.py',
    'tests/databricks/test_databricks_connector.py',
    'tests/databricks/test_databricks_integration.py',
    'tests/test_orchestrator.py',
]

for filepath in test_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Replace @patch decorators and patch calls
    new_content = re.sub(r'@patch\(["\']src\.', r'@patch("cuda_healthcheck.', content)
    new_content = re.sub(r'patch\(["\']src\.', r'patch("cuda_healthcheck.', new_content)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f'Updated patches in: {filepath}')

print('All patches updated!')

