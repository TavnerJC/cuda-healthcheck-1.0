"""Setup script for CUDA Healthcheck Tool."""

from setuptools import find_packages, setup

# Read version from cuda_healthcheck/__init__.py
version = {}
with open("cuda_healthcheck/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

# Read long description from README
try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "CUDA Healthcheck Tool for Databricks GPU Clusters"

setup(
    name="cuda-healthcheck",
    version=version.get("__version__", "1.0.0"),
    author="NVIDIA - CUDA Healthcheck Team",
    description="CUDA version compatibility checker for Databricks GPU clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TavnerJC/cuda-healthcheck-1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "databricks-sdk>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "bandit>=1.7.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Topic :: System :: Hardware",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="cuda gpu databricks healthcheck compatibility nvidia",
    project_urls={
        "Bug Reports": "https://github.com/TavnerJC/cuda-healthcheck-1.0/issues",
        "Source": "https://github.com/TavnerJC/cuda-healthcheck-1.0",
        "Documentation": "https://github.com/TavnerJC/cuda-healthcheck-1.0/blob/main/README.md",
    },
)
