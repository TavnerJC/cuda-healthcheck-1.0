"""
Logging configuration for CUDA Healthcheck Tool.

Provides centralized logging configuration with support for different
log levels and formats for local development and Databricks environments.
"""

import logging
import os
import sys
from typing import Optional, TextIO


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    stream: Optional[TextIO] = None,
) -> None:
    """
    Configure root logger for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom log format string
        stream: Output stream (defaults to sys.stdout)
    """
    if level is None:
        level = os.getenv("CUDA_HEALTHCHECK_LOG_LEVEL", "INFO")

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if stream is None:
        stream = sys.stdout

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        stream=stream,
        force=True,
    )


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger for a module.

    Args:
        name: Logger name (typically __name__)
        level: Optional override for log level

    Returns:
        Configured logger instance.

    Example:
        ```python
        from src.utils.logging_config import get_logger

        logger = get_logger(__name__)
        logger.info("Starting CUDA detection...")
        ```
    """
    logger = logging.getLogger(name)

    # Set level from environment or parameter
    if level is None:
        level = os.getenv("CUDA_HEALTHCHECK_LOG_LEVEL", "INFO")

    logger.setLevel(getattr(logging, level.upper()))

    # Add handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_databricks_logger(name: str) -> logging.Logger:
    """
    Get a logger configured for Databricks notebooks.

    Uses simpler formatting suitable for notebook output.

    Args:
        name: Logger name

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    level = os.getenv("CUDA_HEALTHCHECK_LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Simpler format for notebooks
        formatter = logging.Formatter("%(levelname)s - %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Initialize logging on module import
setup_logging()
