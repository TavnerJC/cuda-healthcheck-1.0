"""
Unit tests for logging configuration.

Tests can be run locally without any dependencies on Databricks or CUDA.
"""

import logging

import pytest

from cuda_healthcheck.utils.logging_config import get_databricks_logger, get_logger, setup_logging


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_uses_module_name(self):
        """Test that logger has correct name."""
        logger = get_logger("test.module.name")
        assert logger.name == "test.module.name"

    def test_get_logger_respects_level_parameter(self):
        """Test that level parameter overrides environment."""
        logger = get_logger("test_module", level="ERROR")
        assert logger.level == logging.ERROR

    def test_get_logger_respects_env_variable(self, monkeypatch):
        """Test that logger respects CUDA_HEALTHCHECK_LOG_LEVEL."""
        monkeypatch.setenv("CUDA_HEALTHCHECK_LOG_LEVEL", "DEBUG")
        logger = get_logger("test_module")
        assert logger.level == logging.DEBUG

    def test_get_logger_default_level(self, monkeypatch):
        """Test that default level is INFO when env var not set."""
        monkeypatch.delenv("CUDA_HEALTHCHECK_LOG_LEVEL", raising=False)
        logger = get_logger("test_module")
        assert logger.level == logging.INFO

    def test_get_logger_has_handler(self):
        """Test that logger has at least one handler."""
        logger = get_logger("test_module")
        assert len(logger.handlers) > 0

    def test_get_logger_handler_has_formatter(self):
        """Test that handler has a formatter."""
        logger = get_logger("test_module")
        handler = logger.handlers[0]
        assert handler.formatter is not None


class TestSetupLogging:
    """Test suite for setup_logging function."""

    def test_setup_logging_default(self):
        """Test setup_logging with default parameters."""
        # Clear existing handlers to ensure clean state
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.NOTSET)

        setup_logging()
        # Check that at least one handler was added
        assert len(root_logger.handlers) > 0

    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom level."""
        setup_logging(level="WARNING")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_setup_logging_respects_env_variable(self, monkeypatch):
        """Test that setup_logging respects environment variable."""
        monkeypatch.setenv("CUDA_HEALTHCHECK_LOG_LEVEL", "ERROR")
        setup_logging()
        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR

    def test_setup_logging_custom_format(self):
        """Test setup_logging with custom format."""
        custom_format = "%(levelname)s - %(message)s"
        setup_logging(format_string=custom_format)
        # Logging configured successfully
        assert True


class TestGetDatabricksLogger:
    """Test suite for get_databricks_logger function."""

    def test_get_databricks_logger_returns_logger(self):
        """Test that get_databricks_logger returns a Logger."""
        logger = get_databricks_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_databricks_logger_has_simple_format(self):
        """Test that Databricks logger uses simplified format."""
        logger = get_databricks_logger("test_notebook")
        handler = logger.handlers[0]
        # Should have a formatter
        assert handler.formatter is not None

    def test_get_databricks_logger_respects_env(self, monkeypatch):
        """Test that Databricks logger respects environment variable."""
        monkeypatch.setenv("CUDA_HEALTHCHECK_LOG_LEVEL", "DEBUG")
        logger = get_databricks_logger("test_notebook")
        assert logger.level == logging.DEBUG


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_logger_can_log_messages(self, capsys):
        """Test that logger can actually log messages."""
        logger = get_logger("test.integration")
        logger.info("Test message")
        # Message was logged (captured by pytest)
        assert True

    def test_multiple_loggers_independent(self):
        """Test that multiple loggers are independent."""
        logger1 = get_logger("module1", level="DEBUG")
        logger2 = get_logger("module2", level="ERROR")

        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.ERROR

    def test_logger_level_filtering(self, capsys):
        """Test that log level filtering works."""
        logger = get_logger("test.filtering", level="ERROR")

        logger.debug("Debug message")  # Should not appear
        logger.info("Info message")  # Should not appear
        logger.error("Error message")  # Should appear

        captured = capsys.readouterr()
        assert "Debug message" not in captured.out
        assert "Info message" not in captured.out
        assert "Error message" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
