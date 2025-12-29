"""
Unit tests for retry utilities.

Tests can be run locally without any dependencies on Databricks or CUDA.
"""

import time
from unittest.mock import Mock

import pytest

from src.utils.retry import retry_on_failure, retry_with_timeout


class TestRetryOnFailure:
    """Test suite for retry_on_failure decorator."""

    def test_retry_on_failure_success_first_try(self):
        """Test decorator when function succeeds on first try."""
        mock_func = Mock(return_value="success")
        decorated = retry_on_failure(max_attempts=3)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_failure_success_after_retries(self):
        """Test decorator when function succeeds after failures."""
        call_count = [0]

        def failing_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return "success"

        decorated = retry_on_failure(max_attempts=5, delay=0.01)(failing_function)
        result = decorated()

        assert result == "success"
        assert call_count[0] == 3

    def test_retry_on_failure_all_attempts_fail(self):
        """Test decorator when all attempts fail."""
        mock_func = Mock(side_effect=Exception("Persistent failure"))
        decorated = retry_on_failure(max_attempts=3, delay=0.01)(mock_func)

        with pytest.raises(Exception, match="Persistent failure"):
            decorated()

        assert mock_func.call_count == 3

    def test_retry_on_failure_respects_max_attempts(self):
        """Test that decorator respects max_attempts parameter."""
        mock_func = Mock(side_effect=Exception("Always fails"))
        decorated = retry_on_failure(max_attempts=5, delay=0.01)(mock_func)

        with pytest.raises(Exception):
            decorated()

        assert mock_func.call_count == 5

    def test_retry_on_failure_exponential_backoff(self):
        """Test that retry delay increases exponentially."""
        call_times = []

        def track_time_function():
            call_times.append(time.time())
            raise Exception("Test failure")

        decorated = retry_on_failure(max_attempts=3, delay=0.1, backoff=2.0)(
            track_time_function
        )

        with pytest.raises(Exception):
            decorated()

        assert len(call_times) == 3
        # Check that delays are increasing (approximately)
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            # Second delay should be roughly 2x first delay
            assert delay2 > delay1

    def test_retry_on_failure_specific_exceptions(self):
        """Test that decorator only catches specified exceptions."""

        def raises_value_error():
            raise ValueError("Not caught")

        decorated = retry_on_failure(max_attempts=3, delay=0.01, exceptions=(IOError,))(
            raises_value_error
        )

        # ValueError should not be caught and retried
        with pytest.raises(ValueError):
            decorated()

    def test_retry_on_failure_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""

        def original_function():
            """Original docstring."""
            return "result"

        decorated = retry_on_failure()(original_function)

        assert decorated.__name__ == "original_function"
        assert "Original docstring" in decorated.__doc__

    def test_retry_on_failure_with_arguments(self):
        """Test decorator with functions that take arguments."""
        mock_func = Mock(return_value="success")
        decorated = retry_on_failure(max_attempts=3)(mock_func)

        result = decorated("arg1", "arg2", kwarg1="value1")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")


class TestRetryWithTimeout:
    """Test suite for retry_with_timeout function."""

    def test_retry_with_timeout_success(self):
        """Test function succeeds within timeout."""
        mock_func = Mock(return_value="success")

        result = retry_with_timeout(mock_func, timeout=5.0, max_attempts=3, delay=0.01)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_with_timeout_success_after_retries(self):
        """Test function succeeds after retries within timeout."""
        call_count = [0]

        def eventually_succeeds():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return "success"

        result = retry_with_timeout(
            eventually_succeeds, timeout=5.0, max_attempts=5, delay=0.01
        )

        assert result == "success"
        assert call_count[0] == 3

    def test_retry_with_timeout_exceeds_timeout(self):
        """Test function times out."""

        def slow_function():
            time.sleep(0.2)
            raise Exception("Too slow")

        result = retry_with_timeout(
            slow_function, timeout=0.3, max_attempts=10, delay=0.1
        )

        # Should return None on timeout
        assert result is None

    def test_retry_with_timeout_max_attempts(self):
        """Test function fails after max attempts within timeout."""
        mock_func = Mock(side_effect=Exception("Always fails"))

        result = retry_with_timeout(mock_func, timeout=5.0, max_attempts=3, delay=0.01)

        assert result is None
        assert mock_func.call_count == 3

    def test_retry_with_timeout_respects_remaining_time(self):
        """Test that retry respects remaining timeout."""
        call_count = [0]
        start_time = time.time()

        def delayed_function():
            call_count[0] += 1
            time.sleep(0.1)
            raise Exception("Test failure")

        result = retry_with_timeout(
            delayed_function, timeout=0.25, max_attempts=10, delay=0.05
        )

        elapsed = time.time() - start_time
        assert result is None
        assert elapsed < 0.5  # Should stop before using all attempts

    def test_retry_with_timeout_specific_exceptions(self):
        """Test that only specified exceptions are retried."""

        def raises_value_error():
            raise ValueError("Not caught")

        # ValueError is not in the exceptions tuple, so it should propagate
        with pytest.raises(ValueError, match="Not caught"):
            retry_with_timeout(
                raises_value_error,
                timeout=5.0,
                max_attempts=3,
                delay=0.01,
                exceptions=(IOError,),
            )


class TestRetryIntegration:
    """Integration tests for retry utilities."""

    def test_retry_decorator_in_real_scenario(self):
        """Test retry decorator in realistic scenario."""
        api_call_count = [0]

        @retry_on_failure(max_attempts=5, delay=0.01, backoff=1.5)
        def flaky_api_call():
            """Simulates flaky API that fails 2 times then succeeds."""
            api_call_count[0] += 1
            if api_call_count[0] < 3:
                raise ConnectionError("Network error")
            return {"status": "success", "data": [1, 2, 3]}

        result = flaky_api_call()

        assert result["status"] == "success"
        assert api_call_count[0] == 3

    def test_retry_with_lambda(self):
        """Test retry with lambda functions."""
        counter = [0]

        def increment_and_fail():
            counter[0] += 1
            if counter[0] < 3:
                raise Exception("Not yet")
            return counter[0]

        decorated = retry_on_failure(max_attempts=5, delay=0.01)(increment_and_fail)
        result = decorated()

        assert result == 3
        assert counter[0] == 3

    def test_nested_retry_decorators(self):
        """Test that retry decorators can be nested."""
        call_count = [0]

        @retry_on_failure(max_attempts=2, delay=0.01)
        @retry_on_failure(max_attempts=2, delay=0.01)
        def double_retry_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("Fail once")
            return "success"

        result = double_retry_function()
        assert result == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
