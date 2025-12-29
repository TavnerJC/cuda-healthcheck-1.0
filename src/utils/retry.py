"""
Retry utilities for handling transient failures.

Provides decorators and functions for retrying operations with
exponential backoff, particularly useful for API calls and cluster operations.
"""

import functools
import time
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., Optional[T]]]:
    """
    Decorator to retry a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each attempt
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function that retries on failure.

    Example:
        ```python
        @retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
        def fetch_cluster_info(cluster_id: str) -> dict:
            # This will retry up to 3 times on any exception
            return api.get_cluster(cluster_id)
        ```
    """

    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
            attempt = 0
            current_delay = delay

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    func_name = getattr(func, "__name__", repr(func))
                    if attempt >= max_attempts:
                        logger.error(
                            f"{func_name} failed after {max_attempts} attempts: {e}",
                            exc_info=True,
                        )
                        raise
                    logger.warning(
                        f"{func_name} attempt {attempt} failed, "
                        f"retrying in {current_delay:.1f}s: {e}"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            return None

        return wrapper

    return decorator


def retry_with_timeout(
    func: Callable[..., T],
    timeout: float,
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Optional[T]:
    """
    Retry a function with a timeout.

    Args:
        func: Function to execute
        timeout: Maximum total time to spend retrying (seconds)
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Result from function or None if all attempts fail.

    Example:
        ```python
        result = retry_with_timeout(
            lambda: expensive_operation(),
            timeout=60.0,
            max_attempts=5
        )
        ```
    """
    start_time = time.time()
    attempt = 0

    while attempt < max_attempts:
        elapsed = time.time() - start_time
        func_name = getattr(func, "__name__", repr(func))
        if elapsed >= timeout:
            logger.error(f"{func_name} timed out after {elapsed:.1f}s")
            return None

        try:
            return func()
        except exceptions as e:
            attempt += 1
            if attempt >= max_attempts:
                logger.error(
                    f"{func_name} failed after {max_attempts} attempts: {e}",
                    exc_info=True,
                )
                return None

            remaining_time = timeout - elapsed
            if remaining_time <= 0:
                logger.error(f"{func_name} timed out")
                return None

            sleep_time = min(delay, remaining_time)
            logger.warning(
                f"{func_name} attempt {attempt} failed, "
                f"retrying in {sleep_time:.1f}s: {e}"
            )
            time.sleep(sleep_time)
            delay *= 2  # Exponential backoff

    return None
