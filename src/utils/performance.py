"""
Performance optimization utilities for CUDA Healthcheck Tool.

Provides caching, memoization, and performance monitoring helpers.
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from collections import OrderedDict
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation.

    Provides efficient caching with automatic eviction of least recently used items.
    """

    def __init__(self, max_size: int = 128) -> None:
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"Cache HIT for key: {key}")
            return self.cache[key]
        self.misses += 1
        logger.debug(f"Cache MISS for key: {key}")
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                evicted_key = next(iter(self.cache))
                del self.cache[evicted_key]
                logger.debug(f"Cache EVICT: {evicted_key}")
        self.cache[key] = value
        logger.debug(f"Cache PUT: {key}")

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.debug("Cache CLEAR")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
        }


# Global cache instance for detector results
_detector_cache = LRUCache(max_size=64)


def cached(
    cache_key_func: Optional[Callable[..., str]] = None, ttl: Optional[int] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to cache function results.

    Args:
        cache_key_func: Function to generate cache key from arguments
        ttl: Time-to-live in seconds (None = infinite)

    Returns:
        Decorated function with caching

    Example:
        ```python
        @cached(cache_key_func=lambda cluster_id: f"cluster_{cluster_id}", ttl=300)
        def get_cluster_info(cluster_id: str) -> dict:
            # Expensive operation
            return fetch_from_api(cluster_id)
        ```
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[str, tuple[Any, float]] = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default: use function name and string of args
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]

                # Check TTL
                if ttl is None or (time.time() - timestamp) < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cast(T, result)
                else:
                    logger.debug(f"Cache expired for {func.__name__}")
                    del cache[cache_key]

            # Call function
            logger.debug(f"Cache miss for {func.__name__}, executing function")
            result = func(*args, **kwargs)

            # Store in cache
            cache[cache_key] = (result, time.time())

            return result

        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()  # type: ignore
        wrapper.cache_info = lambda: {"size": len(cache), "ttl": ttl}  # type: ignore

        return wrapper

    return decorator


def memoize(func: Callable[..., T]) -> Callable[..., T]:
    """
    Simple memoization decorator for pure functions.

    Caches results based on function arguments.
    Best for expensive pure functions with hashable arguments.

    Example:
        ```python
        @memoize
        def fibonacci(n: int) -> int:
            if n < 2:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        ```
    """
    cache: Dict[str, T] = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Create hashable cache key
        try:
            cache_key = str(args) + str(sorted(kwargs.items()))
        except TypeError:
            # Arguments not hashable, don't cache
            logger.warning(f"{func.__name__} has unhashable arguments, skipping cache")
            return func(*args, **kwargs)

        if cache_key not in cache:
            cache[cache_key] = func(*args, **kwargs)
            logger.debug(f"Memoized {func.__name__} with key {cache_key[:50]}...")

        return cache[cache_key]

    return wrapper


class PerformanceTimer:
    """
    Context manager for timing code blocks.

    Example:
        ```python
        with PerformanceTimer("CUDA detection"):
            environment = detector.detect_environment()
        # Logs: "CUDA detection completed in 1.23s"
        ```
    """

    def __init__(self, operation_name: str, log_level: str = "INFO") -> None:
        """
        Initialize performance timer.

        Args:
            operation_name: Name of operation being timed
            log_level: Log level for timing message
        """
        self.operation_name = operation_name
        self.log_level = log_level.upper()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "PerformanceTimer":
        """Start timer."""
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Any
    ) -> None:
        """Stop timer and log result."""
        self.end_time = time.time()
        if self.start_time:
            self.elapsed = self.end_time - self.start_time

            log_func = getattr(logger, self.log_level.lower(), logger.info)
            log_func(f"{self.operation_name} completed in {self.elapsed:.2f}s")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to time function execution.

    Example:
        ```python
        @timed
        def expensive_operation():
            # ... long running code ...
            pass
        ```
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        with PerformanceTimer(func.__name__):
            return func(*args, **kwargs)

    return wrapper


class BatchProcessor:
    """
    Process items in batches for better performance.

    Useful for API calls, database operations, etc.
    """

    def __init__(self, batch_size: int = 100) -> None:
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
        """
        self.batch_size = batch_size

    def process(
        self, items: list[Any], process_func: Callable[[list[Any]], list[Any]]
    ) -> list[Any]:
        """
        Process items in batches.

        Args:
            items: List of items to process
            process_func: Function to process each batch

        Returns:
            List of processed results
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size

        logger.info(f"Processing {len(items)} items in {total_batches} batches")

        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.debug(f"Processing batch {batch_num}/{total_batches}")
            batch_results = process_func(batch)
            results.extend(batch_results)

        logger.info(f"Completed processing {len(results)} results")
        return results


__all__ = [
    "LRUCache",
    "cached",
    "memoize",
    "PerformanceTimer",
    "timed",
    "BatchProcessor",
]
