"""
Caching decorators for FastAPI endpoints.
"""

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, Optional

from starlette.requests import Request

from .redis_client import get_redis_client

logger = logging.getLogger("holded-api.cache")


def _make_cache_key(prefix: str, args: tuple, kwargs: dict) -> str:
    """Generate a cache key from function arguments."""
    # Filter out non-serializable args (Request, HoldedClient, etc.)
    serializable_args = []
    for arg in args:
        if isinstance(arg, (str, int, float, bool, type(None))):
            serializable_args.append(arg)

    serializable_kwargs = {}
    for k, v in kwargs.items():
        if k in ("request", "client"):
            continue
        if isinstance(v, (str, int, float, bool, type(None), list, dict)):
            serializable_kwargs[k] = v

    # Create hash of arguments
    key_data = json.dumps(
        {"args": serializable_args, "kwargs": serializable_kwargs},
        sort_keys=True,
        default=str,
    )
    key_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]

    return f"holded:{prefix}:{key_hash}"


def cached(
    ttl: int = 300,
    prefix: str = "default",
    key_builder: Optional[Callable[..., str]] = None,
):
    """
    Cache decorator for FastAPI endpoints.

    Args:
        ttl: Cache TTL in seconds (default 5 minutes)
        prefix: Cache key prefix (e.g., "contacts", "documents")
        key_builder: Optional custom key builder function

    Usage:
        @cached(ttl=300, prefix="contacts")
        async def list_contacts(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            redis = get_redis_client()

            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = _make_cache_key(prefix, args, kwargs)

            # Check if caching is enabled and Redis is available
            if redis:
                # Try to get from cache
                cached_value = await redis.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache HIT: {cache_key}")
                    # Mark cache hit for audit middleware
                    for arg in args:
                        if isinstance(arg, Request):
                            arg.state.cache_hit = True
                            break
                    return cached_value

            # Cache miss - call the actual function
            logger.debug(f"Cache MISS: {cache_key}")
            result = await func(*args, **kwargs)

            # Store in cache
            if redis and result is not None:
                await redis.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


async def invalidate_cache(pattern: str) -> int:
    """
    Invalidate cache keys matching pattern.

    Args:
        pattern: Glob pattern (e.g., "holded:contacts:*")

    Returns:
        Number of keys deleted
    """
    redis = get_redis_client()
    if redis:
        return await redis.delete_pattern(pattern)
    return 0
