"""
Redis client with connection pooling and async support.
"""

import json
import logging
from typing import Any, Optional

import redis.asyncio as redis

logger = logging.getLogger("holded-api.cache")

# Global client instance
_redis_client: Optional["RedisClient"] = None


class RedisClient:
    """Async Redis client wrapper with JSON serialization."""

    def __init__(self, url: str):
        self.url = url
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        self._pool = redis.ConnectionPool.from_url(
            self.url,
            decode_responses=True,
            max_connections=10,
        )
        self._client = redis.Redis(connection_pool=self._pool)
        # Test connection
        await self._client.ping()
        logger.info("Redis connection established")

    async def close(self) -> None:
        """Close connection pool."""
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Redis connection closed")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache, deserializing JSON."""
        if not self._client:
            return None
        try:
            value = await self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 300,
    ) -> bool:
        """Set value in cache with TTL, serializing to JSON."""
        if not self._client:
            return False
        try:
            await self._client.set(key, json.dumps(value), ex=ttl)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a single key."""
        if not self._client:
            return False
        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern. Returns count deleted."""
        if not self._client:
            return 0
        try:
            count = 0
            async for key in self._client.scan_iter(match=pattern):
                await self._client.delete(key)
                count += 1
            if count > 0:
                logger.info(f"Invalidated {count} cache keys matching {pattern}")
            return count
        except Exception as e:
            logger.warning(f"Cache delete_pattern error for {pattern}: {e}")
            return 0

    async def health_check(self) -> dict:
        """Check Redis health and return stats."""
        if not self._client:
            return {"status": "disconnected"}
        try:
            await self._client.ping()
            info = await self._client.info("memory")
            return {
                "status": "healthy",
                "used_memory": info.get("used_memory_human", "unknown"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


def get_redis_client() -> Optional[RedisClient]:
    """Get the global Redis client instance."""
    return _redis_client


def set_redis_client(client: RedisClient) -> None:
    """Set the global Redis client instance."""
    global _redis_client
    _redis_client = client
