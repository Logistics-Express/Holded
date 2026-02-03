"""
Audit middleware for logging API requests to PostgreSQL.
"""

import asyncio
import logging
import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..db.database import get_db_manager
from ..db.models import ApiAuditLog

logger = logging.getLogger("holded-api.audit")


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware that logs all API requests to the audit table."""

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Skip audit for health checks and non-API routes
        path = request.url.path
        if not self.enabled or path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)

        # Initialize request state for cache tracking
        request.state.cache_hit = False

        # Record start time
        start_time = time.perf_counter()

        # Process request
        error_message = None
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            error_message = str(e)
            status_code = 500
            raise
        finally:
            # Calculate response time
            response_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Get client IP
            client_ip = request.client.host if request.client else None

            # Get query params as dict
            query_params = dict(request.query_params) or None

            # Log asynchronously (fire and forget)
            asyncio.create_task(
                self._log_request(
                    method=request.method,
                    path=path,
                    query_params=query_params,
                    status_code=status_code,
                    response_time_ms=response_time_ms,
                    cache_hit=getattr(request.state, "cache_hit", False),
                    client_ip=client_ip,
                    error_message=error_message,
                )
            )

        return response

    async def _log_request(
        self,
        method: str,
        path: str,
        query_params: dict,
        status_code: int,
        response_time_ms: int,
        cache_hit: bool,
        client_ip: str,
        error_message: str,
    ) -> None:
        """Write audit log entry to database."""
        try:
            db = get_db_manager()
            if not db:
                return

            async with db.get_session() as session:
                log_entry = ApiAuditLog(
                    method=method,
                    path=path,
                    query_params=query_params,
                    status_code=status_code,
                    response_time_ms=response_time_ms,
                    cache_hit=cache_hit,
                    client_ip=client_ip,
                    error_message=error_message,
                )
                session.add(log_entry)
                await session.commit()

            # Log cache hits for debugging
            if cache_hit:
                logger.debug(f"CACHE HIT: {method} {path} ({response_time_ms}ms)")

        except Exception as e:
            # Don't let audit failures break the API
            logger.warning(f"Failed to write audit log: {e}")
