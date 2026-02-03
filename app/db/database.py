"""
Async database connection management with SQLAlchemy.
"""

import logging
from typing import Optional
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .models import Base

logger = logging.getLogger("holded-api.db")

# Global database manager instance
_db_manager: Optional["DatabaseManager"] = None


class DatabaseManager:
    """Async database connection manager."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None
        self._session_factory = None

    async def connect(self) -> None:
        """Initialize database engine and create tables."""
        self._engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
        )

        # Create tables if they don't exist
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("Database connection established")

    async def close(self) -> None:
        """Close database connection."""
        if self._engine:
            await self._engine.dispose()
        logger.info("Database connection closed")

    def get_session(self) -> AsyncSession:
        """Get a new database session."""
        if not self._session_factory:
            raise RuntimeError("Database not connected")
        return self._session_factory()

    @asynccontextmanager
    async def session(self):
        """Async context manager for database sessions.

        Usage:
            async with db_manager.session() as session:
                result = await session.execute(query)
                await session.commit()
        """
        if not self._session_factory:
            raise RuntimeError("Database not connected")
        session = self._session_factory()
        try:
            yield session
        finally:
            await session.close()

    async def health_check(self) -> dict:
        """Check database health."""
        if not self._engine:
            return {"status": "disconnected"}
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


def get_db_manager() -> Optional[DatabaseManager]:
    """Get the global database manager instance."""
    return _db_manager


def set_db_manager(manager: DatabaseManager) -> None:
    """Set the global database manager instance."""
    global _db_manager
    _db_manager = manager
