"""
Configuration settings for Holded API service.
Uses pydantic-settings for environment variable management.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database (PostgreSQL)
    database_url: str = "postgresql+asyncpg://localhost/holded_audit"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Feature flags
    cache_enabled: bool = True
    audit_enabled: bool = True

    # Cache TTLs (seconds)
    cache_ttl_contacts: int = 300       # 5 min
    cache_ttl_contact_single: int = 600 # 10 min
    cache_ttl_documents: int = 900      # 15 min
    cache_ttl_lookup: int = 1800        # 30 min

    # Holded API
    holded_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
