"""
Configuration settings for Holded API service.
Uses pydantic-settings for environment variable management.
"""

from functools import lru_cache
from typing import Dict
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

    # =========================================================================
    # CONTABLE AI AGENT CONFIGURATION
    # =========================================================================

    # OpenAI API
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"  # Model for classification and reasoning

    # Contable Agent Feature Flag
    contable_agent_enabled: bool = True

    # Service-to-Service API Keys (for internal services)
    voice_service_api_key: str = ""
    ticket_service_api_key: str = ""
    secretary_service_api_key: str = ""

    # Risk Assessment Thresholds
    auto_execute_max_amount: float = 500.0  # EUR - max amount for auto-execute
    auto_execute_min_confidence: float = 0.95  # Min confidence for auto-execute
    draft_min_confidence: float = 0.70  # Min confidence for draft (below = reject)
    new_supplier_always_draft: bool = True  # Always draft for new suppliers

    # Rate Limits (requests per minute per service)
    rate_limit_voice: int = 60
    rate_limit_ticket: int = 120
    rate_limit_secretary: int = 30
    rate_limit_default: int = 30

    # Background Jobs
    scheduler_enabled: bool = True
    reconciliation_job_hour: int = 8  # Hour to run daily reconciliation (Madrid time)
    snapshot_job_hour: int = 23  # Hour to run daily snapshot
    anomaly_check_job_hour: int = 9  # Hour to run anomaly detection

    # API Core (Unified Backend)
    api_core_url: str = "https://api-core-production-0f71.up.railway.app"
    api_core_api_key: str = ""  # Service-to-service auth
    api_core_enabled: bool = True  # Publish events to api-core

    # Qonto API (for reconciliation)
    # Company: LOGISTICS EXPRESS ADUANAS, S.L.U.
    qonto_login: str = "logistics-express-aduanas-sociedad-limitada-3984"
    qonto_secret_key: str = ""  # Set via env: QONTO_SECRET_KEY
    qonto_iban: str = "ES2868880001604129394523"  # Default IBAN

    # Additional Qonto accounts (for multi-company support)
    qonto_bm_login: str = "blue-mountain-asesores-sl-7895"
    qonto_bm_secret_key: str = ""
    qonto_le_malaga_login: str = "le-malaga-servicios-logisticos-sl-8900"
    qonto_le_malaga_secret_key: str = ""

    # Holded Treasury ID (for marking payments)
    holded_treasury_id: str = "66eaed086319523a3c03d235"

    # =========================================================================
    # STRIPE CONFIGURATION
    # =========================================================================

    # Stripe API Keys
    stripe_secret_key: str = ""  # Set via env: STRIPE_SECRET_KEY
    stripe_webhook_secret: str = ""  # Set via env: STRIPE_WEBHOOK_SECRET

    # Stripe Treasury ID in Holded (for recording Stripe payments)
    holded_stripe_treasury_id: str = ""  # Set via env: HOLDED_STRIPE_TREASURY_ID

    # Stripe Reconciliation Settings
    stripe_reconciliation_enabled: bool = True  # Enable Stripe reconciliation endpoints
    stripe_auto_reconcile_threshold: float = 0.95  # Confidence threshold for auto-reconciliation

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def service_api_keys(self) -> Dict[str, str]:
        """Get mapping of service names to API keys."""
        return {
            "voice": self.voice_service_api_key,
            "ticket-agent": self.ticket_service_api_key,
            "secretary": self.secretary_service_api_key,
        }

    @property
    def service_rate_limits(self) -> Dict[str, int]:
        """Get mapping of service names to rate limits (per minute)."""
        return {
            "voice": self.rate_limit_voice,
            "ticket-agent": self.rate_limit_ticket,
            "secretary": self.rate_limit_secretary,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
