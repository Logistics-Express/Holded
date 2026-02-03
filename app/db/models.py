"""
SQLAlchemy models for audit trail.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    Text,
    DateTime,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ApiAuditLog(Base):
    """Audit log for API requests."""

    __tablename__ = "api_audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    method = Column(String(10), nullable=False)
    path = Column(String(500), nullable=False)
    query_params = Column(JSONB, nullable=True)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Integer, nullable=False)
    cache_hit = Column(Boolean, default=False)
    client_ip = Column(String(45), nullable=True)
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_audit_ts", "timestamp", postgresql_using="btree"),
        Index("idx_audit_path", "path", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        return f"<ApiAuditLog {self.method} {self.path} {self.status_code}>"
