"""
SQLAlchemy models for audit trail and Contable AI Agent.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    Text,
    DateTime,
    Date,
    Index,
    Numeric,
    ForeignKey,
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


# ============================================================================
# CONTABLE AI AGENT MODELS
# ============================================================================

class ContableActionQueue(Base):
    """Action queue for Contable AI Agent (draft-approve-execute pattern).

    Stores pending actions that need human approval before execution.
    Low-risk actions (<€500 and >95% confidence) are auto-executed.
    """

    __tablename__ = "contable_action_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Action details
    action_type = Column(String(50), nullable=False)  # reconcile, mark_paid, escalate, classify
    action_data = Column(JSONB, nullable=False)  # Full action parameters
    intent = Column(String(100), nullable=True)  # Original intent from caller

    # Risk assessment
    risk_level = Column(String(20), nullable=False)  # auto_execute, draft, reject
    confidence_score = Column(Numeric(3, 2), nullable=True)  # 0.00 to 1.00
    amount_eur = Column(Numeric(12, 2), nullable=True)  # Amount involved (for risk)

    # AI reasoning
    ai_explanation = Column(Text, nullable=True)  # OpenAI explanation
    ai_model = Column(String(50), nullable=True)  # Model used (gpt-4o, etc.)

    # Status tracking
    status = Column(String(20), default="pending", nullable=False)
    # pending, approved, rejected, executed, failed

    # Source tracking
    source_service = Column(String(50), nullable=True)  # voice, ticket-agent, scheduler
    source_request_id = Column(String(100), nullable=True)  # Trace ID

    # Approval tracking
    reviewed_by = Column(String(100), nullable=True)  # Who approved/rejected
    reviewed_at = Column(DateTime, nullable=True)
    review_notes = Column(Text, nullable=True)

    # Execution tracking
    executed_at = Column(DateTime, nullable=True)
    execution_result = Column(JSONB, nullable=True)  # Result of execution
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_action_status", "status", postgresql_using="btree"),
        Index("idx_action_type", "action_type", postgresql_using="btree"),
        Index("idx_action_created", "created_at", postgresql_using="btree"),
        Index("idx_action_source", "source_service", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        return f"<ContableActionQueue {self.action_type} {self.status}>"


class ContableReconciliationMatch(Base):
    """Payment-document reconciliation matches.

    Stores potential matches between Qonto payments and Holded documents
    for the reconciliation workflow.
    """

    __tablename__ = "contable_reconciliation_matches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Qonto payment details
    qonto_transaction_id = Column(String(100), nullable=False)
    qonto_amount = Column(Numeric(12, 2), nullable=False)
    qonto_date = Column(DateTime, nullable=True)
    qonto_reference = Column(String(500), nullable=True)
    qonto_counterparty = Column(String(255), nullable=True)

    # Holded document details
    holded_document_id = Column(String(50), nullable=False)
    holded_document_type = Column(String(20), nullable=False)  # invoice, estimate, purchase
    holded_document_number = Column(String(50), nullable=True)
    holded_amount = Column(Numeric(12, 2), nullable=True)
    holded_contact_name = Column(String(255), nullable=True)

    # Match quality
    match_confidence = Column(Numeric(3, 2), nullable=False)  # 0.00 to 1.00
    match_type = Column(String(50), nullable=False)  # exact_amount, reference, name_similarity
    match_details = Column(JSONB, nullable=True)  # How match was determined

    # Status
    status = Column(String(20), default="pending", nullable=False)
    # pending, approved, rejected, executed

    # Link to action queue
    action_queue_id = Column(UUID(as_uuid=True), ForeignKey("contable_action_queue.id"), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reconciled_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_match_status", "status", postgresql_using="btree"),
        Index("idx_match_qonto", "qonto_transaction_id", postgresql_using="btree"),
        Index("idx_match_holded", "holded_document_id", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        return f"<ContableReconciliationMatch {self.qonto_transaction_id} -> {self.holded_document_id}>"


class StripeReconciliationMatch(Base):
    """Stripe payment to Holded document reconciliation.

    Tracks Stripe payments and their reconciliation with Holded documents.
    Supports partial payments (anticipos) and recurring subscriptions.
    """

    __tablename__ = "stripe_reconciliation_matches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Stripe payment details
    stripe_payment_intent_id = Column(String(100), unique=True, nullable=False)
    stripe_checkout_session_id = Column(String(100), nullable=True)
    stripe_amount_cents = Column(Integer, nullable=False)
    stripe_currency = Column(String(3), default="EUR", nullable=False)
    stripe_customer_email = Column(String(255), nullable=True)
    stripe_payment_method = Column(String(50), nullable=True)  # card, bank_transfer, etc.
    stripe_paid_at = Column(DateTime, nullable=True)

    # Partial payment tracking
    total_document_amount = Column(Numeric(12, 2), nullable=True)  # Full document amount
    paid_amount = Column(Numeric(12, 2), default=0, nullable=False)  # Amount paid so far
    remaining_amount = Column(Numeric(12, 2), nullable=True)  # Outstanding balance
    payment_count = Column(Integer, default=1, nullable=False)  # Number of payments made

    # Holded document link
    holded_document_id = Column(String(50), nullable=True)
    holded_document_type = Column(String(20), nullable=True)  # invoice, estimate, proform
    holded_document_number = Column(String(50), nullable=True)
    holded_contact_id = Column(String(50), nullable=True)
    holded_contact_name = Column(String(255), nullable=True)

    # Reconciliation status
    reconciliation_status = Column(String(20), default="pending", nullable=False)
    # pending, matched, partially_paid, fully_paid, refunded, failed

    # Payment source metadata
    payment_type = Column(String(30), nullable=True)  # presupuesto, invoice, deposit, custom
    presupuesto_id = Column(String(50), nullable=True)  # Link to app-logistics-express
    deposit_percent = Column(Integer, nullable=True)  # For deposit payments

    # Subscription tracking
    stripe_subscription_id = Column(String(100), nullable=True)
    is_recurring = Column(Boolean, default=False, nullable=False)
    subscription_period_start = Column(DateTime, nullable=True)
    subscription_period_end = Column(DateTime, nullable=True)

    # Processing metadata
    source_webhook_event_id = Column(String(100), nullable=True)
    processed_by = Column(String(50), nullable=True)  # webhook, manual, contable-agent
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    reconciled_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_stripe_pi", "stripe_payment_intent_id", postgresql_using="btree"),
        Index("idx_stripe_status", "reconciliation_status", postgresql_using="btree"),
        Index("idx_stripe_holded_doc", "holded_document_id", postgresql_using="btree"),
        Index("idx_stripe_created", "created_at", postgresql_using="btree"),
        Index("idx_stripe_subscription", "stripe_subscription_id", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        return f"<StripeReconciliationMatch {self.stripe_payment_intent_id} -> {self.holded_document_id}>"


class ContableDailySnapshot(Base):
    """Daily financial snapshots for trend analysis.

    Stores end-of-day financial position to track trends over time.
    """

    __tablename__ = "contable_daily_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(Date, unique=True, nullable=False)

    # Receivables (what customers owe us)
    total_receivables = Column(Numeric(14, 2), default=0)
    receivables_current = Column(Numeric(14, 2), default=0)  # Not overdue
    receivables_overdue_30 = Column(Numeric(14, 2), default=0)  # 1-30 days
    receivables_overdue_60 = Column(Numeric(14, 2), default=0)  # 31-60 days
    receivables_overdue_90 = Column(Numeric(14, 2), default=0)  # 61-90 days
    receivables_overdue_90_plus = Column(Numeric(14, 2), default=0)  # 90+ days

    # Payables (what we owe suppliers)
    total_payables = Column(Numeric(14, 2), default=0)
    payables_current = Column(Numeric(14, 2), default=0)
    payables_overdue = Column(Numeric(14, 2), default=0)

    # Cash position
    bank_balance = Column(Numeric(14, 2), nullable=True)  # From Qonto if available

    # Agent activity
    actions_auto_executed = Column(Integer, default=0)
    actions_drafted = Column(Integer, default=0)
    actions_approved = Column(Integer, default=0)
    actions_rejected = Column(Integer, default=0)
    reconciliations_completed = Column(Integer, default=0)

    # Anomalies
    anomalies_detected = Column(Integer, default=0)
    anomalies_resolved = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_snapshot_date", "snapshot_date", postgresql_using="btree"),
    )

    def __repr__(self) -> str:
        return f"<ContableDailySnapshot {self.snapshot_date}>"
