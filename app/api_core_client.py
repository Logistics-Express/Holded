"""
API Core Client for holded-api.

Integrates with api-core's Event Bus for:
- Publishing payment events (payment.received, payment.reconciled)
- Publishing invoice events (invoice.created, invoice.paid)
"""

import logging
from typing import Optional, Dict, Any

import httpx

from app.config import get_settings

logger = logging.getLogger("holded-api")

# Module-level HTTP client for connection reuse
_api_core_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """Get or create the API Core HTTP client."""
    global _api_core_client
    if _api_core_client is None:
        settings = get_settings()
        _api_core_client = httpx.AsyncClient(
            base_url=settings.api_core_url,
            timeout=httpx.Timeout(10.0, connect=3.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            headers={
                "Content-Type": "application/json",
                "X-Service-Name": "holded-api",
            },
        )
    return _api_core_client


# =============================================================================
# EVENT PUBLISHING
# =============================================================================


async def publish_event(
    event_type: str,
    payload: Dict[str, Any],
    subject_type: Optional[str] = None,
    subject_id: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> Optional[str]:
    """
    Publish an event to api-core's event bus.

    Args:
        event_type: Event type (e.g., "payment.reconciled", "invoice.paid")
        payload: Event payload data
        subject_type: Subject type (e.g., "invoice", "payment")
        subject_id: Subject ID (e.g., document ID)
        customer_id: Associated customer ID (UUID string)

    Returns:
        Event ID if successful, None otherwise
    """
    settings = get_settings()
    if not settings.api_core_enabled:
        return None

    client = _get_client()

    try:
        response = await client.post(
            "/api/v1/events/publish",
            json={
                "event_type": event_type,
                "payload": payload,
                "subject_type": subject_type,
                "subject_id": subject_id,
                "customer_id": customer_id,
                "source_service": "holded-api",
            },
        )

        if response.status_code == 200:
            data = response.json()
            event_id = data.get("event_id")
            logger.info(
                f"Event published to api-core: {event_type}, event_id={event_id}"
            )
            return event_id
        else:
            logger.warning(
                f"Failed to publish event to api-core: status={response.status_code}, "
                f"response={response.text[:200]}"
            )
            return None

    except httpx.TimeoutException:
        logger.warning(f"api-core event publish timeout: {event_type}")
        return None
    except Exception as e:
        logger.error(f"api-core event publish error: {e}")
        return None


# =============================================================================
# HELPER FUNCTIONS FOR HOLDED-API
# =============================================================================


async def publish_payment_reconciled_event(
    document_id: str,
    document_number: str,
    document_type: str,
    amount: float,
    qonto_transaction_id: Optional[str] = None,
    contact_name: Optional[str] = None,
    contact_id: Optional[str] = None,
) -> Optional[str]:
    """Publish a payment.reconciled event when a Qonto payment is matched to a Holded document."""
    return await publish_event(
        event_type="payment.reconciled",
        payload={
            "document_id": document_id,
            "document_number": document_number,
            "document_type": document_type,
            "amount": amount,
            "qonto_transaction_id": qonto_transaction_id,
            "contact_name": contact_name,
            "contact_id": contact_id,
            "source": "qonto_reconciliation",
        },
        subject_type="payment",
        subject_id=document_id,
    )


async def publish_invoice_paid_event(
    document_id: str,
    document_type: str,
    amount: float,
    payment_date: Optional[int] = None,
    contact_id: Optional[str] = None,
    contact_name: Optional[str] = None,
) -> Optional[str]:
    """Publish an invoice.paid event when a document is marked as paid."""
    return await publish_event(
        event_type="invoice.paid",
        payload={
            "document_id": document_id,
            "document_type": document_type,
            "amount": amount,
            "payment_date": payment_date,
            "contact_id": contact_id,
            "contact_name": contact_name,
        },
        subject_type="invoice",
        subject_id=document_id,
    )


async def publish_invoice_created_event(
    document_id: str,
    document_number: str,
    document_type: str,
    total: float,
    contact_id: Optional[str] = None,
    contact_name: Optional[str] = None,
) -> Optional[str]:
    """Publish an invoice.created event when a new document is created."""
    return await publish_event(
        event_type="invoice.created",
        payload={
            "document_id": document_id,
            "document_number": document_number,
            "document_type": document_type,
            "total": total,
            "contact_id": contact_id,
            "contact_name": contact_name,
        },
        subject_type="invoice",
        subject_id=document_id,
    )


async def publish_estimate_accepted_event(
    document_id: str,
    document_number: str,
    total: float,
    contact_id: Optional[str] = None,
    contact_name: Optional[str] = None,
    converted_to_invoice_id: Optional[str] = None,
) -> Optional[str]:
    """Publish an estimate.accepted event when an estimate is converted to invoice."""
    return await publish_event(
        event_type="estimate.accepted",
        payload={
            "document_id": document_id,
            "document_number": document_number,
            "total": total,
            "contact_id": contact_id,
            "contact_name": contact_name,
            "converted_to_invoice_id": converted_to_invoice_id,
        },
        subject_type="estimate",
        subject_id=document_id,
    )


async def close_client():
    """Close the HTTP client (call on shutdown)."""
    global _api_core_client
    if _api_core_client:
        await _api_core_client.aclose()
        _api_core_client = None
