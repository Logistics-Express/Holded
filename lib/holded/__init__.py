"""Holded API client library.

Usage:
    from lib.holded import HoldedClient, DocumentBuilder, LineItem

    client = HoldedClient.from_credentials()
    contacts = client.list_contacts()

    # Build document with margin tracking
    builder = DocumentBuilder(contact_id="abc123", doc_type="invoice")
    builder.add_title("Servicios")
    builder.add_item("Consultoría", units=8, price=75, cost_price=45)
    builder.add_suplido("Desplazamiento", amount=150)

    result = client.create_document_with_builder(builder)
"""

from .holded_client import (
    HoldedClient,
    load_credentials,
    DocumentBuilder,
    LineItem,
    ItemKind,
    TaxRate,
    TemplateManager,
    DEFAULT_PAYMENT_METHOD_ID,
    DEFAULT_BANK_INFO,
    TEMPLATES_FILE,
)

__all__ = [
    "HoldedClient",
    "load_credentials",
    "DocumentBuilder",
    "LineItem",
    "ItemKind",
    "TaxRate",
    "TemplateManager",
    "DEFAULT_PAYMENT_METHOD_ID",
    "DEFAULT_BANK_INFO",
    "TEMPLATES_FILE",
]
