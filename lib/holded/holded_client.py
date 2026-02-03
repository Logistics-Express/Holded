"""Holded API client with full CRUD for all modules.

Holded is a Spanish ERP/CRM system with modules for:
- Invoicing: contacts, products, documents, payments, warehouses, services, treasuries
- CRM: leads, funnels, events, bookings
- Projects: projects, tasks, time tracking
- Team: employees, time entries
- Accounting: daily ledger, chart of accounts

API Reference: https://developers.holded.com/reference
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import requests

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30.0
REQUEST_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
CREDENTIALS_FILE = Path.home() / ".holded_credentials.json"
TEMPLATES_FILE = Path.home() / ".holded_templates.json"


def load_credentials() -> dict:
    """Load API key from credentials file."""
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(f"Credentials file not found: {CREDENTIALS_FILE}")
    return json.loads(CREDENTIALS_FILE.read_text())


# ==================== DOCUMENT BUILDER CLASSES ====================

class ItemKind(str, Enum):
    """Types of line items in Holded documents."""
    LINE = "line"
    TITLE = "title"
    SUBTITLE = "subtitle"


class TaxRate(int, Enum):
    """Common Spanish tax rates."""
    IVA_21 = 21
    IVA_10 = 10
    IVA_4 = 4
    IVA_0 = 0


@dataclass
class LineItem:
    """Represents a line item in a Holded document with margin tracking.

    Supports products, services, titles, subtitles, and suplidos (disbursements).

    Example:
        item = LineItem(name="Consultoría", units=8, price=75, cost_price=45, tax=21)
        print(f"Margin: {item.margin}€ ({item.margin_percent:.1f}%)")
    """
    name: str
    units: float = 1.0
    price: float = 0.0  # Sale price per unit
    cost_price: float = 0.0  # Cost price for margin calculation
    tax: int = 21  # IVA percentage (21, 10, 4, 0)
    discount: float = 0.0  # Discount percentage
    retention: int = 0  # IRPF retention percentage
    desc: str = ""  # Description
    supplied: bool = False  # Suplido (disbursement) - no margin
    kind: ItemKind = ItemKind.LINE
    product_id: Optional[str] = None
    service_id: Optional[str] = None
    sku: Optional[str] = None

    @property
    def margin(self) -> float:
        """Calculate margin per unit (sale - cost)."""
        if self.supplied or self.kind != ItemKind.LINE:
            return 0.0
        return self.price - self.cost_price

    @property
    def margin_percent(self) -> float:
        """Calculate margin percentage over cost."""
        if self.cost_price == 0 or self.supplied or self.kind != ItemKind.LINE:
            return 0.0
        return (self.margin / self.cost_price) * 100

    @property
    def subtotal(self) -> float:
        """Calculate subtotal before tax (after discount)."""
        if self.kind != ItemKind.LINE:
            return 0.0
        base = self.price * self.units
        if self.discount:
            base -= base * (self.discount / 100)
        return base

    @property
    def total_margin(self) -> float:
        """Calculate total margin for all units."""
        if self.supplied or self.kind != ItemKind.LINE:
            return 0.0
        return self.margin * self.units

    @property
    def total_cost(self) -> float:
        """Calculate total cost for all units."""
        if self.kind != ItemKind.LINE:
            return 0.0
        return self.cost_price * self.units

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to Holded API format."""
        # Title/subtitle only need name and kind
        if self.kind in (ItemKind.TITLE, ItemKind.SUBTITLE):
            return {
                "name": self.name,
                "kind": self.kind.value,
            }

        # Regular line item
        # Note: Holded API uses "subtotal" for unit price when creating,
        # but returns it as "price" when reading
        item: Dict[str, Any] = {
            "name": self.name,
            "units": self.units,
            "subtotal": self.price,  # API expects subtotal = unit price
            "tax": self.tax,
        }

        if self.cost_price:
            item["costPrice"] = self.cost_price
        if self.desc:
            item["desc"] = self.desc
        if self.discount:
            item["discount"] = self.discount
        if self.retention:
            item["retention"] = self.retention
        if self.supplied:
            item["supplied"] = "Yes"
        if self.product_id:
            item["productId"] = self.product_id
        if self.service_id:
            item["serviceId"] = self.service_id
        if self.sku:
            item["sku"] = self.sku

        return item

    @classmethod
    def from_api_dict(cls, data: Dict[str, Any]) -> "LineItem":
        """Create LineItem from Holded API response."""
        kind_str = data.get("kind", "line")
        try:
            kind = ItemKind(kind_str)
        except ValueError:
            kind = ItemKind.LINE

        return cls(
            name=data.get("name", ""),
            units=float(data.get("units", 1)),
            price=float(data.get("price", 0)),
            cost_price=float(data.get("costPrice", 0)),
            tax=int(data.get("tax", 21)),
            discount=float(data.get("discount", 0)),
            retention=int(data.get("retention", 0)),
            desc=data.get("desc", ""),
            supplied=data.get("supplied") == "Yes",
            kind=kind,
            product_id=data.get("productId"),
            service_id=data.get("serviceId"),
            sku=data.get("sku"),
        )


# Default payment method: Transferencia Qonto
DEFAULT_PAYMENT_METHOD_ID = "66eaee8ca32a44b6f40ec784"

# Bank account info for notes
DEFAULT_BANK_INFO = "ES28 6888 0001 6041 2939 4523 (Logistics Express Aduanas SL)"


@dataclass
class DocumentBuilder:
    """Builder for creating Holded documents with sections and margin tracking.

    Provides a fluent interface for building invoices, estimates, and proformas
    with automatic margin calculation.

    Example:
        builder = DocumentBuilder(contact_id="abc123", doc_type="invoice")
        builder.add_title("Servicios de consultoría")
        builder.add_item("Análisis inicial", units=8, price=75, cost_price=45)
        builder.add_title("Suplidos")
        builder.add_suplido("Desplazamiento", amount=150)

        print(f"Total margin: {builder.total_margin}€ ({builder.margin_percent:.1f}%)")
        doc_data = builder.to_api_dict()

    For complete documents with addresses and payment info:
        builder = DocumentBuilder(contact_id="abc123", doc_type="estimate")
        builder.set_addresses(
            pickup="C/ Origen 123, Málaga",
            delivery="C/ Destino 456, Melilla"
        )
        builder.set_cargo("5 europallets, 450kg, producto alimentario")
        builder.add_item("Expedición Grupaje", units=5, price=105, desc="Málaga-Melilla")
    """
    contact_id: str
    doc_type: str = "invoice"  # invoice, estimate, proform, etc.
    date: Optional[int] = None  # Unix timestamp
    due_date: Optional[int] = None
    notes: str = ""
    language: str = "es"
    currency: str = "EUR"
    items: List[LineItem] = field(default_factory=list)
    num_serie_id: Optional[str] = None  # Numbering series

    # Payment and approval
    payment_method_id: Optional[str] = None  # Default: Transferencia Qonto
    approve_doc: bool = False  # If True, generates document number

    # Address and cargo info (built into notes)
    pickup_address: str = ""
    delivery_address: str = ""
    cargo_description: str = ""
    payment_terms: str = ""
    include_bank_info: bool = True

    def add_title(self, title: str) -> "DocumentBuilder":
        """Add a section title."""
        self.items.append(LineItem(name=title, kind=ItemKind.TITLE))
        return self

    def add_subtitle(self, subtitle: str) -> "DocumentBuilder":
        """Add a section subtitle."""
        self.items.append(LineItem(name=subtitle, kind=ItemKind.SUBTITLE))
        return self

    def add_item(
        self,
        name: str,
        units: float = 1.0,
        price: float = 0.0,
        cost_price: float = 0.0,
        tax: int = 21,
        discount: float = 0.0,
        retention: int = 0,
        desc: str = "",
        supplied: bool = False,
        product_id: Optional[str] = None,
        service_id: Optional[str] = None,
        sku: Optional[str] = None,
    ) -> "DocumentBuilder":
        """Add a line item with optional margin tracking."""
        item = LineItem(
            name=name,
            units=units,
            price=price,
            cost_price=cost_price,
            tax=tax,
            discount=discount,
            retention=retention,
            desc=desc,
            supplied=supplied,
            product_id=product_id,
            service_id=service_id,
            sku=sku,
        )
        self.items.append(item)
        return self

    def add_suplido(
        self,
        name: str,
        amount: float,
        desc: str = "",
        tax: int = 0,
    ) -> "DocumentBuilder":
        """Add a suplido (disbursement) - passed through without margin.

        Suplidos are costs paid on behalf of the client that are reimbursed
        at cost without any markup.
        """
        return self.add_item(
            name=name,
            units=1,
            price=amount,
            cost_price=amount,
            tax=tax,
            desc=desc,
            supplied=True,
        )

    @property
    def line_items(self) -> List[LineItem]:
        """Get only actual line items (excluding titles/subtitles)."""
        return [item for item in self.items if item.kind == ItemKind.LINE]

    @property
    def total_margin(self) -> float:
        """Calculate total margin across all items."""
        return sum(item.total_margin for item in self.line_items)

    @property
    def total_cost(self) -> float:
        """Calculate total cost."""
        return sum(item.total_cost for item in self.line_items)

    @property
    def total_revenue(self) -> float:
        """Calculate total revenue (before tax)."""
        return sum(item.subtotal for item in self.line_items)

    @property
    def margin_percent(self) -> float:
        """Calculate overall margin percentage."""
        if self.total_cost == 0:
            return 0.0
        return (self.total_margin / self.total_cost) * 100

    def set_addresses(self, pickup: str = "", delivery: str = "") -> "DocumentBuilder":
        """Set pickup and delivery addresses."""
        self.pickup_address = pickup
        self.delivery_address = delivery
        return self

    def set_cargo(self, description: str) -> "DocumentBuilder":
        """Set cargo description."""
        self.cargo_description = description
        return self

    def set_payment_terms(self, terms: str) -> "DocumentBuilder":
        """Set payment terms text."""
        self.payment_terms = terms
        return self

    def build_notes(self) -> str:
        """Build notes from addresses, cargo, and payment info."""
        lines = []

        if self.pickup_address:
            lines.append(f"• RECOGIDA: {self.pickup_address}")
        if self.delivery_address:
            lines.append(f"• ENTREGA: {self.delivery_address}")
        if self.cargo_description:
            lines.append(f"• MERCANCÍA: {self.cargo_description}")
        if self.payment_terms:
            lines.append(f"• PAGO: {self.payment_terms}")
        if self.include_bank_info:
            lines.append(f"• IBAN: {DEFAULT_BANK_INFO}")

        # Append any custom notes
        if self.notes:
            if lines:
                lines.append("")  # Empty line separator
            lines.append(self.notes)

        return "\n".join(lines)

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to Holded API format for document creation."""
        doc: Dict[str, Any] = {
            "contactId": self.contact_id,
            "date": self.date or int(time.time()),
            "items": [item.to_api_dict() for item in self.items],
            "language": self.language,
            "currency": self.currency,
        }

        if self.due_date:
            doc["dueDate"] = self.due_date

        # Build notes from addresses/cargo or use custom notes
        built_notes = self.build_notes()
        if built_notes:
            doc["notes"] = built_notes

        if self.num_serie_id:
            doc["numSerieId"] = self.num_serie_id

        # Payment method (default to Transferencia Qonto for invoices)
        payment_id = self.payment_method_id or DEFAULT_PAYMENT_METHOD_ID
        if self.doc_type == "invoice":
            doc["paymentMethodId"] = payment_id

        # Approve document to generate number
        if self.approve_doc:
            doc["approveDoc"] = True

        return doc

    @classmethod
    def from_document(cls, doc: Dict[str, Any], doc_type: str = "invoice") -> "DocumentBuilder":
        """Create a DocumentBuilder from an existing Holded document.

        Useful for cloning documents or converting between types.
        """
        # API returns "contact" field, but we also support "contactId" for flexibility
        contact_id = doc.get("contactId") or doc.get("contact", "")

        builder = cls(
            contact_id=contact_id,
            doc_type=doc_type,
            date=doc.get("date"),
            due_date=doc.get("dueDate"),
            notes=doc.get("notes", ""),
            language=doc.get("language", "es"),
            currency=doc.get("currency", "EUR"),
        )

        # Holded uses "products" in API response, but we also support "items"
        for item_data in doc.get("products", doc.get("items", [])):
            builder.items.append(LineItem.from_api_dict(item_data))

        return builder

    def to_template(self) -> Dict[str, Any]:
        """Convert DocumentBuilder to a reusable template.

        Templates store items and default values but NOT contact_id or date,
        allowing reuse for different clients.
        """
        return {
            "items": [item.to_api_dict() for item in self.items],
            "notes": self.notes,
            "language": self.language,
            "currency": self.currency,
            "pickup_address": self.pickup_address,
            "delivery_address": self.delivery_address,
            "cargo_description": self.cargo_description,
            "payment_terms": self.payment_terms,
            "include_bank_info": self.include_bank_info,
        }

    @classmethod
    def from_template(
        cls,
        template: Dict[str, Any],
        contact_id: str,
        doc_type: str = "estimate",
    ) -> "DocumentBuilder":
        """Create DocumentBuilder from a saved template.

        Args:
            template: Template dict with items and defaults
            contact_id: Contact ID for the new document
            doc_type: Document type (default: estimate)

        Returns:
            DocumentBuilder ready for customization
        """
        builder = cls(
            contact_id=contact_id,
            doc_type=doc_type,
            notes=template.get("notes", ""),
            language=template.get("language", "es"),
            currency=template.get("currency", "EUR"),
            pickup_address=template.get("pickup_address", ""),
            delivery_address=template.get("delivery_address", ""),
            cargo_description=template.get("cargo_description", ""),
            payment_terms=template.get("payment_terms", ""),
            include_bank_info=template.get("include_bank_info", True),
        )

        for item_data in template.get("items", []):
            builder.items.append(LineItem.from_api_dict(item_data))

        return builder


@dataclass
class JournalLine:
    """Represents a single line in a journal entry.

    Journal entries use double-entry bookkeeping: debits must equal credits.

    Example:
        # Record a sale with VAT
        JournalLine(account_id="acc123", account_code="4300", debit=1210.00)  # Receivable
        JournalLine(account_id="acc456", account_code="7000", credit=1000.00)  # Revenue
        JournalLine(account_id="acc789", account_code="4750", credit=210.00)   # VAT payable
    """
    account_id: str
    account_code: str = ""  # For display purposes
    account_name: str = ""  # For display purposes
    debit: float = 0.0
    credit: float = 0.0
    description: str = ""

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to Holded API format for daily ledger line."""
        line: Dict[str, Any] = {
            "accountId": self.account_id,
        }
        if self.debit > 0:
            line["debit"] = self.debit
        if self.credit > 0:
            line["credit"] = self.credit
        if self.description:
            line["desc"] = self.description
        return line


@dataclass
class JournalEntryBuilder:
    """Builder for creating balanced journal entries.

    Provides a fluent interface for building double-entry accounting
    journal entries with automatic balance validation.

    Example:
        builder = JournalEntryBuilder(description="Invoice payment received")
        builder.add_debit("bank_id", 1210.00, "4300", "Bank deposit")
        builder.add_credit("receivable_id", 1210.00, "5720", "Customer payment")

        if builder.is_balanced:
            entry = builder.to_api_dict()
            client.create_daily_ledger_entry(entry)
    """
    description: str = ""
    reference: str = ""
    date: Optional[int] = None  # Unix timestamp
    lines: List[JournalLine] = field(default_factory=list)

    def add_debit(
        self,
        account_id: str,
        amount: float,
        account_code: str = "",
        description: str = "",
        account_name: str = "",
    ) -> "JournalEntryBuilder":
        """Add a debit line to the journal entry.

        Args:
            account_id: Holded account ID
            amount: Debit amount (positive)
            account_code: Account code for display (e.g., '4300')
            description: Line description
            account_name: Account name for display

        Returns:
            Self for chaining
        """
        self.lines.append(JournalLine(
            account_id=account_id,
            account_code=account_code,
            account_name=account_name,
            debit=abs(amount),
            credit=0.0,
            description=description,
        ))
        return self

    def add_credit(
        self,
        account_id: str,
        amount: float,
        account_code: str = "",
        description: str = "",
        account_name: str = "",
    ) -> "JournalEntryBuilder":
        """Add a credit line to the journal entry.

        Args:
            account_id: Holded account ID
            amount: Credit amount (positive)
            account_code: Account code for display (e.g., '7000')
            description: Line description
            account_name: Account name for display

        Returns:
            Self for chaining
        """
        self.lines.append(JournalLine(
            account_id=account_id,
            account_code=account_code,
            account_name=account_name,
            debit=0.0,
            credit=abs(amount),
            description=description,
        ))
        return self

    @property
    def total_debits(self) -> float:
        """Calculate total of all debit amounts."""
        return sum(line.debit for line in self.lines)

    @property
    def total_credits(self) -> float:
        """Calculate total of all credit amounts."""
        return sum(line.credit for line in self.lines)

    @property
    def is_balanced(self) -> bool:
        """Check if debits equal credits (within rounding tolerance)."""
        return abs(self.total_debits - self.total_credits) < 0.01

    @property
    def imbalance(self) -> float:
        """Get the imbalance amount (debits - credits)."""
        return self.total_debits - self.total_credits

    def validate(self) -> List[str]:
        """Return list of validation errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not self.lines:
            errors.append("Journal entry must have at least one line")

        if len(self.lines) < 2:
            errors.append("Journal entry must have at least two lines (debit and credit)")

        if not self.is_balanced:
            errors.append(
                f"Journal entry is not balanced: "
                f"debits={self.total_debits:.2f}, credits={self.total_credits:.2f}, "
                f"imbalance={self.imbalance:.2f}"
            )

        for i, line in enumerate(self.lines):
            if not line.account_id:
                errors.append(f"Line {i+1} missing account_id")
            if line.debit == 0 and line.credit == 0:
                errors.append(f"Line {i+1} has no debit or credit amount")
            if line.debit > 0 and line.credit > 0:
                errors.append(f"Line {i+1} has both debit and credit (use separate lines)")

        return errors

    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to Holded API format for daily ledger entry.

        Raises:
            ValueError: If the entry is not valid (use validate() first)

        Returns:
            Dict ready for create_daily_ledger_entry()
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid journal entry: {'; '.join(errors)}")

        entry: Dict[str, Any] = {
            "lines": [line.to_api_dict() for line in self.lines],
        }

        if self.description:
            entry["desc"] = self.description
        if self.reference:
            entry["reference"] = self.reference
        if self.date:
            entry["date"] = self.date
        else:
            entry["date"] = int(time.time())

        return entry

    def preview(self) -> str:
        """Generate a human-readable preview of the journal entry.

        Returns:
            Formatted string showing the entry
        """
        lines_out = []
        lines_out.append(f"Journal Entry: {self.description or '(no description)'}")
        if self.reference:
            lines_out.append(f"Reference: {self.reference}")
        lines_out.append("-" * 60)
        lines_out.append(f"{'Account':<30} {'Debit':>12} {'Credit':>12}")
        lines_out.append("-" * 60)

        for line in self.lines:
            account = line.account_code or line.account_id[:8]
            if line.account_name:
                account = f"{account} {line.account_name}"
            account = account[:28]

            debit_str = f"{line.debit:.2f}" if line.debit > 0 else ""
            credit_str = f"{line.credit:.2f}" if line.credit > 0 else ""
            lines_out.append(f"{account:<30} {debit_str:>12} {credit_str:>12}")

        lines_out.append("-" * 60)
        lines_out.append(f"{'TOTAL':<30} {self.total_debits:>12.2f} {self.total_credits:>12.2f}")

        if not self.is_balanced:
            lines_out.append(f"⚠ UNBALANCED: difference of {self.imbalance:.2f}")

        return "\n".join(lines_out)


class TemplateManager:
    """Manages reusable document templates stored locally.

    Templates are saved to ~/.holded_templates.json and can be used
    to quickly create new documents with predefined items.

    Usage:
        manager = TemplateManager()

        # Save from existing document
        builder = DocumentBuilder.from_document(doc, doc_type="estimate")
        manager.save("grupaje-melilla", builder.to_template())

        # Create from template
        template = manager.get("grupaje-melilla")
        if template:
            builder = DocumentBuilder.from_template(template, contact_id="abc123")
            client.create_document_with_builder(builder)
    """

    def __init__(self, templates_file: Optional[Path] = None):
        """Initialize TemplateManager.

        Args:
            templates_file: Path to templates JSON file. Defaults to ~/.holded_templates.json
        """
        self.templates_file = templates_file or TEMPLATES_FILE
        self._templates: Optional[Dict[str, Dict]] = None

    def _load(self) -> Dict[str, Dict]:
        """Load templates from file."""
        if self._templates is None:
            if self.templates_file.exists():
                self._templates = json.loads(self.templates_file.read_text())
            else:
                self._templates = {}
        return self._templates

    def _save(self) -> None:
        """Save templates to file."""
        if self._templates is not None:
            self.templates_file.write_text(
                json.dumps(self._templates, indent=2, ensure_ascii=False)
            )

    @property
    def templates(self) -> Dict[str, Dict]:
        """Get all templates."""
        return self._load()

    def list(self) -> List[str]:
        """List all template names."""
        return list(self._load().keys())

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            Template dict or None if not found
        """
        return self._load().get(name)

    def save(self, name: str, template: Dict[str, Any]) -> None:
        """Save a template.

        Args:
            name: Template name (will overwrite if exists)
            template: Template data from DocumentBuilder.to_template()
        """
        self._load()
        self._templates[name] = template
        self._save()

    def delete(self, name: str) -> bool:
        """Delete a template.

        Args:
            name: Template name

        Returns:
            True if deleted, False if not found
        """
        templates = self._load()
        if name in templates:
            del templates[name]
            self._save()
            return True
        return False

    def save_from_document(
        self,
        name: str,
        doc: Dict[str, Any],
        doc_type: str = "estimate",
    ) -> None:
        """Save a template directly from a Holded document.

        Args:
            name: Template name
            doc: Holded document response
            doc_type: Document type for reconstruction
        """
        builder = DocumentBuilder.from_document(doc, doc_type)
        self.save(name, builder.to_template())


class HoldedClient:
    """Holded API client with retry logic and pagination support.

    Usage:
        client = HoldedClient.from_credentials()
        contacts = client.list_contacts()
        invoice = client.get_document("abc123")
    """

    BASE_URL = "https://api.holded.com/api"

    # API path mappings for each module
    INVOICING_PATHS = {
        "contacts": "/invoicing/v1/contacts",
        "products": "/invoicing/v1/products",
        "documents": "/invoicing/v1/documents",
        "payments": "/invoicing/v1/payments",
        "warehouses": "/invoicing/v1/warehouses",
        "services": "/invoicing/v1/services",
        "treasuries": "/invoicing/v1/treasury",
        "expensesaccounts": "/invoicing/v1/expensesaccounts",
        "saleschannels": "/invoicing/v1/saleschannels",
        "taxes": "/invoicing/v1/taxes",
        "numberingseries": "/invoicing/v1/numberingseries",
        "contactgroups": "/invoicing/v1/contactgroups",
    }

    CRM_PATHS = {
        "leads": "/crm/v1/leads",
        "funnels": "/crm/v1/funnels",
        "events": "/crm/v1/events",
        "bookings": "/crm/v1/bookings",
    }

    PROJECTS_PATHS = {
        "projects": "/projects/v1/projects",
        "tasks": "/projects/v1/tasks",
        "timetracking": "/projects/v1/timetracking",
    }

    TEAM_PATHS = {
        "employees": "/team/v1/employees",
        "timetracking": "/team/v1/timetracking",
    }

    ACCOUNTING_PATHS = {
        "dailyledger": "/accounting/v1/dailyledger",
        "chartofaccounts": "/accounting/v1/chartofaccounts",
    }

    # Document types supported by Holded
    DOCUMENT_TYPES = [
        "invoice", "salesreceipt", "creditnote", "receiptnote",
        "estimate", "salesorder", "waybill", "proform",
        "purchase", "purchaserefund", "purchaseorder"
    ]

    def __init__(self, api_key: str) -> None:
        """Initialize client with API key.

        Args:
            api_key: Holded API key from Settings > Developers
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    @classmethod
    def from_credentials(cls, credentials_file: Optional[Path] = None) -> "HoldedClient":
        """Create client from credentials file.

        Args:
            credentials_file: Path to JSON file with api_key. Defaults to ~/.holded_credentials.json
        """
        creds_path = credentials_file or CREDENTIALS_FILE
        if not creds_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {creds_path}")

        creds = json.loads(creds_path.read_text())
        api_key = creds.get("api_key")
        if not api_key:
            raise ValueError("No api_key found in credentials file")

        return cls(api_key)

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retries: int = REQUEST_RETRIES,
    ) -> Union[Dict, List, bytes]:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path (e.g., /invoicing/v1/contacts)
            params: Query parameters
            json_data: JSON body for POST/PUT
            retries: Number of retries remaining

        Returns:
            Parsed JSON response or bytes for binary content
        """
        url = f"{self.BASE_URL}{path}"

        for attempt in range(retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    timeout=REQUEST_TIMEOUT,
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                # Handle retryable errors
                if response.status_code in RETRYABLE_STATUS_CODES:
                    wait_time = 2 ** attempt
                    logger.warning(f"Retryable error {response.status_code}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Raise for other errors
                response.raise_for_status()

                # Return bytes for PDF endpoints
                if "pdf" in path.lower() or response.headers.get("Content-Type", "").startswith("application/pdf"):
                    return response.content

                # Parse JSON
                if response.content:
                    return response.json()
                return {}

            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"Request failed: {e}. Retrying...")
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Request failed after {retries} retries")

    def _paginate(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate through paginated results.

        Args:
            path: API path
            params: Additional query parameters
            limit: Maximum items to return (None for all)

        Yields:
            Individual items from paginated response
        """
        params = params or {}
        page = 1
        count = 0

        while True:
            params["page"] = page
            response = self._request("GET", path, params=params)

            # Handle different response formats
            items = response if isinstance(response, list) else response.get("data", response.get("items", []))

            if not items:
                break

            for item in items:
                yield item
                count += 1
                if limit and count >= limit:
                    return

            # Check if more pages exist
            if len(items) < 50:  # Typical page size
                break

            page += 1

    # ==================== INVOICING MODULE ====================

    def list_contacts(self, page: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """List all contacts."""
        if limit:
            return list(self._paginate(self.INVOICING_PATHS["contacts"], limit=limit))
        return self._request("GET", self.INVOICING_PATHS["contacts"], params={"page": page})

    def get_contact(self, contact_id: str) -> Dict:
        """Get a single contact by ID."""
        return self._request("GET", f"{self.INVOICING_PATHS['contacts']}/{contact_id}")

    def create_contact(self, data: Dict[str, Any]) -> Dict:
        """Create a new contact."""
        return self._request("POST", self.INVOICING_PATHS["contacts"], json_data=data)

    def update_contact(self, contact_id: str, data: Dict[str, Any]) -> Dict:
        """Update an existing contact."""
        return self._request("PUT", f"{self.INVOICING_PATHS['contacts']}/{contact_id}", json_data=data)

    def delete_contact(self, contact_id: str) -> bool:
        """Delete a contact."""
        self._request("DELETE", f"{self.INVOICING_PATHS['contacts']}/{contact_id}")
        return True

    def list_products(self, page: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """List all products."""
        if limit:
            return list(self._paginate(self.INVOICING_PATHS["products"], limit=limit))
        return self._request("GET", self.INVOICING_PATHS["products"], params={"page": page})

    def get_product(self, product_id: str) -> Dict:
        """Get a single product by ID."""
        return self._request("GET", f"{self.INVOICING_PATHS['products']}/{product_id}")

    def create_product(self, data: Dict[str, Any]) -> Dict:
        """Create a new product."""
        return self._request("POST", self.INVOICING_PATHS["products"], json_data=data)

    def update_product(self, product_id: str, data: Dict[str, Any]) -> Dict:
        """Update an existing product."""
        return self._request("PUT", f"{self.INVOICING_PATHS['products']}/{product_id}", json_data=data)

    def delete_product(self, product_id: str) -> bool:
        """Delete a product."""
        self._request("DELETE", f"{self.INVOICING_PATHS['products']}/{product_id}")
        return True

    def list_documents(
        self,
        doc_type: Optional[str] = None,
        page: int = 1,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """List documents (invoices, estimates, etc.).

        Args:
            doc_type: Filter by document type (invoice, estimate, etc.)
            page: Page number
            limit: Maximum items to return
        """
        params = {"page": page}

        # Holded API requires document type in path, not query param
        if doc_type:
            path = f"{self.INVOICING_PATHS['documents']}/{doc_type}"
        else:
            path = self.INVOICING_PATHS["documents"]

        if limit:
            return list(self._paginate(path, params=params, limit=limit))
        return self._request("GET", path, params=params)

    def get_document(self, document_id: str, doc_type: Optional[str] = None) -> Dict:
        """Get a single document by ID.

        Args:
            document_id: Document ID
            doc_type: Document type (invoice, estimate, etc.). If not provided,
                      tries common types until one works.
        """
        if doc_type:
            return self._request("GET", f"{self.INVOICING_PATHS['documents']}/{doc_type}/{document_id}")

        # Try common document types if type not specified
        for try_type in ["invoice", "estimate", "proform", "salesorder", "creditnote", "purchase"]:
            try:
                result = self._request("GET", f"{self.INVOICING_PATHS['documents']}/{try_type}/{document_id}")
                if result:
                    result["_docType"] = try_type  # Store detected type
                    return result
            except Exception:
                continue

        raise ValueError(f"Document {document_id} not found in any document type")

    def create_document(self, data: Dict[str, Any], doc_type: str = "invoice") -> Dict:
        """Create a new document (invoice, estimate, etc.).

        Args:
            data: Document data including contactId, items, notes, etc.
            doc_type: Document type (invoice, estimate, proform, etc.)
        """
        return self._request("POST", f"{self.INVOICING_PATHS['documents']}/{doc_type}", json_data=data)

    def update_document(self, document_id: str, data: Dict[str, Any], doc_type: str = "invoice") -> Dict:
        """Update an existing document.

        Args:
            document_id: Document ID
            data: Fields to update
            doc_type: Document type (invoice, estimate, proform, etc.)
        """
        return self._request("PUT", f"{self.INVOICING_PATHS['documents']}/{doc_type}/{document_id}", json_data=data)

    def delete_document(self, document_id: str, doc_type: str = "invoice") -> bool:
        """Delete a document.

        Args:
            document_id: Document ID
            doc_type: Document type (invoice, estimate, proform, etc.)
        """
        self._request("DELETE", f"{self.INVOICING_PATHS['documents']}/{doc_type}/{document_id}")
        return True

    def send_document(self, document_id: str, emails: Optional[List[str]] = None, doc_type: str = "invoice") -> Dict:
        """Send a document by email.

        Args:
            document_id: Document ID
            emails: List of email addresses (optional, uses contact email if not provided)
            doc_type: Document type (invoice, estimate, proform, etc.)
        """
        data = {"emails": emails} if emails else {}
        return self._request("POST", f"{self.INVOICING_PATHS['documents']}/{doc_type}/{document_id}/send", json_data=data)

    def get_document_pdf(self, document_id: str, doc_type: Optional[str] = None) -> bytes:
        """Download document as PDF.

        Args:
            document_id: Document ID
            doc_type: Document type (invoice, estimate, etc.). If not provided,
                      tries common types until one works.
        """
        if doc_type:
            return self._request("GET", f"{self.INVOICING_PATHS['documents']}/{doc_type}/{document_id}/pdf")

        # Try common document types if type not specified
        for try_type in ["invoice", "estimate", "proform", "salesorder", "creditnote", "purchase"]:
            try:
                result = self._request("GET", f"{self.INVOICING_PATHS['documents']}/{try_type}/{document_id}/pdf")
                if result:
                    return result
            except Exception:
                continue

        raise ValueError(f"Document {document_id} not found in any document type")

    def pay_document(self, document_id: str, data: Dict[str, Any], doc_type: str = "invoice") -> Dict:
        """Mark a document as paid.

        Args:
            document_id: Document ID
            data: Payment data with keys:
                - amount: Payment amount
                - date: Unix timestamp
                - bankId: Treasury/bank account ID (optional)
            doc_type: Document type (invoice, salesreceipt, etc.)
        """
        return self._request("POST", f"{self.INVOICING_PATHS['documents']}/{doc_type}/{document_id}/pay", json_data=data)

    def list_payments(self, page: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """List all payments."""
        if limit:
            return list(self._paginate(self.INVOICING_PATHS["payments"], limit=limit))
        return self._request("GET", self.INVOICING_PATHS["payments"], params={"page": page})

    def get_payment(self, payment_id: str) -> Dict:
        """Get a single payment by ID."""
        return self._request("GET", f"{self.INVOICING_PATHS['payments']}/{payment_id}")

    def create_payment(self, data: Dict[str, Any]) -> Dict:
        """Create a new payment."""
        return self._request("POST", self.INVOICING_PATHS["payments"], json_data=data)

    def delete_payment(self, payment_id: str) -> bool:
        """Delete a payment."""
        self._request("DELETE", f"{self.INVOICING_PATHS['payments']}/{payment_id}")
        return True

    def list_warehouses(self, page: int = 1) -> List[Dict]:
        """List all warehouses."""
        return self._request("GET", self.INVOICING_PATHS["warehouses"], params={"page": page})

    def get_warehouse(self, warehouse_id: str) -> Dict:
        """Get a single warehouse by ID."""
        return self._request("GET", f"{self.INVOICING_PATHS['warehouses']}/{warehouse_id}")

    def create_warehouse(self, data: Dict[str, Any]) -> Dict:
        """Create a new warehouse."""
        return self._request("POST", self.INVOICING_PATHS["warehouses"], json_data=data)

    def update_warehouse(self, warehouse_id: str, data: Dict[str, Any]) -> Dict:
        """Update an existing warehouse."""
        return self._request("PUT", f"{self.INVOICING_PATHS['warehouses']}/{warehouse_id}", json_data=data)

    def delete_warehouse(self, warehouse_id: str) -> bool:
        """Delete a warehouse."""
        self._request("DELETE", f"{self.INVOICING_PATHS['warehouses']}/{warehouse_id}")
        return True

    def list_services(self, page: int = 1) -> List[Dict]:
        """List all services."""
        return self._request("GET", self.INVOICING_PATHS["services"], params={"page": page})

    def list_treasuries(self, page: int = 1) -> List[Dict]:
        """List all treasury accounts."""
        return self._request("GET", self.INVOICING_PATHS["treasuries"], params={"page": page})

    # ==================== CONTACT ATTACHMENTS ====================

    def list_contact_attachments(self, contact_id: str) -> List[str]:
        """List attachment filenames for a contact.

        Args:
            contact_id: Holded contact ID

        Returns:
            List of attachment filenames
        """
        result = self._request("GET", f"{self.INVOICING_PATHS['contacts']}/{contact_id}/attachments/list")
        # API returns list of filenames directly
        if isinstance(result, list):
            return result
        return result.get("files", result.get("attachments", []))

    def get_contact_attachment(self, contact_id: str, filename: str) -> bytes:
        """Download a specific attachment from a contact.

        Args:
            contact_id: Holded contact ID
            filename: Name of the attachment file

        Returns:
            Binary content of the attachment
        """
        return self._request(
            "GET",
            f"{self.INVOICING_PATHS['contacts']}/{contact_id}/attachments/get",
            params={"filename": filename}
        )

    # ==================== CRM MODULE ====================

    def list_leads(self, page: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """List all CRM leads."""
        if limit:
            return list(self._paginate(self.CRM_PATHS["leads"], limit=limit))
        return self._request("GET", self.CRM_PATHS["leads"], params={"page": page})

    def get_lead(self, lead_id: str) -> Dict:
        """Get a single lead by ID."""
        return self._request("GET", f"{self.CRM_PATHS['leads']}/{lead_id}")

    def create_lead(self, data: Dict[str, Any]) -> Dict:
        """Create a new lead."""
        return self._request("POST", self.CRM_PATHS["leads"], json_data=data)

    def update_lead(self, lead_id: str, data: Dict[str, Any]) -> Dict:
        """Update an existing lead."""
        return self._request("PUT", f"{self.CRM_PATHS['leads']}/{lead_id}", json_data=data)

    def delete_lead(self, lead_id: str) -> bool:
        """Delete a lead."""
        self._request("DELETE", f"{self.CRM_PATHS['leads']}/{lead_id}")
        return True

    def list_funnels(self, page: int = 1) -> List[Dict]:
        """List all sales funnels."""
        return self._request("GET", self.CRM_PATHS["funnels"], params={"page": page})

    def get_funnel(self, funnel_id: str) -> Dict:
        """Get a single funnel by ID."""
        return self._request("GET", f"{self.CRM_PATHS['funnels']}/{funnel_id}")

    def list_events(self, page: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """List all CRM events."""
        if limit:
            return list(self._paginate(self.CRM_PATHS["events"], limit=limit))
        return self._request("GET", self.CRM_PATHS["events"], params={"page": page})

    def get_event(self, event_id: str) -> Dict:
        """Get a single event by ID."""
        return self._request("GET", f"{self.CRM_PATHS['events']}/{event_id}")

    def create_event(self, data: Dict[str, Any]) -> Dict:
        """Create a new event."""
        return self._request("POST", self.CRM_PATHS["events"], json_data=data)

    def update_event(self, event_id: str, data: Dict[str, Any]) -> Dict:
        """Update an existing event."""
        return self._request("PUT", f"{self.CRM_PATHS['events']}/{event_id}", json_data=data)

    def delete_event(self, event_id: str) -> bool:
        """Delete an event."""
        self._request("DELETE", f"{self.CRM_PATHS['events']}/{event_id}")
        return True

    def list_bookings(self, page: int = 1) -> List[Dict]:
        """List all bookings."""
        return self._request("GET", self.CRM_PATHS["bookings"], params={"page": page})

    def get_booking(self, booking_id: str) -> Dict:
        """Get a single booking by ID."""
        return self._request("GET", f"{self.CRM_PATHS['bookings']}/{booking_id}")

    def create_booking(self, data: Dict[str, Any]) -> Dict:
        """Create a new booking."""
        return self._request("POST", self.CRM_PATHS["bookings"], json_data=data)

    # ==================== PROJECTS MODULE ====================

    def list_projects(self, page: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """List all projects."""
        if limit:
            return list(self._paginate(self.PROJECTS_PATHS["projects"], limit=limit))
        return self._request("GET", self.PROJECTS_PATHS["projects"], params={"page": page})

    def get_project(self, project_id: str) -> Dict:
        """Get a single project by ID."""
        return self._request("GET", f"{self.PROJECTS_PATHS['projects']}/{project_id}")

    def create_project(self, data: Dict[str, Any]) -> Dict:
        """Create a new project."""
        return self._request("POST", self.PROJECTS_PATHS["projects"], json_data=data)

    def update_project(self, project_id: str, data: Dict[str, Any]) -> Dict:
        """Update an existing project."""
        return self._request("PUT", f"{self.PROJECTS_PATHS['projects']}/{project_id}", json_data=data)

    def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        self._request("DELETE", f"{self.PROJECTS_PATHS['projects']}/{project_id}")
        return True

    def list_tasks(self, project_id: Optional[str] = None, page: int = 1) -> List[Dict]:
        """List tasks, optionally filtered by project."""
        params = {"page": page}
        if project_id:
            params["projectId"] = project_id
        return self._request("GET", self.PROJECTS_PATHS["tasks"], params=params)

    def get_task(self, task_id: str) -> Dict:
        """Get a single task by ID."""
        return self._request("GET", f"{self.PROJECTS_PATHS['tasks']}/{task_id}")

    def create_task(self, data: Dict[str, Any]) -> Dict:
        """Create a new task."""
        return self._request("POST", self.PROJECTS_PATHS["tasks"], json_data=data)

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        self._request("DELETE", f"{self.PROJECTS_PATHS['tasks']}/{task_id}")
        return True

    def list_project_time_entries(self, page: int = 1) -> List[Dict]:
        """List all project time tracking entries."""
        return self._request("GET", self.PROJECTS_PATHS["timetracking"], params={"page": page})

    def create_project_time_entry(self, data: Dict[str, Any]) -> Dict:
        """Create a new project time entry."""
        return self._request("POST", self.PROJECTS_PATHS["timetracking"], json_data=data)

    # ==================== TEAM MODULE ====================

    def list_employees(self, page: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """List all employees."""
        if limit:
            return list(self._paginate(self.TEAM_PATHS["employees"], limit=limit))
        return self._request("GET", self.TEAM_PATHS["employees"], params={"page": page})

    def get_employee(self, employee_id: str) -> Dict:
        """Get a single employee by ID."""
        return self._request("GET", f"{self.TEAM_PATHS['employees']}/{employee_id}")

    def create_employee(self, data: Dict[str, Any]) -> Dict:
        """Create a new employee."""
        return self._request("POST", self.TEAM_PATHS["employees"], json_data=data)

    def update_employee(self, employee_id: str, data: Dict[str, Any]) -> Dict:
        """Update an existing employee."""
        return self._request("PUT", f"{self.TEAM_PATHS['employees']}/{employee_id}", json_data=data)

    def delete_employee(self, employee_id: str) -> bool:
        """Delete an employee."""
        self._request("DELETE", f"{self.TEAM_PATHS['employees']}/{employee_id}")
        return True

    def list_team_time_entries(self, page: int = 1) -> List[Dict]:
        """List all team time tracking entries."""
        return self._request("GET", self.TEAM_PATHS["timetracking"], params={"page": page})

    def create_team_time_entry(self, data: Dict[str, Any]) -> Dict:
        """Create a new team time entry (clock in)."""
        return self._request("POST", self.TEAM_PATHS["timetracking"], json_data=data)

    def update_team_time_entry(self, entry_id: str, data: Dict[str, Any]) -> Dict:
        """Update a team time entry (clock out, pause, etc.)."""
        return self._request("PUT", f"{self.TEAM_PATHS['timetracking']}/{entry_id}", json_data=data)

    # ==================== ACCOUNTING MODULE ====================

    def list_daily_ledger(self, page: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """List daily ledger entries."""
        if limit:
            return list(self._paginate(self.ACCOUNTING_PATHS["dailyledger"], limit=limit))
        return self._request("GET", self.ACCOUNTING_PATHS["dailyledger"], params={"page": page})

    def list_daily_ledger_filtered(
        self,
        account_id: Optional[str] = None,
        date_from: Optional[int] = None,
        date_to: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """List daily ledger entries with filtering.

        Args:
            account_id: Filter by account ID
            date_from: Unix timestamp for start date
            date_to: Unix timestamp for end date
            limit: Maximum items to return

        Returns:
            List of daily ledger entries matching filters
        """
        params: Dict[str, Any] = {}
        if account_id:
            params["accountId"] = account_id
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        if limit:
            return list(self._paginate(
                self.ACCOUNTING_PATHS["dailyledger"],
                params=params,
                limit=limit
            ))
        return self._request("GET", self.ACCOUNTING_PATHS["dailyledger"], params=params)

    def get_daily_ledger_entry(self, entry_id: str) -> Dict:
        """Get a single daily ledger entry by ID.

        Args:
            entry_id: Daily ledger entry ID

        Returns:
            Entry data including date, description, lines, etc.
        """
        return self._request("GET", f"{self.ACCOUNTING_PATHS['dailyledger']}/{entry_id}")

    def create_daily_ledger_entry(self, data: Dict[str, Any]) -> Dict:
        """Create a new daily ledger entry.

        Args:
            data: Entry data including:
                - date: Unix timestamp
                - desc: Description/reference
                - lines: List of {accountId, debit, credit, desc}

        Returns:
            API response with entry ID
        """
        return self._request("POST", self.ACCOUNTING_PATHS["dailyledger"], json_data=data)

    def update_daily_ledger_entry(self, entry_id: str, data: Dict[str, Any]) -> Dict:
        """Update an existing daily ledger entry.

        Args:
            entry_id: Entry ID to update
            data: Fields to update (date, desc, lines)

        Returns:
            Updated entry data
        """
        return self._request("PUT", f"{self.ACCOUNTING_PATHS['dailyledger']}/{entry_id}", json_data=data)

    def delete_daily_ledger_entry(self, entry_id: str) -> bool:
        """Delete a daily ledger entry.

        Args:
            entry_id: Entry ID to delete

        Returns:
            True if successful
        """
        self._request("DELETE", f"{self.ACCOUNTING_PATHS['dailyledger']}/{entry_id}")
        return True

    def list_chart_of_accounts(self, page: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """List chart of accounts.

        Args:
            page: Page number
            limit: Maximum items to return

        Returns:
            List of accounts with code, name, type, etc.
        """
        if limit:
            return list(self._paginate(self.ACCOUNTING_PATHS["chartofaccounts"], limit=limit))
        return self._request("GET", self.ACCOUNTING_PATHS["chartofaccounts"], params={"page": page})

    def get_account(self, account_id: str) -> Dict:
        """Get a single account from chart of accounts.

        Args:
            account_id: Account ID

        Returns:
            Account data including code, name, type, balance
        """
        return self._request("GET", f"{self.ACCOUNTING_PATHS['chartofaccounts']}/{account_id}")

    def get_account_by_code(self, code: str) -> Optional[Dict]:
        """Find an account by its code (e.g., '4300', '7000').

        Args:
            code: Account code to search for

        Returns:
            Account data if found, None otherwise
        """
        accounts = self.list_chart_of_accounts(limit=1000)
        for account in accounts:
            if account.get("code") == code:
                return account
        return None

    def create_account(self, data: Dict[str, Any]) -> Dict:
        """Create a new account in chart of accounts.

        Args:
            data: Account data including:
                - code: Account code (e.g., '4300')
                - name: Account name
                - type: Account type (asset, liability, equity, revenue, expense)

        Returns:
            API response with account ID
        """
        return self._request("POST", self.ACCOUNTING_PATHS["chartofaccounts"], json_data=data)

    def update_account(self, account_id: str, data: Dict[str, Any]) -> Dict:
        """Update an existing account.

        Args:
            account_id: Account ID to update
            data: Fields to update (name, code, type)

        Returns:
            Updated account data
        """
        return self._request("PUT", f"{self.ACCOUNTING_PATHS['chartofaccounts']}/{account_id}", json_data=data)

    def delete_account(self, account_id: str) -> bool:
        """Delete an account from chart of accounts.

        Warning: This may fail if the account has associated entries.

        Args:
            account_id: Account ID to delete

        Returns:
            True if successful
        """
        self._request("DELETE", f"{self.ACCOUNTING_PATHS['chartofaccounts']}/{account_id}")
        return True

    def create_journal_entry_with_builder(self, builder: "JournalEntryBuilder") -> Dict:
        """Create a journal entry using a JournalEntryBuilder instance.

        Args:
            builder: JournalEntryBuilder with lines configured

        Returns:
            API response with entry ID

        Example:
            builder = JournalEntryBuilder(description="Invoice F261277 paid")
            builder.add_debit(bank_id, 1210.00, "5720", "Deposit")
            builder.add_credit(receivable_id, 1210.00, "4300", "Customer payment")
            result = client.create_journal_entry_with_builder(builder)
        """
        data = builder.to_api_dict()
        return self.create_daily_ledger_entry(data)

    # ==================== GENERIC CRUD ====================

    def generic_list(self, module: str, resource: str, page: int = 1, limit: Optional[int] = None) -> List[Dict]:
        """Generic list operation for any resource.

        Args:
            module: Module name (invoicing, crm, projects, team, accounting)
            resource: Resource name (contacts, leads, projects, etc.)
            page: Page number
            limit: Maximum items to return
        """
        paths = {
            "invoicing": self.INVOICING_PATHS,
            "crm": self.CRM_PATHS,
            "projects": self.PROJECTS_PATHS,
            "team": self.TEAM_PATHS,
            "accounting": self.ACCOUNTING_PATHS,
        }

        module_paths = paths.get(module.lower())
        if not module_paths:
            raise ValueError(f"Unknown module: {module}. Available: {list(paths.keys())}")

        path = module_paths.get(resource.lower())
        if not path:
            raise ValueError(f"Unknown resource: {resource}. Available: {list(module_paths.keys())}")

        if limit:
            return list(self._paginate(path, limit=limit))
        return self._request("GET", path, params={"page": page})

    def generic_get(self, module: str, resource: str, resource_id: str) -> Dict:
        """Generic get operation for any resource."""
        paths = {
            "invoicing": self.INVOICING_PATHS,
            "crm": self.CRM_PATHS,
            "projects": self.PROJECTS_PATHS,
            "team": self.TEAM_PATHS,
            "accounting": self.ACCOUNTING_PATHS,
        }

        module_paths = paths.get(module.lower())
        if not module_paths:
            raise ValueError(f"Unknown module: {module}")

        path = module_paths.get(resource.lower())
        if not path:
            raise ValueError(f"Unknown resource: {resource}")

        return self._request("GET", f"{path}/{resource_id}")

    def generic_create(self, module: str, resource: str, data: Dict[str, Any]) -> Dict:
        """Generic create operation for any resource."""
        paths = {
            "invoicing": self.INVOICING_PATHS,
            "crm": self.CRM_PATHS,
            "projects": self.PROJECTS_PATHS,
            "team": self.TEAM_PATHS,
            "accounting": self.ACCOUNTING_PATHS,
        }

        module_paths = paths.get(module.lower())
        if not module_paths:
            raise ValueError(f"Unknown module: {module}")

        path = module_paths.get(resource.lower())
        if not path:
            raise ValueError(f"Unknown resource: {resource}")

        return self._request("POST", path, json_data=data)

    def generic_update(self, module: str, resource: str, resource_id: str, data: Dict[str, Any]) -> Dict:
        """Generic update operation for any resource."""
        paths = {
            "invoicing": self.INVOICING_PATHS,
            "crm": self.CRM_PATHS,
            "projects": self.PROJECTS_PATHS,
            "team": self.TEAM_PATHS,
            "accounting": self.ACCOUNTING_PATHS,
        }

        module_paths = paths.get(module.lower())
        if not module_paths:
            raise ValueError(f"Unknown module: {module}")

        path = module_paths.get(resource.lower())
        if not path:
            raise ValueError(f"Unknown resource: {resource}")

        return self._request("PUT", f"{path}/{resource_id}", json_data=data)

    def generic_delete(self, module: str, resource: str, resource_id: str) -> bool:
        """Generic delete operation for any resource."""
        paths = {
            "invoicing": self.INVOICING_PATHS,
            "crm": self.CRM_PATHS,
            "projects": self.PROJECTS_PATHS,
            "team": self.TEAM_PATHS,
            "accounting": self.ACCOUNTING_PATHS,
        }

        module_paths = paths.get(module.lower())
        if not module_paths:
            raise ValueError(f"Unknown module: {module}")

        path = module_paths.get(resource.lower())
        if not path:
            raise ValueError(f"Unknown resource: {resource}")

        self._request("DELETE", f"{path}/{resource_id}")
        return True

    # ==================== DOCUMENT BUILDER METHODS ====================

    def create_document_with_builder(self, builder: DocumentBuilder) -> Dict:
        """Create a document using a DocumentBuilder instance.

        Args:
            builder: DocumentBuilder with items and metadata configured

        Returns:
            API response with document ID, number, etc.

        Example:
            builder = DocumentBuilder(contact_id="abc123", doc_type="invoice")
            builder.add_item("Service", units=1, price=100, cost_price=60)
            result = client.create_document_with_builder(builder)
        """
        doc_type = builder.doc_type
        data = builder.to_api_dict()
        return self._request(
            "POST",
            f"{self.INVOICING_PATHS['documents']}/{doc_type}",
            json_data=data
        )

    def clone_document(
        self,
        source_id: str,
        target_type: str,
        new_contact_id: Optional[str] = None,
        new_date: Optional[int] = None,
    ) -> Dict:
        """Clone an existing document to a new type.

        Useful for converting estimates to invoices, etc.

        Args:
            source_id: ID of the document to clone
            target_type: Type for the new document (invoice, estimate, proform, etc.)
            new_contact_id: Override the contact (optional)
            new_date: Override the date (optional, defaults to now)

        Returns:
            API response with new document ID

        Example:
            # Convert estimate to invoice
            result = client.clone_document("abc123", "invoice")
        """
        source = self.get_document(source_id)

        builder = DocumentBuilder.from_document(source, doc_type=target_type)

        if new_contact_id:
            builder.contact_id = new_contact_id
        if new_date:
            builder.date = new_date
        else:
            builder.date = int(time.time())  # Reset to now

        return self.create_document_with_builder(builder)

    def get_document_with_margins(self, document_id: str, doc_type: Optional[str] = None) -> Dict:
        """Get document with calculated margins for each line item.

        Args:
            document_id: Document ID
            doc_type: Document type (invoice, estimate, etc.). Auto-detected if not provided.

        Adds computed fields to items:
        - _subtotal: Item subtotal after discount
        - _cost_total: Total cost for item
        - _margin: Margin for item
        - _margin_percent: Margin percentage

        And document-level:
        - _total_revenue: Total revenue before tax
        - _total_cost: Total cost
        - _total_margin: Total margin
        - _margin_percent: Overall margin percentage

        Returns:
            Document dict with additional computed margin fields
        """
        doc = self.get_document(document_id, doc_type)

        total_cost = 0.0
        total_revenue = 0.0
        total_margin = 0.0

        # Holded uses "products" for line items in the API response
        for item in doc.get("products", doc.get("items", [])):
            kind = item.get("kind", "line")
            if kind not in (None, "line"):
                continue

            units = float(item.get("units", 1))
            price = float(item.get("price", 0))
            cost = float(item.get("costPrice", 0))
            discount = float(item.get("discount", 0))
            supplied = item.get("supplied") == "Yes"

            # Calculate subtotal after discount
            subtotal = price * units * (1 - discount / 100)
            item_cost = cost * units
            margin = 0.0 if supplied else (price - cost) * units

            # Add calculated fields
            item["_subtotal"] = round(subtotal, 2)
            item["_cost_total"] = round(item_cost, 2)
            item["_margin"] = round(margin, 2)
            item["_margin_percent"] = round(((price - cost) / cost * 100), 1) if cost > 0 and not supplied else 0

            total_revenue += subtotal
            total_cost += item_cost
            total_margin += margin

        # Add document-level calculations
        doc["_total_revenue"] = round(total_revenue, 2)
        doc["_total_cost"] = round(total_cost, 2)
        doc["_total_margin"] = round(total_margin, 2)
        doc["_margin_percent"] = round((total_margin / total_cost * 100), 1) if total_cost > 0 else 0

        return doc

    def list_documents_filtered(
        self,
        doc_type: Optional[str] = None,
        contact_id: Optional[str] = None,
        date_from: Optional[int] = None,
        date_to: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """List documents with advanced filtering.

        Args:
            doc_type: Filter by document type (invoice, estimate, etc.)
            contact_id: Filter by contact ID
            date_from: Unix timestamp for start date
            date_to: Unix timestamp for end date
            limit: Maximum items to return

        Returns:
            List of documents matching filters
        """
        params: Dict[str, Any] = {}
        if contact_id:
            params["contactId"] = contact_id
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        # Holded API requires document type in path, not query param
        if doc_type:
            path = f"{self.INVOICING_PATHS['documents']}/{doc_type}"
        else:
            path = self.INVOICING_PATHS["documents"]

        if limit:
            return list(self._paginate(
                path,
                params=params,
                limit=limit
            ))
        return self._request("GET", path, params=params)

    # ==================== SERVICES CRUD ====================

    def get_service(self, service_id: str) -> Dict:
        """Get a single service by ID."""
        return self._request("GET", f"{self.INVOICING_PATHS['services']}/{service_id}")

    def create_service(self, data: Dict[str, Any]) -> Dict:
        """Create a new service.

        Args:
            data: Service data including:
                - name (str): Service name (required)
                - price (float): Sale price
                - costPrice (float): Cost price for margin
                - tax (int): Tax rate percentage
                - desc (str): Description

        Returns:
            API response with service ID

        Example:
            service = client.create_service({
                "name": "Consultoría logística",
                "price": 75,
                "costPrice": 45,
                "tax": 21,
                "desc": "Hora de consultoría"
            })
        """
        return self._request("POST", self.INVOICING_PATHS["services"], json_data=data)

    def update_service(self, service_id: str, data: Dict[str, Any]) -> Dict:
        """Update an existing service.

        Args:
            service_id: ID of the service to update
            data: Fields to update

        Returns:
            Updated service data
        """
        return self._request("PUT", f"{self.INVOICING_PATHS['services']}/{service_id}", json_data=data)

    def delete_service(self, service_id: str) -> bool:
        """Delete a service.

        Args:
            service_id: ID of the service to delete

        Returns:
            True if successful
        """
        self._request("DELETE", f"{self.INVOICING_PATHS['services']}/{service_id}")
        return True
