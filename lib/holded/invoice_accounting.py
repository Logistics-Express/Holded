"""Invoice journalization module for Holded accounting.

Automatically creates journal entries when invoices are created or paid,
following Spanish PGC (Plan General Contable) standards.

Usage:
    from holded.holded_client import HoldedClient
    from holded.invoice_accounting import InvoiceAccountant

    client = HoldedClient.from_credentials()
    accountant = InvoiceAccountant(client)

    # Preview journal entry for an invoice
    builder = accountant.journalize_invoice("invoice_id")
    print(builder.preview())

    # Create the entry
    result = accountant.journalize_invoice_and_post("invoice_id")

    # Record a payment
    from datetime import datetime
    payment_builder = accountant.journalize_payment("invoice_id", datetime.now())
    result = client.create_journal_entry_with_builder(payment_builder)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .holded_client import HoldedClient, JournalEntryBuilder


# Default account mapping based on Spanish PGC
DEFAULT_ACCOUNT_MAPPING = {
    "revenue": {
        "default": "7000",        # Ventas de mercaderías
        "services": "7050",       # Prestaciones de servicios
        "shipping": "7060",       # Servicios de transporte
    },
    "expenses": {
        "default": "6000",        # Compras de mercaderías
        "transport_cost": "6240", # Transportes
        "services": "6290",       # Otros servicios
    },
    "assets": {
        "bank_qonto": "5720",          # Bancos - cuenta Qonto
        "accounts_receivable": "4300", # Clientes
        "vat_receivable": "4720",      # HP IVA soportado
    },
    "liabilities": {
        "vat_payable": "4750",         # HP IVA repercutido
        "accounts_payable": "4000",    # Proveedores
        "retention_payable": "4751",   # HP acreedora retenciones
    },
}

ACCOUNT_MAPPING_FILE = Path.home() / ".holded_account_mapping.json"


def load_account_mapping() -> Dict[str, Any]:
    """Load account mapping from config file or use defaults.

    Returns:
        Account mapping dictionary
    """
    if ACCOUNT_MAPPING_FILE.exists():
        try:
            custom = json.loads(ACCOUNT_MAPPING_FILE.read_text())
            # Merge with defaults
            mapping = DEFAULT_ACCOUNT_MAPPING.copy()
            for category, accounts in custom.items():
                if category in mapping:
                    mapping[category].update(accounts)
                else:
                    mapping[category] = accounts
            return mapping
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_ACCOUNT_MAPPING


class InvoiceAccountant:
    """Handles automatic journalization of invoices and payments.

    Creates proper double-entry journal entries following Spanish PGC
    standards for:
    - Sales invoices (factura de venta)
    - Invoice payments received
    - Credit notes (abono)

    Account codes are configurable via ~/.holded_account_mapping.json
    """

    def __init__(
        self,
        client: "HoldedClient",
        mapping: Optional[Dict[str, Any]] = None,
    ):
        """Initialize accountant with client and mapping.

        Args:
            client: Authenticated HoldedClient instance
            mapping: Custom account mapping (uses config file or defaults if None)
        """
        self.client = client
        self.mapping = mapping or load_account_mapping()
        self._accounts_cache: Optional[Dict[str, Dict]] = None

    def _get_accounts_by_code(self) -> Dict[str, Dict]:
        """Get chart of accounts indexed by code."""
        if self._accounts_cache is None:
            accounts = self.client.list_chart_of_accounts(limit=1000)
            self._accounts_cache = {a.get("code", ""): a for a in accounts}
        return self._accounts_cache

    def _resolve_account_id(self, code: str) -> str:
        """Get account ID for a code, or raise if not found.

        Args:
            code: Account code (e.g., '4300')

        Returns:
            Holded account ID

        Raises:
            ValueError: If account code not found
        """
        accounts = self._get_accounts_by_code()
        account = accounts.get(code)
        if not account:
            raise ValueError(f"Account code '{code}' not found in chart of accounts")
        return account["id"]

    def _get_revenue_code(self, invoice: Dict[str, Any]) -> str:
        """Determine revenue account code based on invoice type/items.

        Args:
            invoice: Invoice document data

        Returns:
            Revenue account code (e.g., '7000', '7050')
        """
        # Could be enhanced to look at invoice items/categories
        # For now, use default services account for service-based business
        return self.mapping["revenue"].get("services", self.mapping["revenue"]["default"])

    def _get_bank_code(self, bank_account: Optional[str] = None) -> str:
        """Get bank account code.

        Args:
            bank_account: Optional bank account identifier

        Returns:
            Bank account code (e.g., '5720')
        """
        if bank_account and bank_account in self.mapping.get("assets", {}):
            return self.mapping["assets"][bank_account]
        return self.mapping["assets"].get("bank_qonto", "5720")

    def journalize_invoice(self, invoice_id: str) -> "JournalEntryBuilder":
        """Create journal entry for a sales invoice.

        Standard sales invoice entry:
            DR Accounts Receivable (4300) - Total with VAT
            CR Revenue (7000/7050)        - Net amount
            CR VAT Payable (4750)         - VAT amount

        If there's IRPF retention:
            DR Accounts Receivable (4300) - Total (net - retention + VAT)
            DR Retention Receivable (4730) - Retention amount
            CR Revenue (7000/7050)        - Net amount
            CR VAT Payable (4750)         - VAT amount

        Args:
            invoice_id: Holded invoice document ID

        Returns:
            JournalEntryBuilder ready for posting or preview
        """
        from .holded_client import JournalEntryBuilder

        invoice = self.client.get_document(invoice_id, doc_type="invoice")

        # Extract amounts
        subtotal = float(invoice.get("subtotal", 0))  # Net before tax
        tax_amount = float(invoice.get("tax", 0))     # IVA
        retention = float(invoice.get("retention", 0)) # IRPF retention
        total = float(invoice.get("total", 0))        # Final total

        # Get account IDs
        receivable_code = self.mapping["assets"]["accounts_receivable"]
        revenue_code = self._get_revenue_code(invoice)
        vat_code = self.mapping["liabilities"]["vat_payable"]

        receivable_id = self._resolve_account_id(receivable_code)
        revenue_id = self._resolve_account_id(revenue_code)
        vat_id = self._resolve_account_id(vat_code)

        # Build reference
        doc_num = invoice.get("docNumber", invoice.get("invoiceNum", ""))
        contact_name = invoice.get("contactName", "")
        description = f"Factura {doc_num} - {contact_name}"

        # Create builder
        builder = JournalEntryBuilder(
            description=description,
            reference=doc_num,
            date=invoice.get("date"),
        )

        # Add lines
        # Debit: Accounts Receivable (total amount customer owes)
        builder.add_debit(
            receivable_id,
            total,
            receivable_code,
            f"Cliente {contact_name}",
            "Clientes"
        )

        # Handle IRPF retention if present
        if retention > 0:
            retention_code = "4730"  # HP deudora retenciones
            try:
                retention_id = self._resolve_account_id(retention_code)
                builder.add_debit(
                    retention_id,
                    retention,
                    retention_code,
                    "Retención IRPF",
                    "HP deudora retenciones"
                )
            except ValueError:
                # If retention account doesn't exist, add to revenue
                pass

        # Credit: Revenue (net amount)
        builder.add_credit(
            revenue_id,
            subtotal,
            revenue_code,
            f"Ingresos {doc_num}",
            "Ingresos por servicios"
        )

        # Credit: VAT Payable (if any)
        if tax_amount > 0:
            builder.add_credit(
                vat_id,
                tax_amount,
                vat_code,
                f"IVA factura {doc_num}",
                "HP IVA repercutido"
            )

        return builder

    def journalize_payment(
        self,
        invoice_id: str,
        payment_date: datetime,
        bank_account: Optional[str] = None,
        amount: Optional[float] = None,
    ) -> "JournalEntryBuilder":
        """Create journal entry for invoice payment received.

        Payment entry:
            DR Bank (5720)                - Amount received
            CR Accounts Receivable (4300) - Amount cleared

        Args:
            invoice_id: Holded invoice document ID
            payment_date: Date payment was received
            bank_account: Bank account identifier (default: Qonto)
            amount: Payment amount (default: invoice total)

        Returns:
            JournalEntryBuilder ready for posting or preview
        """
        from .holded_client import JournalEntryBuilder

        invoice = self.client.get_document(invoice_id, doc_type="invoice")

        # Get payment amount
        payment_amount = amount if amount is not None else float(invoice.get("total", 0))

        # Get account IDs
        bank_code = self._get_bank_code(bank_account)
        receivable_code = self.mapping["assets"]["accounts_receivable"]

        bank_id = self._resolve_account_id(bank_code)
        receivable_id = self._resolve_account_id(receivable_code)

        # Build reference
        doc_num = invoice.get("docNumber", invoice.get("invoiceNum", ""))
        contact_name = invoice.get("contactName", "")
        description = f"Cobro factura {doc_num} - {contact_name}"

        # Create builder
        builder = JournalEntryBuilder(
            description=description,
            reference=f"COBRO-{doc_num}",
            date=int(payment_date.timestamp()),
        )

        # Debit: Bank (money received)
        builder.add_debit(
            bank_id,
            payment_amount,
            bank_code,
            f"Cobro {contact_name}",
            "Bancos"
        )

        # Credit: Accounts Receivable (clear the debt)
        builder.add_credit(
            receivable_id,
            payment_amount,
            receivable_code,
            f"Cobro factura {doc_num}",
            "Clientes"
        )

        return builder

    def journalize_invoice_and_post(self, invoice_id: str) -> Dict[str, Any]:
        """Create and post journal entry for invoice.

        Args:
            invoice_id: Holded invoice document ID

        Returns:
            API response with created entry ID
        """
        builder = self.journalize_invoice(invoice_id)
        return self.client.create_journal_entry_with_builder(builder)

    def journalize_payment_and_post(
        self,
        invoice_id: str,
        payment_date: datetime,
        bank_account: Optional[str] = None,
        amount: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create and post journal entry for payment.

        Args:
            invoice_id: Holded invoice document ID
            payment_date: Date payment was received
            bank_account: Bank account identifier (default: Qonto)
            amount: Payment amount (default: invoice total)

        Returns:
            API response with created entry ID
        """
        builder = self.journalize_payment(invoice_id, payment_date, bank_account, amount)
        return self.client.create_journal_entry_with_builder(builder)

    def journalize_credit_note(self, credit_note_id: str) -> "JournalEntryBuilder":
        """Create journal entry for a credit note (abono).

        Credit note reverses the original invoice entry:
            DR Revenue (7000/7050)        - Net amount
            DR VAT Payable (4750)         - VAT amount
            CR Accounts Receivable (4300) - Total

        Args:
            credit_note_id: Holded credit note document ID

        Returns:
            JournalEntryBuilder ready for posting or preview
        """
        from .holded_client import JournalEntryBuilder

        credit_note = self.client.get_document(credit_note_id, doc_type="creditnote")

        # Extract amounts (should be positive in credit note)
        subtotal = float(credit_note.get("subtotal", 0))
        tax_amount = float(credit_note.get("tax", 0))
        total = float(credit_note.get("total", 0))

        # Get account IDs
        receivable_code = self.mapping["assets"]["accounts_receivable"]
        revenue_code = self._get_revenue_code(credit_note)
        vat_code = self.mapping["liabilities"]["vat_payable"]

        receivable_id = self._resolve_account_id(receivable_code)
        revenue_id = self._resolve_account_id(revenue_code)
        vat_id = self._resolve_account_id(vat_code)

        # Build reference
        doc_num = credit_note.get("docNumber", credit_note.get("invoiceNum", ""))
        contact_name = credit_note.get("contactName", "")
        description = f"Abono {doc_num} - {contact_name}"

        # Create builder
        builder = JournalEntryBuilder(
            description=description,
            reference=doc_num,
            date=credit_note.get("date"),
        )

        # Reverse entries:
        # Debit: Revenue (reduce revenue)
        builder.add_debit(
            revenue_id,
            subtotal,
            revenue_code,
            f"Abono ingresos {doc_num}",
            "Ingresos por servicios"
        )

        # Debit: VAT Payable (reduce VAT liability)
        if tax_amount > 0:
            builder.add_debit(
                vat_id,
                tax_amount,
                vat_code,
                f"Abono IVA {doc_num}",
                "HP IVA repercutido"
            )

        # Credit: Accounts Receivable (reduce what customer owes)
        builder.add_credit(
            receivable_id,
            total,
            receivable_code,
            f"Abono cliente {contact_name}",
            "Clientes"
        )

        return builder

    def find_unjournalized_invoices(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100,
    ) -> list:
        """Find invoices that don't have corresponding journal entries.

        Compares invoice references in daily ledger with existing invoices.

        Args:
            date_from: Start date filter
            date_to: End date filter
            limit: Maximum invoices to check

        Returns:
            List of invoice IDs without journal entries
        """
        # Get invoices
        from_ts = int(date_from.timestamp()) if date_from else None
        to_ts = int(date_to.timestamp()) if date_to else None

        invoices = self.client.list_documents_filtered(
            doc_type="invoice",
            date_from=from_ts,
            date_to=to_ts,
            limit=limit,
        )

        # Get journal entries with invoice references
        entries = self.client.list_daily_ledger_filtered(
            date_from=from_ts,
            date_to=to_ts,
            limit=1000,
        )

        # Build set of journalized invoice numbers
        journalized = set()
        for entry in entries:
            ref = entry.get("reference", "") or entry.get("desc", "")
            # Look for invoice number patterns (F-, FA-, etc.)
            for inv in invoices:
                doc_num = inv.get("docNumber", inv.get("invoiceNum", ""))
                if doc_num and doc_num in ref:
                    journalized.add(inv.get("id"))

        # Return unjournalized
        unjournalized = []
        for inv in invoices:
            if inv.get("id") not in journalized:
                unjournalized.append({
                    "id": inv.get("id"),
                    "docNumber": inv.get("docNumber", inv.get("invoiceNum", "")),
                    "contactName": inv.get("contactName", ""),
                    "total": inv.get("total", 0),
                    "date": inv.get("date"),
                })

        return unjournalized

    def batch_journalize_invoices(
        self,
        invoice_ids: list,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Journalize multiple invoices at once.

        Args:
            invoice_ids: List of invoice IDs to journalize
            dry_run: If True, only preview without posting

        Returns:
            Dict with results: {posted: [], errors: [], previews: []}
        """
        results = {
            "posted": [],
            "errors": [],
            "previews": [],
        }

        for inv_id in invoice_ids:
            try:
                builder = self.journalize_invoice(inv_id)

                if dry_run:
                    results["previews"].append({
                        "invoice_id": inv_id,
                        "reference": builder.reference,
                        "total_debit": builder.total_debits,
                        "preview": builder.preview(),
                    })
                else:
                    response = self.client.create_journal_entry_with_builder(builder)
                    results["posted"].append({
                        "invoice_id": inv_id,
                        "reference": builder.reference,
                        "entry_id": response.get("id"),
                    })

            except Exception as e:
                results["errors"].append({
                    "invoice_id": inv_id,
                    "error": str(e),
                })

        return results
