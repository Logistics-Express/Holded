"""
Qonto Banking API Client

Provides access to Qonto banking operations for payment reconciliation.
API Documentation: https://docs.qonto.com

Company: LOGISTICS EXPRESS ADUANAS, S.L.U.
IBAN: ES2868880001604129394523
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from decimal import Decimal

import requests

logger = logging.getLogger("holded-api.qonto")

# Qonto API base URL
QONTO_API_BASE = "https://thirdparty.qonto.com/v2"


@dataclass
class QontoTransaction:
    """Represents a Qonto bank transaction."""
    id: str
    transaction_id: str
    amount: Decimal
    amount_cents: int
    currency: str
    side: str  # credit or debit
    operation_type: str
    label: str
    reference: Optional[str]
    settled_at: Optional[datetime]
    emitted_at: datetime
    status: str
    counterparty_name: Optional[str]
    counterparty_iban: Optional[str]
    note: Optional[str]

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "QontoTransaction":
        """Create from Qonto API response."""
        return cls(
            id=data.get("id", ""),
            transaction_id=data.get("transaction_id", ""),
            amount=Decimal(str(data.get("amount", 0))),
            amount_cents=data.get("amount_cents", 0),
            currency=data.get("currency", "EUR"),
            side=data.get("side", ""),
            operation_type=data.get("operation_type", ""),
            label=data.get("label", ""),
            reference=data.get("reference"),
            settled_at=datetime.fromisoformat(data["settled_at"].replace("Z", "+00:00")) if data.get("settled_at") else None,
            emitted_at=datetime.fromisoformat(data["emitted_at"].replace("Z", "+00:00")) if data.get("emitted_at") else datetime.now(),
            status=data.get("status", ""),
            counterparty_name=data.get("label"),  # Qonto uses label for counterparty
            counterparty_iban=data.get("counterparty_iban"),
            note=data.get("note"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "amount": float(self.amount),
            "currency": self.currency,
            "side": self.side,
            "operation_type": self.operation_type,
            "label": self.label,
            "reference": self.reference,
            "settled_at": self.settled_at.isoformat() if self.settled_at else None,
            "emitted_at": self.emitted_at.isoformat() if self.emitted_at else None,
            "status": self.status,
            "counterparty_name": self.counterparty_name,
            "counterparty_iban": self.counterparty_iban,
            "note": self.note,
        }


class QontoClient:
    """Client for Qonto Banking API.

    Usage:
        client = QontoClient(login="org-slug", secret_key="your-secret")
        transactions = client.list_transactions(side="credit", days=30)
    """

    def __init__(
        self,
        login: str,
        secret_key: str,
        iban: Optional[str] = None,
    ):
        """Initialize Qonto client.

        Args:
            login: Organization slug (e.g., "logistics-express-aduanas-sociedad-limitada-3984")
            secret_key: API secret key
            iban: Default IBAN for transactions (optional)
        """
        self.login = login
        self.secret_key = secret_key
        self.iban = iban
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"{login}:{secret_key}",
            "Content-Type": "application/json",
        })
        self._organization: Optional[Dict[str, Any]] = None
        self._bank_accounts: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_credentials_file(cls, path: str = "~/.qonto_credentials.json") -> "QontoClient":
        """Create client from credentials file."""
        import json
        import os

        path = os.path.expanduser(path)
        with open(path) as f:
            creds = json.load(f)

        return cls(
            login=creds["login"],
            secret_key=creds["secret_key"],
            iban=creds.get("iban"),
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make API request."""
        url = f"{QONTO_API_BASE}{endpoint}"

        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Qonto API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Qonto request failed: {e}")
            raise

    def get_organization(self) -> Dict[str, Any]:
        """Get organization details."""
        if self._organization is None:
            result = self._request("GET", "/organization")
            self._organization = result.get("organization", {})
        return self._organization

    def list_bank_accounts(self) -> List[Dict[str, Any]]:
        """List all bank accounts."""
        if self._bank_accounts is None:
            result = self._request("GET", "/organization")
            self._bank_accounts = result.get("organization", {}).get("bank_accounts", [])
        return self._bank_accounts

    def get_bank_account(self, iban: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get bank account by IBAN."""
        target_iban = iban or self.iban
        if not target_iban:
            accounts = self.list_bank_accounts()
            return accounts[0] if accounts else None

        for account in self.list_bank_accounts():
            if account.get("iban") == target_iban:
                return account
        return None

    def get_balance(self, iban: Optional[str] = None) -> Dict[str, Any]:
        """Get account balance."""
        account = self.get_bank_account(iban)
        if not account:
            return {"error": "Account not found"}

        return {
            "iban": account.get("iban"),
            "name": account.get("name"),
            "balance": account.get("balance"),
            "balance_cents": account.get("balance_cents"),
            "currency": account.get("currency", "EUR"),
            "authorized_balance": account.get("authorized_balance"),
            "updated_at": account.get("updated_at"),
        }

    def list_transactions(
        self,
        iban: Optional[str] = None,
        side: Optional[str] = None,  # "credit" or "debit"
        status: str = "completed",
        days: int = 30,
        settled_at_from: Optional[str] = None,
        settled_at_to: Optional[str] = None,
        per_page: int = 100,
        current_page: int = 1,
    ) -> List[QontoTransaction]:
        """List transactions for an account.

        Args:
            iban: Bank account IBAN (uses default if not provided)
            side: Filter by "credit" (incoming) or "debit" (outgoing)
            status: Transaction status (default: "completed")
            days: Number of days to look back (default: 30)
            settled_at_from: Start date (ISO format, overrides days)
            settled_at_to: End date (ISO format)
            per_page: Results per page (max 100)
            current_page: Page number

        Returns:
            List of QontoTransaction objects
        """
        target_iban = iban or self.iban
        if not target_iban:
            account = self.get_bank_account()
            target_iban = account.get("iban") if account else None

        if not target_iban:
            logger.error("No IBAN available for transaction listing")
            return []

        # Build params
        params = {
            "iban": target_iban,
            "status[]": status,
            "per_page": min(per_page, 100),
            "current_page": current_page,
        }

        if side:
            params["side[]"] = side

        # Date range
        if settled_at_from:
            params["settled_at_from"] = settled_at_from
        else:
            from_date = datetime.now() - timedelta(days=days)
            params["settled_at_from"] = from_date.strftime("%Y-%m-%dT00:00:00.000Z")

        if settled_at_to:
            params["settled_at_to"] = settled_at_to

        try:
            result = self._request("GET", "/transactions", params=params)
            transactions = result.get("transactions", [])
            return [QontoTransaction.from_api(t) for t in transactions]
        except Exception as e:
            logger.error(f"Error listing transactions: {e}")
            return []

    def list_credits(
        self,
        iban: Optional[str] = None,
        days: int = 30,
    ) -> List[QontoTransaction]:
        """List incoming payments (credits) for reconciliation."""
        return self.list_transactions(
            iban=iban,
            side="credit",
            days=days,
        )

    def list_debits(
        self,
        iban: Optional[str] = None,
        days: int = 30,
    ) -> List[QontoTransaction]:
        """List outgoing payments (debits)."""
        return self.list_transactions(
            iban=iban,
            side="debit",
            days=days,
        )

    def get_transaction(self, transaction_id: str) -> Optional[QontoTransaction]:
        """Get a single transaction by ID."""
        try:
            result = self._request("GET", f"/transactions/{transaction_id}")
            transaction = result.get("transaction")
            return QontoTransaction.from_api(transaction) if transaction else None
        except Exception as e:
            logger.error(f"Error getting transaction {transaction_id}: {e}")
            return None

    def search_transactions(
        self,
        query: str,
        iban: Optional[str] = None,
        days: int = 90,
    ) -> List[QontoTransaction]:
        """Search transactions by label or reference.

        Note: Qonto API doesn't have native search, so we fetch and filter locally.
        """
        transactions = self.list_transactions(iban=iban, days=days)
        query_lower = query.lower()

        matches = []
        for t in transactions:
            if query_lower in (t.label or "").lower():
                matches.append(t)
            elif query_lower in (t.reference or "").lower():
                matches.append(t)
            elif query_lower in (t.note or "").lower():
                matches.append(t)

        return matches

    def find_payment_for_invoice(
        self,
        invoice_number: str,
        amount: Optional[float] = None,
        days: int = 90,
        tolerance: float = 0.02,
    ) -> List[QontoTransaction]:
        """Find potential payment matches for an invoice.

        Searches by:
        1. Invoice number in label/reference
        2. Exact amount match (if provided)

        Args:
            invoice_number: Invoice number to search for (e.g., "FA-2026-0123")
            amount: Expected payment amount (optional)
            days: Days to search back
            tolerance: Amount tolerance in EUR (default 2 cents)

        Returns:
            List of matching transactions, sorted by relevance
        """
        credits = self.list_credits(days=days)
        matches = []

        # Normalize invoice number for matching
        invoice_normalized = invoice_number.upper().replace("-", "").replace(" ", "")

        for t in credits:
            score = 0

            # Check label for invoice reference
            label_normalized = (t.label or "").upper().replace("-", "").replace(" ", "")
            if invoice_normalized in label_normalized:
                score += 50
            elif invoice_number.upper() in (t.label or "").upper():
                score += 40

            # Check reference field
            ref_normalized = (t.reference or "").upper().replace("-", "").replace(" ", "")
            if invoice_normalized in ref_normalized:
                score += 30

            # Check amount match
            if amount is not None:
                amount_diff = abs(float(t.amount) - amount)
                if amount_diff <= tolerance:
                    score += 25
                elif amount_diff <= 1.0:
                    score += 10

            if score > 0:
                matches.append((score, t))

        # Sort by score descending
        matches.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in matches]
