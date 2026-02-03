"""Financial reporting module for Holded accounting.

Provides trial balance, income statement, balance sheet, and account ledger
reports based on chart of accounts and daily ledger data.

Usage:
    from holded.holded_client import HoldedClient
    from holded.accounting_reports import AccountingReporter

    client = HoldedClient.from_credentials()
    reporter = AccountingReporter(client)

    # Get trial balance
    tb = reporter.trial_balance()
    print(f"Total debits: {tb['totals']['debit']}")

    # Income statement for 2024
    from datetime import datetime
    pnl = reporter.income_statement(
        datetime(2024, 1, 1),
        datetime(2024, 12, 31)
    )
    print(f"Net income: {pnl['net_income']}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .holded_client import HoldedClient


# Spanish PGC (Plan General Contable) account type prefixes
ACCOUNT_TYPE_PREFIXES = {
    # Assets (Activo)
    "1": "equity",     # Financiación básica (sometimes equity/liability mix)
    "2": "asset",      # Activo no corriente (non-current assets)
    "3": "asset",      # Existencias (inventory)
    "4": "mixed",      # Acreedores y deudores (receivables/payables) - needs subclassification
    "5": "asset",      # Cuentas financieras (financial accounts, cash, bank)
    # P&L accounts
    "6": "expense",    # Compras y gastos (purchases and expenses)
    "7": "revenue",    # Ventas e ingresos (sales and revenue)
}

# More specific account ranges for Spanish PGC
ACCOUNT_RANGES = {
    # Liabilities within group 1
    ("10", "19"): "equity",       # Capital, reserves
    # Liabilities within group 4
    ("40", "41"): "liability",    # Proveedores (suppliers payable)
    ("43", "44"): "asset",        # Clientes (customers receivable)
    ("47", "47"): "liability",    # HP acreedora (tax payable)
    ("46", "46"): "asset",        # HP deudora (tax receivable)
    ("47", "48"): "liability",    # Passivos por impuestos
    # Financial accounts
    ("52", "52"): "liability",    # Deudas a corto plazo
    ("57", "57"): "asset",        # Tesorería (cash/bank)
}


def classify_account_type(code: str) -> str:
    """Classify account type based on Spanish PGC code.

    Args:
        code: Account code (e.g., '4300', '7000')

    Returns:
        One of: 'asset', 'liability', 'equity', 'revenue', 'expense'
    """
    if not code:
        return "unknown"

    # Check specific ranges first
    for (start, end), acc_type in ACCOUNT_RANGES.items():
        if code.startswith(start) or (start <= code[:2] <= end):
            return acc_type

    # Fall back to first digit classification
    first_digit = code[0]
    acc_type = ACCOUNT_TYPE_PREFIXES.get(first_digit, "unknown")

    # Handle mixed group 4
    if acc_type == "mixed":
        # Default: 40-42 = liability, 43-46 = asset
        if code[:2] in ("40", "41", "42"):
            return "liability"
        return "asset"

    return acc_type


@dataclass
class AccountBalance:
    """Represents an account with its balance for reports."""
    id: str
    code: str
    name: str
    account_type: str  # asset, liability, equity, revenue, expense
    debit: float = 0.0
    credit: float = 0.0

    @property
    def balance(self) -> float:
        """Calculate net balance (debit - credit for assets/expenses, credit - debit for others)."""
        if self.account_type in ("asset", "expense"):
            return self.debit - self.credit
        return self.credit - self.debit

    @property
    def normal_balance(self) -> float:
        """Balance in normal form (positive for typical entries)."""
        return abs(self.balance)


class AccountingReporter:
    """Financial reporting based on Holded accounting data.

    Generates standard financial reports:
    - Trial Balance: All accounts with debit/credit totals
    - Income Statement (P&L): Revenue - Expenses = Net Income
    - Balance Sheet: Assets = Liabilities + Equity
    - Account Ledger: Detailed transactions for a single account

    Note: Reports are calculated from daily ledger entries, not from
    Holded's built-in reports. This allows filtering by date range.
    """

    def __init__(self, client: "HoldedClient"):
        """Initialize reporter with Holded client.

        Args:
            client: Authenticated HoldedClient instance
        """
        self.client = client
        self._accounts_cache: Optional[Dict[str, Dict]] = None

    def _get_accounts_map(self) -> Dict[str, Dict]:
        """Get chart of accounts as a map of id -> account."""
        if self._accounts_cache is None:
            accounts = self.client.list_chart_of_accounts(limit=1000)
            self._accounts_cache = {a["id"]: a for a in accounts}
        return self._accounts_cache

    def _get_ledger_entries(
        self,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[Dict]:
        """Get daily ledger entries within date range."""
        from_ts = int(date_from.timestamp()) if date_from else None
        to_ts = int(date_to.timestamp()) if date_to else None

        return self.client.list_daily_ledger_filtered(
            date_from=from_ts,
            date_to=to_ts,
            limit=10000,
        )

    def trial_balance(self, as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate trial balance report.

        A trial balance lists all accounts with their debit and credit
        totals. Total debits should equal total credits.

        Args:
            as_of_date: Calculate balances as of this date (default: now)

        Returns:
            Dict with:
                - date: Report date (ISO format)
                - accounts: List of account balances
                - totals: {debit, credit}
                - is_balanced: Boolean
        """
        accounts_map = self._get_accounts_map()
        entries = self._get_ledger_entries(date_to=as_of_date)

        # Accumulate balances per account
        balances: Dict[str, AccountBalance] = {}

        for entry in entries:
            for line in entry.get("lines", []):
                acc_id = line.get("accountId", "")
                if not acc_id:
                    continue

                if acc_id not in balances:
                    acc_data = accounts_map.get(acc_id, {})
                    code = acc_data.get("code", "")
                    balances[acc_id] = AccountBalance(
                        id=acc_id,
                        code=code,
                        name=acc_data.get("name", "Unknown"),
                        account_type=classify_account_type(code),
                    )

                balances[acc_id].debit += float(line.get("debit", 0))
                balances[acc_id].credit += float(line.get("credit", 0))

        # Sort by account code
        sorted_balances = sorted(balances.values(), key=lambda a: a.code)

        # Calculate totals
        total_debit = sum(a.debit for a in sorted_balances)
        total_credit = sum(a.credit for a in sorted_balances)

        report_date = as_of_date or datetime.now()

        return {
            "date": report_date.strftime("%Y-%m-%d"),
            "accounts": [
                {
                    "id": a.id,
                    "code": a.code,
                    "name": a.name,
                    "type": a.account_type,
                    "debit": round(a.debit, 2),
                    "credit": round(a.credit, 2),
                    "balance": round(a.balance, 2),
                }
                for a in sorted_balances
                if a.debit != 0 or a.credit != 0  # Skip zero-balance accounts
            ],
            "totals": {
                "debit": round(total_debit, 2),
                "credit": round(total_credit, 2),
            },
            "is_balanced": abs(total_debit - total_credit) < 0.01,
        }

    def income_statement(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> Dict[str, Any]:
        """Generate income statement (P&L) report.

        The income statement shows revenue minus expenses for a period,
        resulting in net income (or loss).

        Args:
            date_from: Start of period
            date_to: End of period

        Returns:
            Dict with:
                - period: {from, to}
                - revenue: List of revenue accounts with amounts
                - expenses: List of expense accounts with amounts
                - total_revenue: Sum of revenue
                - total_expenses: Sum of expenses
                - net_income: Revenue - Expenses
        """
        accounts_map = self._get_accounts_map()
        entries = self._get_ledger_entries(date_from=date_from, date_to=date_to)

        # Accumulate P&L balances
        balances: Dict[str, AccountBalance] = {}

        for entry in entries:
            for line in entry.get("lines", []):
                acc_id = line.get("accountId", "")
                if not acc_id:
                    continue

                acc_data = accounts_map.get(acc_id, {})
                code = acc_data.get("code", "")
                acc_type = classify_account_type(code)

                # Only include P&L accounts (revenue/expense)
                if acc_type not in ("revenue", "expense"):
                    continue

                if acc_id not in balances:
                    balances[acc_id] = AccountBalance(
                        id=acc_id,
                        code=code,
                        name=acc_data.get("name", "Unknown"),
                        account_type=acc_type,
                    )

                balances[acc_id].debit += float(line.get("debit", 0))
                balances[acc_id].credit += float(line.get("credit", 0))

        # Separate revenue and expenses
        revenue_accounts = sorted(
            [a for a in balances.values() if a.account_type == "revenue"],
            key=lambda a: a.code
        )
        expense_accounts = sorted(
            [a for a in balances.values() if a.account_type == "expense"],
            key=lambda a: a.code
        )

        # Calculate totals
        # Revenue: normally credit balance (credit - debit = positive revenue)
        total_revenue = sum(a.credit - a.debit for a in revenue_accounts)
        # Expenses: normally debit balance (debit - credit = positive expense)
        total_expenses = sum(a.debit - a.credit for a in expense_accounts)
        net_income = total_revenue - total_expenses

        return {
            "period": {
                "from": date_from.strftime("%Y-%m-%d"),
                "to": date_to.strftime("%Y-%m-%d"),
            },
            "revenue": [
                {
                    "id": a.id,
                    "code": a.code,
                    "name": a.name,
                    "amount": round(a.credit - a.debit, 2),
                }
                for a in revenue_accounts
                if a.credit != a.debit  # Skip zero amounts
            ],
            "expenses": [
                {
                    "id": a.id,
                    "code": a.code,
                    "name": a.name,
                    "amount": round(a.debit - a.credit, 2),
                }
                for a in expense_accounts
                if a.debit != a.credit  # Skip zero amounts
            ],
            "total_revenue": round(total_revenue, 2),
            "total_expenses": round(total_expenses, 2),
            "net_income": round(net_income, 2),
        }

    def balance_sheet(self, as_of_date: datetime) -> Dict[str, Any]:
        """Generate balance sheet report.

        The balance sheet shows Assets = Liabilities + Equity at a point
        in time. Net income from the period is added to retained earnings.

        Args:
            as_of_date: Balance sheet date

        Returns:
            Dict with:
                - date: Report date
                - assets: List of asset accounts
                - liabilities: List of liability accounts
                - equity: List of equity accounts
                - total_assets: Sum of assets
                - total_liabilities: Sum of liabilities
                - total_equity: Sum of equity (including retained earnings)
                - total_liabilities_equity: Should equal total_assets
        """
        accounts_map = self._get_accounts_map()
        entries = self._get_ledger_entries(date_to=as_of_date)

        # Accumulate balances
        balances: Dict[str, AccountBalance] = {}

        for entry in entries:
            for line in entry.get("lines", []):
                acc_id = line.get("accountId", "")
                if not acc_id:
                    continue

                acc_data = accounts_map.get(acc_id, {})
                code = acc_data.get("code", "")
                acc_type = classify_account_type(code)

                if acc_id not in balances:
                    balances[acc_id] = AccountBalance(
                        id=acc_id,
                        code=code,
                        name=acc_data.get("name", "Unknown"),
                        account_type=acc_type,
                    )

                balances[acc_id].debit += float(line.get("debit", 0))
                balances[acc_id].credit += float(line.get("credit", 0))

        # Separate by type
        assets = sorted(
            [a for a in balances.values() if a.account_type == "asset"],
            key=lambda a: a.code
        )
        liabilities = sorted(
            [a for a in balances.values() if a.account_type == "liability"],
            key=lambda a: a.code
        )
        equity = sorted(
            [a for a in balances.values() if a.account_type == "equity"],
            key=lambda a: a.code
        )

        # P&L accounts for retained earnings calculation
        revenue = [a for a in balances.values() if a.account_type == "revenue"]
        expenses = [a for a in balances.values() if a.account_type == "expense"]

        # Calculate totals
        # Assets: debit - credit = positive
        total_assets = sum(a.debit - a.credit for a in assets)
        # Liabilities: credit - debit = positive
        total_liabilities = sum(a.credit - a.debit for a in liabilities)
        # Equity: credit - debit = positive
        total_equity = sum(a.credit - a.debit for a in equity)
        # Retained earnings (net income for the period)
        total_revenue = sum(a.credit - a.debit for a in revenue)
        total_expenses = sum(a.debit - a.credit for a in expenses)
        retained_earnings = total_revenue - total_expenses

        return {
            "date": as_of_date.strftime("%Y-%m-%d"),
            "assets": [
                {
                    "id": a.id,
                    "code": a.code,
                    "name": a.name,
                    "balance": round(a.debit - a.credit, 2),
                }
                for a in assets
                if abs(a.debit - a.credit) > 0.01
            ],
            "liabilities": [
                {
                    "id": a.id,
                    "code": a.code,
                    "name": a.name,
                    "balance": round(a.credit - a.debit, 2),
                }
                for a in liabilities
                if abs(a.credit - a.debit) > 0.01
            ],
            "equity": [
                {
                    "id": a.id,
                    "code": a.code,
                    "name": a.name,
                    "balance": round(a.credit - a.debit, 2),
                }
                for a in equity
                if abs(a.credit - a.debit) > 0.01
            ],
            "retained_earnings": round(retained_earnings, 2),
            "total_assets": round(total_assets, 2),
            "total_liabilities": round(total_liabilities, 2),
            "total_equity": round(total_equity + retained_earnings, 2),
            "total_liabilities_equity": round(total_liabilities + total_equity + retained_earnings, 2),
            "is_balanced": abs(total_assets - (total_liabilities + total_equity + retained_earnings)) < 0.01,
        }

    def account_ledger(
        self,
        account_id: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Generate detailed ledger for a single account.

        Shows all transactions affecting the account with running balance.

        Args:
            account_id: Holded account ID
            date_from: Start date filter (optional)
            date_to: End date filter (optional)

        Returns:
            Dict with:
                - account: Account info (id, code, name, type)
                - period: {from, to}
                - opening_balance: Balance before period
                - entries: List of transactions with running balance
                - closing_balance: Final balance
        """
        accounts_map = self._get_accounts_map()
        account_data = accounts_map.get(account_id)

        if not account_data:
            # Try to fetch directly
            account_data = self.client.get_account(account_id)

        code = account_data.get("code", "")
        acc_type = classify_account_type(code)

        # Get entries for opening balance (before date_from)
        opening_balance = 0.0
        if date_from:
            prior_entries = self._get_ledger_entries(date_to=date_from)
            for entry in prior_entries:
                for line in entry.get("lines", []):
                    if line.get("accountId") == account_id:
                        debit = float(line.get("debit", 0))
                        credit = float(line.get("credit", 0))
                        if acc_type in ("asset", "expense"):
                            opening_balance += debit - credit
                        else:
                            opening_balance += credit - debit

        # Get entries for the period
        entries = self._get_ledger_entries(date_from=date_from, date_to=date_to)

        # Build transaction list with running balance
        running_balance = opening_balance
        transactions = []

        for entry in entries:
            entry_date = entry.get("date", 0)
            if isinstance(entry_date, int):
                entry_date_str = datetime.fromtimestamp(entry_date).strftime("%Y-%m-%d")
            else:
                entry_date_str = str(entry_date)

            for line in entry.get("lines", []):
                if line.get("accountId") != account_id:
                    continue

                debit = float(line.get("debit", 0))
                credit = float(line.get("credit", 0))

                # Update running balance
                if acc_type in ("asset", "expense"):
                    running_balance += debit - credit
                else:
                    running_balance += credit - debit

                transactions.append({
                    "date": entry_date_str,
                    "entry_id": entry.get("id", ""),
                    "description": entry.get("desc", "") or line.get("desc", ""),
                    "reference": entry.get("reference", ""),
                    "debit": round(debit, 2) if debit else None,
                    "credit": round(credit, 2) if credit else None,
                    "balance": round(running_balance, 2),
                })

        return {
            "account": {
                "id": account_id,
                "code": code,
                "name": account_data.get("name", "Unknown"),
                "type": acc_type,
            },
            "period": {
                "from": date_from.strftime("%Y-%m-%d") if date_from else None,
                "to": date_to.strftime("%Y-%m-%d") if date_to else None,
            },
            "opening_balance": round(opening_balance, 2),
            "entries": transactions,
            "closing_balance": round(running_balance, 2),
            "total_debits": round(sum(t["debit"] or 0 for t in transactions), 2),
            "total_credits": round(sum(t["credit"] or 0 for t in transactions), 2),
        }

    def format_trial_balance(self, report: Dict[str, Any]) -> str:
        """Format trial balance as human-readable text."""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"  BALANCE DE COMPROBACIÓN - {report['date']}")
        lines.append(f"{'='*70}\n")
        lines.append(f"  {'Código':<8} {'Cuenta':<30} {'Debe':>12} {'Haber':>12}")
        lines.append(f"  {'-'*8} {'-'*30} {'-'*12} {'-'*12}")

        for acc in report["accounts"]:
            code = acc["code"][:6]
            name = acc["name"][:28]
            debit = f"{acc['debit']:,.2f}" if acc["debit"] else ""
            credit = f"{acc['credit']:,.2f}" if acc["credit"] else ""
            lines.append(f"  {code:<8} {name:<30} {debit:>12} {credit:>12}")

        lines.append(f"  {'-'*8} {'-'*30} {'-'*12} {'-'*12}")
        lines.append(f"  {'TOTAL':<39} {report['totals']['debit']:>12,.2f} {report['totals']['credit']:>12,.2f}")

        if report["is_balanced"]:
            lines.append(f"\n  [OK] Cuadrado")
        else:
            diff = report["totals"]["debit"] - report["totals"]["credit"]
            lines.append(f"\n  [!] Descuadre: {diff:,.2f}")

        return "\n".join(lines)

    def format_income_statement(self, report: Dict[str, Any]) -> str:
        """Format income statement as human-readable text."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  CUENTA DE RESULTADOS")
        lines.append(f"  {report['period']['from']} a {report['period']['to']}")
        lines.append(f"{'='*60}\n")

        lines.append("  INGRESOS")
        lines.append(f"  {'-'*56}")
        for acc in report["revenue"]:
            name = f"{acc['code']} {acc['name']}"[:40]
            lines.append(f"    {name:<40} {acc['amount']:>12,.2f}")
        lines.append(f"  {'-'*56}")
        lines.append(f"  {'Total Ingresos':<40} {report['total_revenue']:>12,.2f}")

        lines.append(f"\n  GASTOS")
        lines.append(f"  {'-'*56}")
        for acc in report["expenses"]:
            name = f"{acc['code']} {acc['name']}"[:40]
            lines.append(f"    {name:<40} {acc['amount']:>12,.2f}")
        lines.append(f"  {'-'*56}")
        lines.append(f"  {'Total Gastos':<40} {report['total_expenses']:>12,.2f}")

        lines.append(f"\n  {'='*56}")
        result_label = "BENEFICIO NETO" if report["net_income"] >= 0 else "PÉRDIDA NETA"
        lines.append(f"  {result_label:<40} {report['net_income']:>12,.2f}")

        return "\n".join(lines)

    def format_balance_sheet(self, report: Dict[str, Any]) -> str:
        """Format balance sheet as human-readable text."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  BALANCE DE SITUACIÓN - {report['date']}")
        lines.append(f"{'='*60}\n")

        lines.append("  ACTIVO")
        lines.append(f"  {'-'*56}")
        for acc in report["assets"]:
            name = f"{acc['code']} {acc['name']}"[:40]
            lines.append(f"    {name:<40} {acc['balance']:>12,.2f}")
        lines.append(f"  {'-'*56}")
        lines.append(f"  {'Total Activo':<40} {report['total_assets']:>12,.2f}")

        lines.append(f"\n  PASIVO")
        lines.append(f"  {'-'*56}")
        for acc in report["liabilities"]:
            name = f"{acc['code']} {acc['name']}"[:40]
            lines.append(f"    {name:<40} {acc['balance']:>12,.2f}")
        lines.append(f"  {'-'*56}")
        lines.append(f"  {'Total Pasivo':<40} {report['total_liabilities']:>12,.2f}")

        lines.append(f"\n  PATRIMONIO NETO")
        lines.append(f"  {'-'*56}")
        for acc in report["equity"]:
            name = f"{acc['code']} {acc['name']}"[:40]
            lines.append(f"    {name:<40} {acc['balance']:>12,.2f}")
        if report["retained_earnings"] != 0:
            lines.append(f"    {'Resultado del ejercicio':<40} {report['retained_earnings']:>12,.2f}")
        lines.append(f"  {'-'*56}")
        lines.append(f"  {'Total Patrimonio Neto':<40} {report['total_equity']:>12,.2f}")

        lines.append(f"\n  {'='*56}")
        lines.append(f"  {'Total Pasivo + Patrimonio':<40} {report['total_liabilities_equity']:>12,.2f}")

        if report["is_balanced"]:
            lines.append(f"\n  [OK] Cuadrado")
        else:
            diff = report["total_assets"] - report["total_liabilities_equity"]
            lines.append(f"\n  [!] Descuadre: {diff:,.2f}")

        return "\n".join(lines)

    def format_account_ledger(self, report: Dict[str, Any]) -> str:
        """Format account ledger as human-readable text."""
        lines = []
        acc = report["account"]
        lines.append(f"\n{'='*80}")
        lines.append(f"  MAYOR - {acc['code']} {acc['name']}")
        if report["period"]["from"] or report["period"]["to"]:
            period = f"{report['period']['from'] or 'inicio'} a {report['period']['to'] or 'hoy'}"
            lines.append(f"  Periodo: {period}")
        lines.append(f"{'='*80}\n")

        lines.append(f"  Saldo inicial: {report['opening_balance']:,.2f}")
        lines.append(f"\n  {'Fecha':<12} {'Descripción':<30} {'Debe':>12} {'Haber':>12} {'Saldo':>12}")
        lines.append(f"  {'-'*12} {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

        for entry in report["entries"]:
            date = entry["date"]
            desc = (entry["description"] or entry["reference"])[:28]
            debit = f"{entry['debit']:,.2f}" if entry["debit"] else ""
            credit = f"{entry['credit']:,.2f}" if entry["credit"] else ""
            balance = f"{entry['balance']:,.2f}"
            lines.append(f"  {date:<12} {desc:<30} {debit:>12} {credit:>12} {balance:>12}")

        lines.append(f"  {'-'*12} {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
        lines.append(f"  {'TOTALES':<43} {report['total_debits']:>12,.2f} {report['total_credits']:>12,.2f}")
        lines.append(f"\n  Saldo final: {report['closing_balance']:,.2f}")

        return "\n".join(lines)
