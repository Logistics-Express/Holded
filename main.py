"""
Holded API Service
Internal microservice for Holded ERP operations.

Endpoints:
- /health - Health check
- /api/v1/contacts - Contact CRUD
- /api/v1/documents - Document operations (invoices, estimates, proformas)
- /api/v1/products - Product catalog CRUD
- /api/v1/services - Service catalog CRUD
- /api/v1/payments - Payment management
- /api/v1/treasuries - Bank accounts
- /api/v1/accounting - Trial balance, ledger, journal entries
- /api/v1/debt - Debt/outstanding invoice checking
- /api/v1/contable/agent - Contable AI Agent (autonomous accounting)
"""

import os
import sys
import logging
import json
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date
from decimal import Decimal
from collections import defaultdict
import time
import asyncio

from fastapi import FastAPI, HTTPException, Query, Depends, Request, Header, Security
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Add lib directory to path for HoldedClient
# Supports both local development (claude-tools) and production (bundled)
LIB_PATHS = [
    Path(__file__).parent / "lib",  # Bundled in repo
    Path("/app/lib"),  # Docker container
    Path.home() / "Desarrollos/claude-tools/lib",  # Local development
]

lib_loaded = False
for lib_path in LIB_PATHS:
    if lib_path.exists():
        if str(lib_path) not in sys.path:
            sys.path.insert(0, str(lib_path))
        lib_loaded = True
        break

if not lib_loaded:
    raise RuntimeError(f"Could not find holded library in any of: {LIB_PATHS}")

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from holded.holded_client import HoldedClient, DocumentBuilder, JournalEntryBuilder

# Import app modules
from app.config import get_settings, Settings
from app.cache.redis_client import RedisClient, set_redis_client, get_redis_client
from app.cache.decorators import cached, invalidate_cache
from app.db.database import DatabaseManager, set_db_manager, get_db_manager
from app.middleware.audit import AuditMiddleware
from app.db.models import (
    Base,
    ContableActionQueue,
    ContableReconciliationMatch,
    ContableDailySnapshot,
)
from app.qonto.client import QontoClient, QontoTransaction

# OpenAI for Contable AI Agent
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# APScheduler for background jobs
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    AsyncIOScheduler = None
    CronTrigger = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("holded-api")

# Global client instances
_client: Optional[HoldedClient] = None
_openai_client: Optional["OpenAI"] = None
_qonto_client: Optional[QontoClient] = None
_scheduler: Optional["AsyncIOScheduler"] = None

# Rate limiting state (in-memory, per-service)
_rate_limit_state: Dict[str, List[float]] = defaultdict(list)


def get_qonto_client() -> Optional[QontoClient]:
    """Get or create QontoClient instance."""
    global _qonto_client
    if _qonto_client is None:
        settings = get_settings()
        if settings.qonto_secret_key:
            _qonto_client = QontoClient(
                login=settings.qonto_login,
                secret_key=settings.qonto_secret_key,
                iban=settings.qonto_iban,
            )
        else:
            # Try to load from credentials file
            try:
                _qonto_client = QontoClient.from_credentials_file()
            except Exception as e:
                logger.warning(f"Could not load Qonto credentials: {e}")
    return _qonto_client


def get_openai_client() -> Optional["OpenAI"]:
    """Get or create OpenAI client instance."""
    global _openai_client
    if _openai_client is None and OPENAI_AVAILABLE:
        settings = get_settings()
        if settings.openai_api_key:
            _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def get_client() -> HoldedClient:
    """Get or create HoldedClient instance."""
    global _client
    if _client is None:
        settings = get_settings()
        if settings.holded_api_key:
            _client = HoldedClient(settings.holded_api_key)
        else:
            api_key = os.getenv("HOLDED_API_KEY")
            if api_key:
                _client = HoldedClient(api_key)
            else:
                _client = HoldedClient.from_credentials()
    return _client


# ============================================================================
# SERVICE AUTHENTICATION & RATE LIMITING
# ============================================================================

# API key header for service-to-service auth
service_api_key_header = APIKeyHeader(name="X-Service-API-Key", auto_error=False)
service_name_header = APIKeyHeader(name="X-Service-Name", auto_error=False)


def check_rate_limit(service_name: str) -> bool:
    """Check if service is within rate limit. Returns True if allowed."""
    settings = get_settings()
    limit = settings.service_rate_limits.get(service_name, settings.rate_limit_default)

    now = time.time()
    window_start = now - 60  # 1 minute window

    # Clean old entries
    _rate_limit_state[service_name] = [
        t for t in _rate_limit_state[service_name] if t > window_start
    ]

    # Check limit
    if len(_rate_limit_state[service_name]) >= limit:
        return False

    # Record this request
    _rate_limit_state[service_name].append(now)
    return True


async def verify_service_auth(
    api_key: Optional[str] = Security(service_api_key_header),
    service_name: Optional[str] = Security(service_name_header),
) -> Tuple[str, str]:
    """Verify service API key and return (service_name, api_key).

    Raises HTTPException if auth fails or rate limit exceeded.
    """
    settings = get_settings()

    if not api_key or not service_name:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Service-API-Key or X-Service-Name header"
        )

    # Check if service is known
    expected_key = settings.service_api_keys.get(service_name)
    if not expected_key:
        raise HTTPException(
            status_code=401,
            detail=f"Unknown service: {service_name}"
        )

    # Verify API key
    if api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    # Check rate limit
    if not check_rate_limit(service_name):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for service {service_name}"
        )

    return service_name, api_key


# ============================================================================
# CONTABLE AI AGENT - OPENAI INTEGRATION
# ============================================================================

CONTABLE_SYSTEM_PROMPT = """Eres el Agente Contable de Logistics Express, un sistema de IA especializado
en contabilidad y finanzas españolas. Tu empresa es LOGISTICS EXPRESS ADUANAS, S.L.U. (NIF: B44995355).

REGLAS DE SEGURIDAD:
1. NUNCA ejecutes acciones por más de 500 EUR sin aprobación humana
2. NUNCA modifiques asientos contables ya cerrados
3. SIEMPRE genera borrador antes de ejecutar acciones irreversibles
4. Para proveedores nuevos, SIEMPRE requiere aprobación

CLASIFICACIÓN PGC (Plan General Contable español):
- 6200: Arrendamientos y cánones (alquiler, renting, leasing)
- 6210: Reparaciones y conservación (taller, mantenimiento)
- 6220: Servicios profesionales (abogado, asesor, consultor)
- 6230: Transportes (flete, envío, logística)
- 6240: Primas de seguros
- 6250: Servicios bancarios
- 6260: Publicidad y propaganda
- 6270: Suministros (luz, agua, teléfono, internet)
- 6280: Combustibles (gasolina, gasóleo, diesel)
- 6290: Otros servicios
- 6300: Impuesto sobre beneficios
- 6310: Otros tributos (tasas, IAE, IBI)
- 6400: Sueldos y salarios
- 6420: Seguridad Social
- 6620: Intereses de deudas
- 6690: Otros gastos financieros

NIVELES DE ESCALACIÓN PARA IMPAGOS:
- Nivel 1 (0-30 días): Monitorear
- Nivel 2 (31-60 días): Recordatorio cordial
- Nivel 3 (61-90 días): Notificación formal (citar Ley 3/2004)
- Nivel 4 (90+ días): Escalación legal

INTENTS DISPONIBLES:
- reconcile: Conciliar pago con factura
- mark_paid: Marcar documento como pagado
- classify_expense: Clasificar gasto a cuenta PGC
- create_escalation: Crear borrador de escalación de impago
- check_invoice: Consultar estado de factura

Responde SIEMPRE en JSON con la estructura:
{
  "intent": "string",
  "confidence": 0.0-1.0,
  "action_data": {...},
  "explanation": "string",
  "risk_assessment": {
    "level": "auto_execute|draft|reject",
    "reason": "string"
  }
}
"""

CONTABLE_FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "reconcile_payment",
            "description": "Conciliar un pago bancario con una factura o presupuesto en Holded",
            "parameters": {
                "type": "object",
                "properties": {
                    "payment_amount": {"type": "number", "description": "Importe del pago en EUR"},
                    "payment_reference": {"type": "string", "description": "Referencia del pago (número de factura, etc.)"},
                    "payment_date": {"type": "string", "description": "Fecha del pago (YYYY-MM-DD)"},
                    "document_id": {"type": "string", "description": "ID del documento en Holded (si se conoce)"},
                    "contact_name": {"type": "string", "description": "Nombre del cliente/proveedor"},
                },
                "required": ["payment_amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_expense",
            "description": "Clasificar un gasto a una cuenta del Plan General Contable español",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Descripción del gasto"},
                    "supplier_name": {"type": "string", "description": "Nombre del proveedor"},
                    "amount": {"type": "number", "description": "Importe del gasto en EUR"},
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_escalation",
            "description": "Crear borrador de escalación para factura impagada",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string", "description": "ID de la factura en Holded"},
                    "invoice_number": {"type": "string", "description": "Número de factura"},
                    "days_overdue": {"type": "integer", "description": "Días de retraso"},
                    "amount": {"type": "number", "description": "Importe de la factura"},
                    "contact_name": {"type": "string", "description": "Nombre del cliente"},
                },
                "required": ["invoice_id", "days_overdue"],
            },
        },
    },
]


def assess_risk(
    action_type: str,
    amount: Optional[float],
    confidence: float,
    is_new_supplier: bool = False,
) -> Tuple[str, str]:
    """Assess risk level for an action.

    Returns: (risk_level, reason)
    """
    settings = get_settings()

    # New supplier always requires draft
    if is_new_supplier and settings.new_supplier_always_draft:
        return "draft", "Proveedor nuevo - requiere revisión"

    # Low confidence always rejected
    if confidence < settings.draft_min_confidence:
        return "reject", f"Confianza muy baja ({confidence:.0%})"

    # High amount always requires draft
    if amount and amount > settings.auto_execute_max_amount:
        if amount > 5000:
            return "reject", f"Importe muy alto (€{amount:,.2f}) - requiere aprobación manual"
        return "draft", f"Importe superior a €{settings.auto_execute_max_amount} - requiere aprobación"

    # Medium confidence requires draft
    if confidence < settings.auto_execute_min_confidence:
        return "draft", f"Confianza media ({confidence:.0%}) - requiere revisión"

    # All checks passed - auto execute
    return "auto_execute", "Importe bajo y alta confianza"


async def process_with_openai(
    intent: str,
    parameters: Dict[str, Any],
    source_service: str,
) -> Dict[str, Any]:
    """Process an intent using OpenAI GPT-4o.

    Returns structured response with action data and risk assessment.
    """
    openai_client = get_openai_client()
    settings = get_settings()

    if not openai_client:
        # Fallback to rule-based processing
        return await process_rule_based(intent, parameters)

    # Build user message
    user_message = f"""
Intent: {intent}
Parámetros: {json.dumps(parameters, ensure_ascii=False)}
Servicio origen: {source_service}

Analiza esta solicitud y devuelve tu respuesta en JSON.
"""

    try:
        response = openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": CONTABLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            tools=CONTABLE_FUNCTION_TOOLS,
            tool_choice="auto",
            temperature=0.1,  # Low temperature for deterministic responses
            max_tokens=1000,
        )

        # Extract response
        message = response.choices[0].message

        # Check if function was called
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Determine confidence from model response
            confidence = 0.85  # Default for function calls

            # Get amount for risk assessment
            amount = function_args.get("amount") or function_args.get("payment_amount")

            # Assess risk
            risk_level, risk_reason = assess_risk(
                action_type=function_name,
                amount=amount,
                confidence=confidence,
            )

            return {
                "intent": function_name,
                "confidence": confidence,
                "action_data": function_args,
                "explanation": f"OpenAI seleccionó {function_name}",
                "risk_assessment": {
                    "level": risk_level,
                    "reason": risk_reason,
                },
                "ai_model": settings.openai_model,
            }

        # No function call - parse text response
        content = message.content or "{}"
        # Strip markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json) and last line (```)
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            result = json.loads(content)
            # Add risk assessment if not present
            if "risk_assessment" not in result:
                amount = result.get("action_data", {}).get("amount")
                confidence = result.get("confidence", 0.5)
                risk_level, risk_reason = assess_risk(
                    action_type=result.get("intent", "unknown"),
                    amount=amount,
                    confidence=confidence,
                )
                result["risk_assessment"] = {"level": risk_level, "reason": risk_reason}
            result["ai_model"] = settings.openai_model
            return result
        except json.JSONDecodeError:
            return {
                "intent": intent,
                "confidence": 0.5,
                "action_data": parameters,
                "explanation": content,
                "risk_assessment": {"level": "draft", "reason": "No se pudo parsear respuesta AI"},
                "ai_model": settings.openai_model,
            }

    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        # Fallback to rule-based
        return await process_rule_based(intent, parameters)


async def process_rule_based(
    intent: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Fallback rule-based processing when OpenAI is unavailable."""

    if intent == "classify_expense":
        description = parameters.get("description", "")
        supplier = parameters.get("supplier_name", "")
        classification = classify_expense(description, supplier)

        return {
            "intent": "classify_expense",
            "confidence": classification.confidence,
            "action_data": {
                "account_code": classification.account_code,
                "account_name": classification.account_name,
                "description": description,
            },
            "explanation": f"Clasificado como {classification.account_code} ({classification.account_name})",
            "risk_assessment": {
                "level": "auto_execute" if classification.confidence > 0.9 else "draft",
                "reason": classification.method,
            },
        }

    elif intent == "reconcile":
        amount = parameters.get("payment_amount")
        risk_level, risk_reason = assess_risk("reconcile", amount, 0.7)
        return {
            "intent": "reconcile",
            "confidence": 0.7,
            "action_data": parameters,
            "explanation": "Conciliación pendiente de validación",
            "risk_assessment": {"level": risk_level, "reason": risk_reason},
        }

    elif intent == "create_escalation":
        days = parameters.get("days_overdue", 0)
        level = 1 if days <= 30 else (2 if days <= 60 else (3 if days <= 90 else 4))
        return {
            "intent": "create_escalation",
            "confidence": 0.95,
            "action_data": {**parameters, "escalation_level": level},
            "explanation": f"Nivel de escalación {level} para {days} días de retraso",
            "risk_assessment": {"level": "draft", "reason": "Escalación siempre requiere aprobación"},
        }

    else:
        return {
            "intent": intent,
            "confidence": 0.5,
            "action_data": parameters,
            "explanation": "Intent no reconocido - requiere revisión manual",
            "risk_assessment": {"level": "reject", "reason": "Intent desconocido"},
        }


# ============================================================================
# BACKGROUND JOB FUNCTIONS
# ============================================================================

async def run_daily_reconciliation():
    """Background job: Daily Qonto-Holded reconciliation.

    Runs at configured hour (default 8 AM Madrid time).
    Fetches recent Qonto transactions and matches with pending Holded documents.
    """
    logger.info("Running daily reconciliation job...")
    try:
        # Get database manager
        db_manager = get_db_manager()

        # Get Holded client
        holded_client = get_client()

        # Get Qonto client
        qonto_client = get_qonto_client()
        if not qonto_client:
            logger.warning("Qonto client not available - skipping reconciliation")
            return

        # Get pending invoices and estimates from Holded
        pending_invoices = holded_client.list_documents("invoice", limit=500)
        pending_estimates = holded_client.list_documents("estimate", limit=500)

        pending_docs = []
        for doc in pending_invoices + pending_estimates:
            if not doc.get("paid"):
                pending_docs.append({
                    "id": doc.get("id"),
                    "number": doc.get("docNumber", ""),
                    "type": "invoice" if doc in pending_invoices else "estimate",
                    "amount": float(doc.get("total", 0)),
                    "contact_name": doc.get("contactName", ""),
                    "contact_id": doc.get("contactId"),
                })

        logger.info(f"Reconciliation: Found {len(pending_docs)} pending documents")

        # Get recent credits from Qonto (last 60 days)
        credits = qonto_client.list_credits(days=60)
        logger.info(f"Reconciliation: Found {len(credits)} Qonto credits")

        # Match payments to documents
        matches_found = 0
        for transaction in credits:
            # Skip small amounts (likely fees or returns)
            if float(transaction.amount) < 10:
                continue

            # Try to find matching document
            best_match = None
            best_confidence = 0

            for doc in pending_docs:
                confidence = 0

                # Exact amount match (within 2 cents tolerance)
                amount_diff = abs(float(transaction.amount) - doc["amount"])
                if amount_diff <= 0.02:
                    confidence += 0.4

                # Check for invoice/estimate number in transaction label or reference
                doc_number = doc["number"].upper().replace("-", "").replace(" ", "")
                label_normalized = (transaction.label or "").upper().replace("-", "").replace(" ", "")
                ref_normalized = (transaction.reference or "").upper().replace("-", "").replace(" ", "")

                if doc_number and doc_number in label_normalized:
                    confidence += 0.4
                elif doc_number and doc_number in ref_normalized:
                    confidence += 0.35

                # Check contact name similarity
                if doc["contact_name"]:
                    contact_lower = doc["contact_name"].lower()
                    label_lower = (transaction.label or "").lower()
                    # Simple substring match
                    if contact_lower in label_lower or label_lower in contact_lower:
                        confidence += 0.2

                if confidence > best_confidence and confidence >= 0.5:
                    best_confidence = confidence
                    best_match = doc

            # If we found a match, save it
            if best_match and db_manager:
                matches_found += 1
                async with db_manager.session() as session:
                    # Check if match already exists
                    from sqlalchemy import select
                    existing = await session.execute(
                        select(ContableReconciliationMatch).where(
                            ContableReconciliationMatch.qonto_transaction_id == transaction.transaction_id,
                            ContableReconciliationMatch.holded_document_id == best_match["id"],
                        )
                    )
                    if existing.scalar_one_or_none():
                        continue  # Already matched

                    # Determine match type
                    amount_diff = abs(float(transaction.amount) - best_match["amount"])
                    if amount_diff <= 0.02:
                        match_type = "exact_amount"
                    elif best_match["number"].upper() in (transaction.label or "").upper():
                        match_type = "reference"
                    else:
                        match_type = "name_similarity"

                    # Create match record
                    match_record = ContableReconciliationMatch(
                        qonto_transaction_id=transaction.transaction_id,
                        qonto_amount=Decimal(str(round(float(transaction.amount), 2))),
                        qonto_date=transaction.settled_at or transaction.emitted_at,
                        qonto_reference=transaction.reference,
                        qonto_counterparty=transaction.label,
                        holded_document_id=best_match["id"],
                        holded_document_type=best_match["type"],
                        holded_document_number=best_match["number"],
                        holded_amount=Decimal(str(round(best_match["amount"], 2))),
                        holded_contact_name=best_match["contact_name"],
                        match_confidence=Decimal(str(round(best_confidence, 2))),
                        match_type=match_type,
                        match_details={
                            "amount_diff": round(amount_diff, 2),
                            "transaction_label": transaction.label,
                        },
                        status="pending",
                    )
                    session.add(match_record)
                    await session.commit()

        logger.info(f"Reconciliation completed: {matches_found} new matches found")

    except Exception as e:
        logger.error(f"Reconciliation job error: {e}")


async def run_daily_snapshot():
    """Background job: Daily financial snapshot.

    Runs at configured hour (default 11 PM Madrid time).
    Captures end-of-day financial position for trend analysis.
    """
    logger.info("Running daily snapshot job...")
    try:
        db_manager = get_db_manager()
        if not db_manager:
            logger.warning("Database not available for snapshot job")
            return

        client = get_client()
        now = int(datetime.now().timestamp())
        today = date.today()

        # Get all invoices
        invoices = client.list_documents("invoice", limit=500)
        purchases = client.list_documents("purchase", limit=500)

        # Calculate receivables
        total_receivables = 0
        receivables_current = 0
        receivables_30 = 0
        receivables_60 = 0
        receivables_90 = 0
        receivables_90_plus = 0

        for inv in invoices:
            if inv.get("paid"):
                continue
            amount = float(inv.get("total", 0))
            total_receivables += amount

            due_date = inv.get("dueDate", now)
            if due_date >= now:
                receivables_current += amount
            else:
                days_overdue = (now - due_date) // 86400
                if days_overdue <= 30:
                    receivables_30 += amount
                elif days_overdue <= 60:
                    receivables_60 += amount
                elif days_overdue <= 90:
                    receivables_90 += amount
                else:
                    receivables_90_plus += amount

        # Calculate payables
        total_payables = 0
        payables_current = 0
        payables_overdue = 0

        for pur in purchases:
            if pur.get("paid"):
                continue
            amount = float(pur.get("total", 0))
            total_payables += amount

            due_date = pur.get("dueDate", now)
            if due_date >= now:
                payables_current += amount
            else:
                payables_overdue += amount

        # Get bank balance from Qonto
        bank_balance = None
        qonto = get_qonto_client()
        if qonto:
            try:
                balance_data = qonto.get_balance()
                if "balance" in balance_data:
                    bank_balance = float(balance_data["balance"])
            except Exception as e:
                logger.warning(f"Could not fetch Qonto balance: {e}")

        # Create snapshot record
        async with db_manager.session() as session:
            from sqlalchemy import select

            # Check if snapshot exists for today
            result = await session.execute(
                select(ContableDailySnapshot).where(
                    ContableDailySnapshot.snapshot_date == today
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing
                existing.total_receivables = Decimal(str(round(total_receivables, 2)))
                existing.receivables_current = Decimal(str(round(receivables_current, 2)))
                existing.receivables_overdue_30 = Decimal(str(round(receivables_30, 2)))
                existing.receivables_overdue_60 = Decimal(str(round(receivables_60, 2)))
                existing.receivables_overdue_90 = Decimal(str(round(receivables_90, 2)))
                existing.receivables_overdue_90_plus = Decimal(str(round(receivables_90_plus, 2)))
                existing.total_payables = Decimal(str(round(total_payables, 2)))
                existing.payables_current = Decimal(str(round(payables_current, 2)))
                existing.payables_overdue = Decimal(str(round(payables_overdue, 2)))
                if bank_balance is not None:
                    existing.bank_balance = Decimal(str(round(bank_balance, 2)))
            else:
                # Create new
                snapshot = ContableDailySnapshot(
                    snapshot_date=today,
                    total_receivables=Decimal(str(round(total_receivables, 2))),
                    receivables_current=Decimal(str(round(receivables_current, 2))),
                    receivables_overdue_30=Decimal(str(round(receivables_30, 2))),
                    receivables_overdue_60=Decimal(str(round(receivables_60, 2))),
                    receivables_overdue_90=Decimal(str(round(receivables_90, 2))),
                    receivables_overdue_90_plus=Decimal(str(round(receivables_90_plus, 2))),
                    total_payables=Decimal(str(round(total_payables, 2))),
                    payables_current=Decimal(str(round(payables_current, 2))),
                    payables_overdue=Decimal(str(round(payables_overdue, 2))),
                    bank_balance=Decimal(str(round(bank_balance, 2))) if bank_balance else None,
                )
                session.add(snapshot)

            await session.commit()

        balance_str = f", bank={bank_balance:.2f}" if bank_balance else ""
        logger.info(f"Snapshot created: receivables={total_receivables:.2f}, payables={total_payables:.2f}{balance_str}")

    except Exception as e:
        logger.error(f"Snapshot job error: {e}")


async def run_anomaly_check():
    """Background job: Daily anomaly detection.

    Runs at configured hour (default 9 AM Madrid time).
    Checks for duplicate invoices, unusual amounts, etc.
    """
    logger.info("Running daily anomaly check job...")
    try:
        client = get_client()
        db_manager = get_db_manager()

        # Get recent documents
        invoices = client.list_documents("invoice", limit=500)
        purchases = client.list_documents("purchase", limit=500)

        anomalies_found = 0

        # Check for duplicate invoice numbers
        invoice_numbers = {}
        for inv in invoices:
            num = inv.get("docNumber", "")
            if num in invoice_numbers:
                anomalies_found += 1
                logger.warning(f"Anomaly: Duplicate invoice number {num}")
            else:
                invoice_numbers[num] = inv.get("id")

        # Check for same-day duplicate purchases
        purchase_keys = {}
        for pur in purchases:
            supplier = pur.get("contactId", "")
            amount = round(float(pur.get("total", 0)), 2)
            pur_date = pur.get("date", 0)
            key = f"{supplier}:{amount}:{pur_date}"

            if key in purchase_keys and supplier:
                anomalies_found += 1
                logger.warning(f"Anomaly: Possible duplicate purchase for supplier {supplier}")
            else:
                purchase_keys[key] = pur.get("id")

        # Update daily snapshot with anomaly count
        if db_manager:
            async with db_manager.session() as session:
                from sqlalchemy import select
                today = date.today()
                result = await session.execute(
                    select(ContableDailySnapshot).where(
                        ContableDailySnapshot.snapshot_date == today
                    )
                )
                snapshot = result.scalar_one_or_none()
                if snapshot:
                    snapshot.anomalies_detected = anomalies_found
                    await session.commit()

        logger.info(f"Anomaly check completed: {anomalies_found} anomalies found")

    except Exception as e:
        logger.error(f"Anomaly check job error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _scheduler
    settings = get_settings()
    logger.info("Holded API service starting...")

    # Initialize Redis
    redis_client = None
    if settings.cache_enabled:
        try:
            redis_client = RedisClient(settings.redis_url)
            await redis_client.connect()
            set_redis_client(redis_client)
            logger.info("Redis cache enabled")
        except Exception as e:
            logger.warning(f"Redis connection failed, caching disabled: {e}")

    # Initialize Database
    db_manager = None
    if settings.audit_enabled:
        try:
            db_manager = DatabaseManager(settings.database_url)
            await db_manager.connect()
            set_db_manager(db_manager)
            logger.info("Audit logging enabled")
        except Exception as e:
            logger.warning(f"Database connection failed, audit disabled: {e}")

    # Initialize Scheduler for Contable Agent background jobs
    if settings.scheduler_enabled and SCHEDULER_AVAILABLE:
        try:
            _scheduler = AsyncIOScheduler()

            # Daily reconciliation job (8 AM Madrid time)
            _scheduler.add_job(
                run_daily_reconciliation,
                CronTrigger(hour=settings.reconciliation_job_hour, timezone="Europe/Madrid"),
                id="daily_reconciliation",
                name="Daily Qonto-Holded Reconciliation",
                replace_existing=True,
            )

            # Daily snapshot job (11 PM Madrid time)
            _scheduler.add_job(
                run_daily_snapshot,
                CronTrigger(hour=settings.snapshot_job_hour, timezone="Europe/Madrid"),
                id="daily_snapshot",
                name="Daily Financial Snapshot",
                replace_existing=True,
            )

            # Daily anomaly check (9 AM Madrid time)
            _scheduler.add_job(
                run_anomaly_check,
                CronTrigger(hour=settings.anomaly_check_job_hour, timezone="Europe/Madrid"),
                id="daily_anomaly_check",
                name="Daily Anomaly Detection",
                replace_existing=True,
            )

            _scheduler.start()
            logger.info("Background scheduler started with 3 daily jobs")
        except Exception as e:
            logger.warning(f"Scheduler initialization failed: {e}")
            _scheduler = None

    # Initialize OpenAI client
    if settings.contable_agent_enabled and settings.openai_api_key:
        openai_client = get_openai_client()
        if openai_client:
            logger.info(f"Contable AI Agent enabled (model: {settings.openai_model})")
        else:
            logger.warning("Contable AI Agent: OpenAI client not available")

    yield

    # Cleanup
    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("Background scheduler stopped")
    if redis_client:
        await redis_client.close()
    if db_manager:
        await db_manager.close()

    logger.info("Holded API service shutting down...")


app = FastAPI(
    title="Holded API Service",
    description="Internal microservice for Holded ERP operations",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for internal services
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Internal service
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audit middleware (added after CORS)
settings = get_settings()
app.add_middleware(AuditMiddleware, enabled=settings.audit_enabled)


# ============================================================================
# MODELS
# ============================================================================

class ContactCreate(BaseModel):
    """Create contact request."""
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    nif: Optional[str] = Field(None, description="NIF/CIF")
    address: Optional[str] = None


class EstimateCreate(BaseModel):
    """Create estimate/quotation request."""
    contact_name: str
    contact_email: str
    destination: str
    pallets: int
    price: float = Field(..., description="Price without IVA")
    origin: str = "Málaga"
    weight_kg: Optional[float] = None
    notes: Optional[str] = None
    send_email: bool = True


class EstimateItem(BaseModel):
    """Line item for estimate."""
    description: str
    quantity: float = 1
    price: float
    tax: int = 21


class GenericEstimateCreate(BaseModel):
    """Create generic estimate with custom items."""
    contact_name: str
    contact_email: Optional[str] = None
    items: List[EstimateItem]
    notes: Optional[str] = None
    send_email: bool = False


class DebtCheckRequest(BaseModel):
    """Debt check request."""
    contact_name: Optional[str] = None
    contact_id: Optional[str] = None
    nif: Optional[str] = None


class InvoiceLookupRequest(BaseModel):
    """Invoice lookup request."""
    invoice_number: str


# Phase 1: Critical Operations Models
class ContactUpdate(BaseModel):
    """Update contact request."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    nif: Optional[str] = Field(None, description="NIF/CIF")
    address: Optional[str] = None
    city: Optional[str] = None
    province: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None


class PaymentData(BaseModel):
    """Payment data for marking document as paid."""
    amount: float
    date: Optional[int] = Field(None, description="Unix timestamp")
    treasury_id: Optional[str] = Field(None, description="Bank account ID")


class CloneRequest(BaseModel):
    """Clone document request."""
    target_type: str = Field(..., description="Target doc type: invoice, salesorder, proform, etc.")
    new_contact_id: Optional[str] = None
    new_date: Optional[int] = Field(None, description="Unix timestamp, defaults to now")


# Phase 2: Catalog Management Models
class ProductCreate(BaseModel):
    """Create product request."""
    name: str
    price: float = Field(..., description="Sale price")
    cost_price: Optional[float] = Field(None, description="Cost price for margin calculation")
    tax: int = Field(21, description="Tax rate percentage")
    sku: Optional[str] = None
    desc: Optional[str] = None
    stock: Optional[float] = None


class ProductUpdate(BaseModel):
    """Update product request."""
    name: Optional[str] = None
    price: Optional[float] = None
    cost_price: Optional[float] = None
    tax: Optional[int] = None
    sku: Optional[str] = None
    desc: Optional[str] = None
    stock: Optional[float] = None


class ServiceCreate(BaseModel):
    """Create service request."""
    name: str
    price: float = Field(..., description="Sale price")
    cost_price: Optional[float] = Field(None, description="Cost price for margin calculation")
    tax: int = Field(21, description="Tax rate percentage")
    desc: Optional[str] = None


class ServiceUpdate(BaseModel):
    """Update service request."""
    name: Optional[str] = None
    price: Optional[float] = None
    cost_price: Optional[float] = None
    tax: Optional[int] = None
    desc: Optional[str] = None


# Phase 3: Financial Operations Models
class PaymentCreate(BaseModel):
    """Create payment request."""
    contact_id: str
    amount: float
    date: Optional[int] = Field(None, description="Unix timestamp")
    treasury_id: Optional[str] = Field(None, description="Bank account ID")
    notes: Optional[str] = None
    document_id: Optional[str] = Field(None, description="Link to invoice/document")


class DocumentUpdate(BaseModel):
    """Update document request."""
    notes: Optional[str] = None
    due_date: Optional[int] = Field(None, description="Unix timestamp")
    contact_id: Optional[str] = None


# Phase 4: Accounting Models
class JournalLineItem(BaseModel):
    """Journal entry line item."""
    account_id: str
    debit: float = 0
    credit: float = 0
    description: Optional[str] = None


class JournalEntryCreate(BaseModel):
    """Create journal entry request."""
    description: str
    reference: Optional[str] = None
    date: Optional[int] = Field(None, description="Unix timestamp")
    lines: List[JournalLineItem]


# Phase 5: Contable Agent Models
class ExpenseClassification(BaseModel):
    """Expense classification result."""
    account_code: str
    account_name: str
    confidence: float
    method: str  # keyword, supplier, ai
    alternative_codes: List[str] = []


class ReconciliationMatch(BaseModel):
    """Payment-document match for reconciliation."""
    payment_id: str
    payment_amount: float
    payment_date: str
    payment_reference: str
    document_id: str
    document_type: str
    document_number: str
    document_amount: float
    contact_name: str
    match_confidence: float
    match_type: str  # exact_amount, reference, name_similarity
    can_auto_approve: bool = False


class OverdueInvoice(BaseModel):
    """Overdue invoice with bucket classification."""
    invoice_id: str
    invoice_number: str
    contact_id: str
    contact_name: str
    amount: float
    due_date: str
    days_overdue: int
    bucket: str  # 0-30, 31-60, 61-90, 90+


class EscalationDraft(BaseModel):
    """Escalation draft for overdue invoice."""
    invoice_id: str
    invoice_number: str
    contact_name: str
    amount: float
    days_overdue: int
    escalation_level: int  # 1=friendly, 2=formal, 3=firm, 4=legal
    recommended_action: str
    email_draft: Optional[str] = None


class Anomaly(BaseModel):
    """Detected accounting anomaly."""
    anomaly_type: str  # duplicate_number, unusual_amount, same_day_duplicate
    severity: str  # low, medium, high
    document_id: str
    document_number: str
    description: str
    details: Dict[str, Any] = {}


# ============================================================================
# CONTABLE AI AGENT MODELS
# ============================================================================

class AgentExecuteRequest(BaseModel):
    """Request to execute an action via Contable AI Agent."""
    intent: str = Field(..., description="Intent: reconcile, mark_paid, classify_expense, create_escalation, check_invoice")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the intent")
    force_draft: bool = Field(False, description="Force draft mode even for low-risk actions")


class AgentExecuteResponse(BaseModel):
    """Response from Contable AI Agent execution."""
    action_id: str = Field(..., description="UUID of the action")
    status: str = Field(..., description="auto_executed, draft_created, rejected")
    risk_level: str = Field(..., description="auto_execute, draft, reject")
    confidence: float = Field(..., description="AI confidence 0.0-1.0")
    result: Optional[Dict[str, Any]] = Field(None, description="Result if auto-executed")
    explanation: str = Field(..., description="AI explanation")


class DraftAction(BaseModel):
    """Draft action waiting for approval."""
    id: str
    action_type: str
    action_data: Dict[str, Any]
    risk_level: str
    confidence: float
    amount_eur: Optional[float]
    ai_explanation: str
    source_service: Optional[str]
    created_at: str
    status: str


class DraftApproveRequest(BaseModel):
    """Request to approve a draft action."""
    reviewed_by: str = Field(..., description="Name/ID of reviewer")
    notes: Optional[str] = Field(None, description="Review notes")


class DraftRejectRequest(BaseModel):
    """Request to reject a draft action."""
    reviewed_by: str = Field(..., description="Name/ID of reviewer")
    reason: str = Field(..., description="Reason for rejection")


class AgentStatsResponse(BaseModel):
    """Statistics for Contable AI Agent."""
    total_actions: int
    actions_by_status: Dict[str, int]
    actions_by_type: Dict[str, int]
    auto_executed_today: int
    drafts_pending: int
    avg_confidence: float
    total_amount_processed: float


class SchedulerStatusResponse(BaseModel):
    """Status of background job scheduler."""
    enabled: bool
    running: bool
    jobs: List[Dict[str, Any]]
    next_runs: Dict[str, Optional[str]]


# Spanish PGC expense account classification hints
EXPENSE_ACCOUNT_HINTS = {
    # 62xx - Servicios exteriores
    "6200": {"name": "Arrendamientos y cánones", "keywords": ["alquiler", "renting", "leasing", "arrendamiento", "canon"]},
    "6210": {"name": "Reparaciones y conservación", "keywords": ["reparación", "reparacion", "mantenimiento", "taller", "mecánico", "mecanico"]},
    "6220": {"name": "Servicios profesionales independientes", "keywords": ["abogado", "asesor", "consultor", "notario", "gestor", "auditor"]},
    "6230": {"name": "Transportes", "keywords": ["transporte", "envío", "envio", "flete", "logística", "logistica", "mensajería", "paquetería"]},
    "6240": {"name": "Primas de seguros", "keywords": ["seguro", "póliza", "poliza", "aseguradora"]},
    "6250": {"name": "Servicios bancarios", "keywords": ["comisión bancaria", "comision bancaria", "transferencia", "swift", "sepa"]},
    "6260": {"name": "Publicidad y propaganda", "keywords": ["publicidad", "marketing", "anuncio", "promoción", "promocion", "google ads", "facebook"]},
    "6270": {"name": "Suministros", "keywords": ["luz", "agua", "gas", "electricidad", "teléfono", "telefono", "internet", "móvil", "movil"]},
    "6280": {"name": "Combustibles", "keywords": ["gasolina", "gasóleo", "gasoleo", "diesel", "combustible", "carburante", "repostaje"]},
    "6290": {"name": "Otros servicios", "keywords": ["limpieza", "vigilancia", "seguridad", "suscripción", "suscripcion", "software", "hosting"]},
    # 63xx - Tributos
    "6300": {"name": "Impuesto sobre beneficios", "keywords": ["impuesto sociedades", "is"]},
    "6310": {"name": "Otros tributos", "keywords": ["tasa", "arbitrio", "iae", "ibi", "ivtm"]},
    # 64xx - Gastos de personal
    "6400": {"name": "Sueldos y salarios", "keywords": ["nómina", "nomina", "salario", "sueldo"]},
    "6420": {"name": "Seguridad Social", "keywords": ["seguridad social", "ss", "cotización", "cotizacion"]},
    # 65xx - Otros gastos de gestión
    "6590": {"name": "Otros gastos de gestión corriente", "keywords": ["multa", "sanción", "sancion", "recargo"]},
    # 66xx - Gastos financieros
    "6620": {"name": "Intereses de deudas", "keywords": ["interés", "interes", "financiación", "financiacion", "préstamo", "prestamo"]},
    "6690": {"name": "Otros gastos financieros", "keywords": ["comisión financiera", "comision financiera", "descuento pronto pago"]},
}


def classify_expense(description: str, supplier_name: str = "") -> ExpenseClassification:
    """Classify expense to PGC account code based on description and supplier."""
    description_lower = description.lower()
    supplier_lower = supplier_name.lower() if supplier_name else ""

    best_match = None
    best_score = 0
    alternatives = []

    for code, hints in EXPENSE_ACCOUNT_HINTS.items():
        score = 0
        for keyword in hints["keywords"]:
            if keyword in description_lower:
                score += 2
            if keyword in supplier_lower:
                score += 1

        if score > 0:
            if score > best_score:
                if best_match:
                    alternatives.append(best_match[0])
                best_match = (code, hints["name"], score)
                best_score = score
            elif score == best_score and best_match:
                alternatives.append(code)

    if best_match:
        confidence = min(0.95, 0.5 + (best_score * 0.1))
        return ExpenseClassification(
            account_code=best_match[0],
            account_name=best_match[1],
            confidence=confidence,
            method="keyword",
            alternative_codes=alternatives[:3]
        )

    # Default to "Otros servicios"
    return ExpenseClassification(
        account_code="6290",
        account_name="Otros servicios",
        confidence=0.3,
        method="default",
        alternative_codes=["6230", "6280"]
    )


# ============================================================================
# HEALTH
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint with service status."""
    settings = get_settings()
    status = {
        "status": "healthy",
        "service": "holded-api",
        "version": "2.0.0",
    }

    # Check Redis
    redis = get_redis_client()
    if redis:
        status["redis"] = await redis.health_check()
    else:
        status["redis"] = {"status": "disabled"}

    # Check Database
    db = get_db_manager()
    if db:
        status["database"] = await db.health_check()
    else:
        status["database"] = {"status": "disabled"}

    # Check cache/audit feature flags
    status["features"] = {
        "cache_enabled": settings.cache_enabled and redis is not None,
        "audit_enabled": settings.audit_enabled and db is not None,
    }

    return status


# ============================================================================
# CONTACTS
# ============================================================================

@app.get("/api/v1/contacts")
@cached(ttl=300, prefix="contacts:list")
async def list_contacts(
    request: Request,
    limit: int = Query(50, le=500),
    client: HoldedClient = Depends(get_client)
):
    """List contacts."""
    try:
        contacts = client.list_contacts(limit=limit)
        return {"contacts": contacts, "count": len(contacts)}
    except Exception as e:
        logger.error(f"Error listing contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/contacts/{contact_id}")
@cached(ttl=600, prefix="contacts:id")
async def get_contact(
    request: Request,
    contact_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Get contact by ID."""
    try:
        contact = client.get_contact(contact_id)
        return contact
    except Exception as e:
        logger.error(f"Error getting contact: {e}")
        raise HTTPException(status_code=404, detail="Contact not found")


@app.get("/api/v1/contacts/search/{query}")
@cached(ttl=300, prefix="contacts:search")
async def search_contacts(
    request: Request,
    query: str,
    client: HoldedClient = Depends(get_client)
):
    """Search contacts by name, email, or NIF."""
    try:
        contacts = client.list_contacts(limit=500)
        query_lower = query.lower()

        matches = []
        for contact in contacts:
            name = contact.get("name", "").lower()
            email = contact.get("email", "").lower()
            nif = contact.get("code", "").lower()

            if query_lower in name or query_lower in email or query_lower in nif:
                matches.append(contact)

        return {"contacts": matches, "count": len(matches)}
    except Exception as e:
        logger.error(f"Error searching contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/contacts")
async def create_contact(
    data: ContactCreate,
    client: HoldedClient = Depends(get_client)
):
    """Create a new contact."""
    try:
        contact_data = {
            "name": data.name,
        }
        if data.email:
            contact_data["email"] = data.email
        if data.phone:
            contact_data["phone"] = data.phone
        if data.nif:
            contact_data["code"] = data.nif
        if data.address:
            contact_data["billAddress"] = {"address": data.address}

        result = client.create_contact(contact_data)

        # Invalidate contacts cache
        await invalidate_cache("holded:contacts:*")

        return {"success": True, "contact_id": result.get("id"), "contact": result}
    except Exception as e:
        logger.error(f"Error creating contact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/contacts/{contact_id}")
async def update_contact(
    contact_id: str,
    data: ContactUpdate,
    client: HoldedClient = Depends(get_client)
):
    """Update an existing contact."""
    try:
        update_data = {}
        if data.name is not None:
            update_data["name"] = data.name
        if data.email is not None:
            update_data["email"] = data.email
        if data.phone is not None:
            update_data["phone"] = data.phone
        if data.nif is not None:
            update_data["code"] = data.nif

        # Build address if any address fields provided
        address_fields = {}
        if data.address is not None:
            address_fields["address"] = data.address
        if data.city is not None:
            address_fields["city"] = data.city
        if data.province is not None:
            address_fields["province"] = data.province
        if data.postal_code is not None:
            address_fields["postalCode"] = data.postal_code
        if data.country is not None:
            address_fields["country"] = data.country

        if address_fields:
            update_data["billAddress"] = address_fields

        result = client.update_contact(contact_id, update_data)

        # Invalidate contacts cache
        await invalidate_cache("holded:contacts:*")

        return {"success": True, "contact_id": contact_id, "contact": result}
    except Exception as e:
        logger.error(f"Error updating contact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/contacts/{contact_id}")
async def delete_contact(
    contact_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Delete a contact."""
    try:
        client.delete_contact(contact_id)

        # Invalidate contacts cache
        await invalidate_cache("holded:contacts:*")

        return {"success": True, "deleted": contact_id}
    except Exception as e:
        logger.error(f"Error deleting contact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DOCUMENTS
# ============================================================================

@app.get("/api/v1/documents/invoices")
@cached(ttl=900, prefix="docs:invoices")
async def list_invoices(
    request: Request,
    limit: int = Query(50, le=500),
    contact_id: Optional[str] = None,
    year: Optional[int] = Query(None, description="Filter by year (e.g., 2024)"),
    date_from: Optional[int] = Query(None, description="Unix timestamp start"),
    date_to: Optional[int] = Query(None, description="Unix timestamp end"),
    client: HoldedClient = Depends(get_client)
):
    """List invoices with optional year/date filtering."""
    try:
        # Convert year to date range if provided
        if year and not date_from and not date_to:
            date_from = int(datetime(year, 1, 1).timestamp())
            date_to = int(datetime(year, 12, 31, 23, 59, 59).timestamp())

        invoices = client.list_documents_filtered(
            doc_type="invoice",
            contact_id=contact_id,
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )
        return {"invoices": invoices, "count": len(invoices)}
    except Exception as e:
        logger.error(f"Error listing invoices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/estimates")
@cached(ttl=900, prefix="docs:estimates")
async def list_estimates(
    request: Request,
    limit: int = Query(50, le=500),
    contact_id: Optional[str] = None,
    year: Optional[int] = Query(None, description="Filter by year (e.g., 2024)"),
    date_from: Optional[int] = Query(None, description="Unix timestamp start"),
    date_to: Optional[int] = Query(None, description="Unix timestamp end"),
    client: HoldedClient = Depends(get_client)
):
    """List estimates/quotations with optional year/date filtering."""
    try:
        # Convert year to date range if provided
        if year and not date_from and not date_to:
            date_from = int(datetime(year, 1, 1).timestamp())
            date_to = int(datetime(year, 12, 31, 23, 59, 59).timestamp())

        estimates = client.list_documents_filtered(
            doc_type="estimate",
            contact_id=contact_id,
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )
        return {"estimates": estimates, "count": len(estimates)}
    except Exception as e:
        logger.error(f"Error listing estimates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{doc_type}/year/{year}")
@cached(ttl=900, prefix="docs:year")
async def list_documents_by_year(
    request: Request,
    doc_type: str,
    year: int,
    limit: int = Query(500, le=1000),
    client: HoldedClient = Depends(get_client)
):
    """List all documents of a type for a specific year."""
    try:
        date_from = int(datetime(year, 1, 1).timestamp())
        date_to = int(datetime(year, 12, 31, 23, 59, 59).timestamp())

        docs = client.list_documents_filtered(
            doc_type=doc_type,
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )
        return {"documents": docs, "count": len(docs), "year": year}
    except Exception as e:
        logger.error(f"Error listing documents by year: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{doc_type}/{document_id}")
@cached(ttl=600, prefix="docs:single")
async def get_document(
    request: Request,
    doc_type: str,
    document_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Get document by ID."""
    try:
        doc = client.get_document(document_id, doc_type=doc_type)
        return doc
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/api/v1/documents/{doc_type}/{document_id}/send")
async def send_document(
    doc_type: str,
    document_id: str,
    emails: Optional[List[str]] = None,
    client: HoldedClient = Depends(get_client)
):
    """Send document by email."""
    try:
        result = client.send_document(document_id, emails=emails, doc_type=doc_type)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Error sending document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{doc_type}/{document_id}/pdf")
async def get_document_pdf(
    doc_type: str,
    document_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Download document as PDF."""
    try:
        pdf_bytes = client.get_document_pdf(document_id, doc_type=doc_type)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={doc_type}_{document_id}.pdf"
            }
        )
    except Exception as e:
        logger.error(f"Error getting document PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/{document_id}/clone")
async def clone_document(
    document_id: str,
    data: CloneRequest,
    client: HoldedClient = Depends(get_client)
):
    """Clone a document to a new type (e.g., estimate → invoice)."""
    try:
        result = client.clone_document(
            source_id=document_id,
            target_type=data.target_type,
            new_contact_id=data.new_contact_id,
            new_date=data.new_date,
        )

        # Invalidate cache for target document type
        await invalidate_cache(f"holded:docs:{data.target_type}:*")

        return {
            "success": True,
            "source_id": document_id,
            "new_document_id": result.get("id"),
            "new_document_number": result.get("docNumber"),
            "target_type": data.target_type,
        }
    except Exception as e:
        logger.error(f"Error cloning document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/{doc_type}/{document_id}/pay")
async def pay_document(
    doc_type: str,
    document_id: str,
    data: PaymentData,
    client: HoldedClient = Depends(get_client)
):
    """Mark a document as paid."""
    try:
        payment_data = {
            "amount": data.amount,
        }
        if data.date:
            payment_data["date"] = data.date
        else:
            payment_data["date"] = int(time.time())
        if data.treasury_id:
            payment_data["bankId"] = data.treasury_id

        result = client.pay_document(document_id, payment_data, doc_type=doc_type)

        # Invalidate document cache
        await invalidate_cache(f"holded:docs:{doc_type}:*")

        return {"success": True, "document_id": document_id, "result": result}
    except Exception as e:
        logger.error(f"Error paying document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/documents/{doc_type}/{document_id}")
async def update_document(
    doc_type: str,
    document_id: str,
    data: DocumentUpdate,
    client: HoldedClient = Depends(get_client)
):
    """Update a document (notes, due date, etc.)."""
    try:
        update_data = {}
        if data.notes is not None:
            update_data["notes"] = data.notes
        if data.due_date is not None:
            update_data["dueDate"] = data.due_date
        if data.contact_id is not None:
            update_data["contactId"] = data.contact_id

        result = client.update_document(document_id, update_data, doc_type=doc_type)

        # Invalidate document cache
        await invalidate_cache(f"holded:docs:{doc_type}:*")
        await invalidate_cache(f"holded:docs:single:*")

        return {"success": True, "document_id": document_id, "document": result}
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/documents/{doc_type}/{document_id}")
async def delete_document(
    doc_type: str,
    document_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Delete a document."""
    try:
        client.delete_document(document_id, doc_type=doc_type)

        # Invalidate document cache
        await invalidate_cache(f"holded:docs:{doc_type}:*")
        await invalidate_cache(f"holded:docs:single:*")

        return {"success": True, "deleted": document_id}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ESTIMATES/QUOTATIONS
# ============================================================================

@app.post("/api/v1/estimates/shipping")
async def create_shipping_estimate(
    data: EstimateCreate,
    client: HoldedClient = Depends(get_client)
):
    """Create a shipping estimate (presupuesto) and optionally send by email."""
    try:
        # 1. Find or create contact
        contact_id = await _find_or_create_contact(
            client, data.contact_name, data.contact_email
        )

        # 2. Build estimate
        builder = DocumentBuilder(contact_id=contact_id, doc_type="estimate")

        description = f"Transporte {data.origin} → {data.destination}"
        if data.weight_kg:
            description += f" ({data.weight_kg}kg)"

        builder.add_item(
            name=description,
            units=data.pallets,
            price=data.price / data.pallets,
            tax=21,
            desc=f"{data.pallets} europalets"
        )

        builder.set_addresses(
            pickup=f"Recogida en {data.origin}",
            delivery=f"Entrega en {data.destination}"
        )

        if data.notes:
            builder.notes = data.notes

        # 3. Create estimate
        result = client.create_document_with_builder(builder)

        estimate_id = result.get("id")
        estimate_number = result.get("docNumber")

        # 4. Send by email if requested
        sent = False
        if data.send_email and data.contact_email:
            try:
                client.send_document(estimate_id, [data.contact_email], "estimate")
                sent = True
            except Exception as e:
                logger.warning(f"Failed to send estimate: {e}")

        # 5. Invalidate estimates cache
        await invalidate_cache("holded:docs:estimates:*")

        return {
            "success": True,
            "estimate_id": estimate_id,
            "estimate_number": estimate_number,
            "total": round(data.price * 1.21, 2),
            "total_without_tax": data.price,
            "pdf_url": f"https://app.holded.com/document/{estimate_id}",
            "sent": sent
        }
    except Exception as e:
        logger.error(f"Error creating shipping estimate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/estimates")
async def create_estimate(
    data: GenericEstimateCreate,
    client: HoldedClient = Depends(get_client)
):
    """Create a generic estimate with custom items."""
    try:
        # Find or create contact
        contact_id = await _find_or_create_contact(
            client, data.contact_name, data.contact_email
        )

        # Build estimate
        builder = DocumentBuilder(contact_id=contact_id, doc_type="estimate")

        for item in data.items:
            builder.add_item(
                name=item.description,
                units=item.quantity,
                price=item.price,
                tax=item.tax
            )

        if data.notes:
            builder.notes = data.notes

        # Create estimate
        result = client.create_document_with_builder(builder)

        estimate_id = result.get("id")
        estimate_number = result.get("docNumber")

        # Send by email if requested
        sent = False
        if data.send_email and data.contact_email:
            try:
                client.send_document(estimate_id, [data.contact_email], "estimate")
                sent = True
            except Exception as e:
                logger.warning(f"Failed to send estimate: {e}")

        # Invalidate estimates cache
        await invalidate_cache("holded:docs:estimates:*")

        return {
            "success": True,
            "estimate_id": estimate_id,
            "estimate_number": estimate_number,
            "pdf_url": f"https://app.holded.com/document/{estimate_id}",
            "sent": sent
        }
    except Exception as e:
        logger.error(f"Error creating estimate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DEBT CHECKING
# ============================================================================

@app.post("/api/v1/debt/check")
async def check_outstanding_debt(
    data: DebtCheckRequest,
    client: HoldedClient = Depends(get_client)
):
    """Check outstanding debt for a contact."""
    try:
        # Find contact
        contact_id = data.contact_id
        if not contact_id:
            contact_id = await _find_contact_id(
                client, data.contact_name, data.nif
            )

        if not contact_id:
            return {
                "found": False,
                "has_debt": False,
                "error": "contact_not_found"
            }

        # Get invoices for contact
        invoices = client.list_documents_filtered(
            doc_type="invoice",
            contact_id=contact_id,
            limit=100
        )

        # Filter to pending/overdue
        from datetime import datetime
        now = datetime.now().timestamp()

        pending_docs = []
        total_outstanding = 0
        overdue_count = 0
        oldest_overdue_days = 0

        for inv in invoices:
            status = inv.get("status", "")
            if status not in ("pending", "overdue", "unpaid"):
                continue

            total = inv.get("total", 0)
            total_outstanding += total

            due_date = inv.get("dueDate", 0)
            days_overdue = 0
            if due_date and due_date < now:
                days_overdue = int((now - due_date) / 86400)
                overdue_count += 1
                oldest_overdue_days = max(oldest_overdue_days, days_overdue)

            pending_docs.append({
                "id": inv.get("id"),
                "number": inv.get("docNumber"),
                "total": total,
                "date": inv.get("date"),
                "due_date": due_date,
                "days_overdue": days_overdue,
                "status": status
            })

        return {
            "found": True,
            "contact_id": contact_id,
            "has_debt": total_outstanding > 0,
            "total_outstanding": round(total_outstanding, 2),
            "pending_count": len(pending_docs),
            "overdue_count": overdue_count,
            "oldest_overdue_days": oldest_overdue_days,
            "documents": pending_docs[:10]  # Limit for API response
        }
    except Exception as e:
        logger.error(f"Error checking debt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# INVOICE LOOKUP
# ============================================================================

@app.post("/api/v1/invoices/lookup")
@cached(ttl=1800, prefix="lookup:invoice")
async def lookup_invoice(
    request: Request,
    data: InvoiceLookupRequest,
    year: Optional[int] = Query(None, description="Limit search to specific year"),
    client: HoldedClient = Depends(get_client)
):
    """Look up invoice by number with optional year filter."""
    try:
        # Normalize invoice number
        invoice_number = data.invoice_number.upper().strip()

        # Remove common prefixes if partial
        for prefix in ["FA-", "FAC-", "F-"]:
            if invoice_number.startswith(prefix):
                invoice_number = invoice_number[len(prefix):]

        # Build date filter
        date_from = date_to = None
        if year:
            date_from = int(datetime(year, 1, 1).timestamp())
            date_to = int(datetime(year, 12, 31, 23, 59, 59).timestamp())

        # Search invoices
        invoices = client.list_documents_filtered(
            doc_type="invoice",
            date_from=date_from,
            date_to=date_to,
            limit=500
        )

        for inv in invoices:
            doc_number = inv.get("docNumber", "").upper()
            if invoice_number in doc_number or doc_number.endswith(invoice_number):
                # Get contact name
                contact_id = inv.get("contact")
                contact_name = "Unknown"
                if contact_id:
                    try:
                        contact = client.get_contact(contact_id)
                        contact_name = contact.get("name", "Unknown")
                    except Exception:
                        pass

                return {
                    "found": True,
                    "invoice_id": inv.get("id"),
                    "invoice_number": inv.get("docNumber"),
                    "total": inv.get("total"),
                    "status": inv.get("status"),
                    "date": inv.get("date"),
                    "due_date": inv.get("dueDate"),
                    "contact_id": contact_id,
                    "contact_name": contact_name,
                    "pdf_url": f"https://app.holded.com/document/{inv.get('id')}"
                }

        return {"found": False, "error": "invoice_not_found"}
    except Exception as e:
        logger.error(f"Error looking up invoice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/invoices/{invoice_number}")
@cached(ttl=1800, prefix="lookup:invoice:get")
async def get_invoice_by_number(
    request: Request,
    invoice_number: str,
    year: Optional[int] = Query(None, description="Limit search to specific year"),
    client: HoldedClient = Depends(get_client)
):
    """Get invoice by number (convenience endpoint) with optional year filter."""
    from pydantic import BaseModel as BM

    class Req(BM):
        invoice_number: str

    return await lookup_invoice(request, Req(invoice_number=invoice_number), year, client)


# ============================================================================
# HELPERS
# ============================================================================

async def _find_or_create_contact(
    client: HoldedClient,
    name: str,
    email: Optional[str] = None
) -> str:
    """Find existing contact or create new one."""
    # Search by name
    contacts = client.list_contacts(limit=500)

    name_lower = name.lower()
    for contact in contacts:
        if contact.get("name", "").lower() == name_lower:
            return contact.get("id")

    # Search by email
    if email:
        email_lower = email.lower()
        for contact in contacts:
            if contact.get("email", "").lower() == email_lower:
                return contact.get("id")

    # Create new contact
    contact_data = {"name": name, "tags": ["jarvis", "voice-api"]}
    if email:
        contact_data["email"] = email

    result = client.create_contact(contact_data)

    # Invalidate contacts cache since we created a new one
    await invalidate_cache("holded:contacts:*")

    return result.get("id")


async def _find_contact_id(
    client: HoldedClient,
    name: Optional[str] = None,
    nif: Optional[str] = None
) -> Optional[str]:
    """Find contact by name or NIF."""
    if not name and not nif:
        return None

    contacts = client.list_contacts(limit=500)

    if nif:
        nif_normalized = nif.upper().replace("-", "").replace(" ", "")
        for contact in contacts:
            contact_nif = contact.get("code", "").upper().replace("-", "").replace(" ", "")
            if contact_nif == nif_normalized:
                return contact.get("id")

    if name:
        name_lower = name.lower()
        for contact in contacts:
            if name_lower in contact.get("name", "").lower():
                return contact.get("id")

    return None


# ============================================================================
# PRODUCTS (Phase 2)
# ============================================================================

@app.get("/api/v1/products")
@cached(ttl=900, prefix="products:list")
async def list_products(
    request: Request,
    limit: int = Query(50, le=500),
    client: HoldedClient = Depends(get_client)
):
    """List products."""
    try:
        products = client.list_products(limit=limit)
        return {"products": products, "count": len(products)}
    except Exception as e:
        logger.error(f"Error listing products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/products/{product_id}")
@cached(ttl=600, prefix="products:id")
async def get_product(
    request: Request,
    product_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Get product by ID."""
    try:
        product = client.get_product(product_id)
        return product
    except Exception as e:
        logger.error(f"Error getting product: {e}")
        raise HTTPException(status_code=404, detail="Product not found")


@app.post("/api/v1/products")
async def create_product(
    data: ProductCreate,
    client: HoldedClient = Depends(get_client)
):
    """Create a new product."""
    try:
        product_data = {
            "name": data.name,
            "price": data.price,
            "tax": data.tax,
        }
        if data.cost_price is not None:
            product_data["costPrice"] = data.cost_price
        if data.sku:
            product_data["sku"] = data.sku
        if data.desc:
            product_data["desc"] = data.desc
        if data.stock is not None:
            product_data["stock"] = data.stock

        result = client.create_product(product_data)

        # Invalidate products cache
        await invalidate_cache("holded:products:*")

        return {"success": True, "product_id": result.get("id"), "product": result}
    except Exception as e:
        logger.error(f"Error creating product: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/products/{product_id}")
async def update_product(
    product_id: str,
    data: ProductUpdate,
    client: HoldedClient = Depends(get_client)
):
    """Update an existing product."""
    try:
        update_data = {}
        if data.name is not None:
            update_data["name"] = data.name
        if data.price is not None:
            update_data["price"] = data.price
        if data.cost_price is not None:
            update_data["costPrice"] = data.cost_price
        if data.tax is not None:
            update_data["tax"] = data.tax
        if data.sku is not None:
            update_data["sku"] = data.sku
        if data.desc is not None:
            update_data["desc"] = data.desc
        if data.stock is not None:
            update_data["stock"] = data.stock

        result = client.update_product(product_id, update_data)

        # Invalidate products cache
        await invalidate_cache("holded:products:*")

        return {"success": True, "product_id": product_id, "product": result}
    except Exception as e:
        logger.error(f"Error updating product: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/products/{product_id}")
async def delete_product(
    product_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Delete a product."""
    try:
        client.delete_product(product_id)

        # Invalidate products cache
        await invalidate_cache("holded:products:*")

        return {"success": True, "deleted": product_id}
    except Exception as e:
        logger.error(f"Error deleting product: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SERVICES (Phase 2)
# ============================================================================

@app.get("/api/v1/services")
@cached(ttl=900, prefix="services:list")
async def list_services(
    request: Request,
    limit: int = Query(50, le=500),
    client: HoldedClient = Depends(get_client)
):
    """List services."""
    try:
        services = client.list_services()
        return {"services": services, "count": len(services)}
    except Exception as e:
        logger.error(f"Error listing services: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/services/{service_id}")
@cached(ttl=600, prefix="services:id")
async def get_service(
    request: Request,
    service_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Get service by ID."""
    try:
        service = client.get_service(service_id)
        return service
    except Exception as e:
        logger.error(f"Error getting service: {e}")
        raise HTTPException(status_code=404, detail="Service not found")


@app.post("/api/v1/services")
async def create_service(
    data: ServiceCreate,
    client: HoldedClient = Depends(get_client)
):
    """Create a new service."""
    try:
        service_data = {
            "name": data.name,
            "price": data.price,
            "tax": data.tax,
        }
        if data.cost_price is not None:
            service_data["costPrice"] = data.cost_price
        if data.desc:
            service_data["desc"] = data.desc

        result = client.create_service(service_data)

        # Invalidate services cache
        await invalidate_cache("holded:services:*")

        return {"success": True, "service_id": result.get("id"), "service": result}
    except Exception as e:
        logger.error(f"Error creating service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/services/{service_id}")
async def update_service(
    service_id: str,
    data: ServiceUpdate,
    client: HoldedClient = Depends(get_client)
):
    """Update an existing service."""
    try:
        update_data = {}
        if data.name is not None:
            update_data["name"] = data.name
        if data.price is not None:
            update_data["price"] = data.price
        if data.cost_price is not None:
            update_data["costPrice"] = data.cost_price
        if data.tax is not None:
            update_data["tax"] = data.tax
        if data.desc is not None:
            update_data["desc"] = data.desc

        result = client.update_service(service_id, update_data)

        # Invalidate services cache
        await invalidate_cache("holded:services:*")

        return {"success": True, "service_id": service_id, "service": result}
    except Exception as e:
        logger.error(f"Error updating service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/services/{service_id}")
async def delete_service(
    service_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Delete a service."""
    try:
        client.delete_service(service_id)

        # Invalidate services cache
        await invalidate_cache("holded:services:*")

        return {"success": True, "deleted": service_id}
    except Exception as e:
        logger.error(f"Error deleting service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PAYMENTS (Phase 3)
# ============================================================================

@app.get("/api/v1/payments")
@cached(ttl=300, prefix="payments:list")
async def list_payments(
    request: Request,
    limit: int = Query(50, le=500),
    client: HoldedClient = Depends(get_client)
):
    """List payments."""
    try:
        payments = client.list_payments(limit=limit)
        return {"payments": payments, "count": len(payments)}
    except Exception as e:
        logger.error(f"Error listing payments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/payments/{payment_id}")
@cached(ttl=300, prefix="payments:id")
async def get_payment(
    request: Request,
    payment_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Get payment by ID."""
    try:
        payment = client.get_payment(payment_id)
        return payment
    except Exception as e:
        logger.error(f"Error getting payment: {e}")
        raise HTTPException(status_code=404, detail="Payment not found")


@app.post("/api/v1/payments")
async def create_payment(
    data: PaymentCreate,
    client: HoldedClient = Depends(get_client)
):
    """Create a new payment."""
    try:
        payment_data = {
            "contactId": data.contact_id,
            "amount": data.amount,
        }
        if data.date:
            payment_data["date"] = data.date
        else:
            payment_data["date"] = int(time.time())
        if data.treasury_id:
            payment_data["bankId"] = data.treasury_id
        if data.notes:
            payment_data["notes"] = data.notes
        if data.document_id:
            payment_data["docId"] = data.document_id

        result = client.create_payment(payment_data)

        # Invalidate payments cache
        await invalidate_cache("holded:payments:*")

        return {"success": True, "payment_id": result.get("id"), "payment": result}
    except Exception as e:
        logger.error(f"Error creating payment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/payments/{payment_id}")
async def delete_payment(
    payment_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Delete a payment."""
    try:
        client.delete_payment(payment_id)

        # Invalidate payments cache
        await invalidate_cache("holded:payments:*")

        return {"success": True, "deleted": payment_id}
    except Exception as e:
        logger.error(f"Error deleting payment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TREASURIES (Phase 3)
# ============================================================================

@app.get("/api/v1/treasuries")
@cached(ttl=1800, prefix="treasuries:list")
async def list_treasuries(
    request: Request,
    client: HoldedClient = Depends(get_client)
):
    """List treasury/bank accounts."""
    try:
        treasuries = client.list_treasuries()
        return {"treasuries": treasuries, "count": len(treasuries)}
    except Exception as e:
        logger.error(f"Error listing treasuries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ACCOUNTING (Phase 4)
# ============================================================================

@app.get("/api/v1/accounting/accounts")
@cached(ttl=1800, prefix="accounting:accounts")
async def list_chart_of_accounts(
    request: Request,
    limit: int = Query(500, le=1000),
    client: HoldedClient = Depends(get_client)
):
    """List chart of accounts."""
    try:
        accounts = client.list_chart_of_accounts(limit=limit)
        return {"accounts": accounts, "count": len(accounts)}
    except Exception as e:
        logger.error(f"Error listing accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/accounting/accounts/{account_id}")
@cached(ttl=1800, prefix="accounting:account")
async def get_account(
    request: Request,
    account_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Get account by ID."""
    try:
        account = client.get_account(account_id)
        return account
    except Exception as e:
        logger.error(f"Error getting account: {e}")
        raise HTTPException(status_code=404, detail="Account not found")


@app.get("/api/v1/accounting/accounts/code/{code}")
@cached(ttl=1800, prefix="accounting:account:code")
async def get_account_by_code(
    request: Request,
    code: str,
    client: HoldedClient = Depends(get_client)
):
    """Get account by code (e.g., '4300', '7000')."""
    try:
        account = client.get_account_by_code(code)
        if not account:
            raise HTTPException(status_code=404, detail=f"Account with code {code} not found")
        return account
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account by code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/accounting/ledger")
@cached(ttl=300, prefix="accounting:ledger")
async def list_daily_ledger(
    request: Request,
    account_id: Optional[str] = None,
    date_from: Optional[int] = Query(None, description="Unix timestamp start"),
    date_to: Optional[int] = Query(None, description="Unix timestamp end"),
    limit: int = Query(100, le=500),
    client: HoldedClient = Depends(get_client)
):
    """List daily ledger entries with optional filtering."""
    try:
        entries = client.list_daily_ledger_filtered(
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )
        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        logger.error(f"Error listing ledger: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/accounting/ledger/{entry_id}")
@cached(ttl=300, prefix="accounting:ledger:id")
async def get_ledger_entry(
    request: Request,
    entry_id: str,
    client: HoldedClient = Depends(get_client)
):
    """Get daily ledger entry by ID."""
    try:
        entry = client.get_daily_ledger_entry(entry_id)
        return entry
    except Exception as e:
        logger.error(f"Error getting ledger entry: {e}")
        raise HTTPException(status_code=404, detail="Ledger entry not found")


@app.post("/api/v1/accounting/journal-entry")
async def create_journal_entry(
    data: JournalEntryCreate,
    client: HoldedClient = Depends(get_client)
):
    """Create a new journal entry (daily ledger entry)."""
    try:
        # Build journal entry using JournalEntryBuilder
        builder = JournalEntryBuilder(
            description=data.description,
            reference=data.reference or "",
            date=data.date,
        )

        for line in data.lines:
            if line.debit > 0:
                builder.add_debit(
                    account_id=line.account_id,
                    amount=line.debit,
                    description=line.description or "",
                )
            if line.credit > 0:
                builder.add_credit(
                    account_id=line.account_id,
                    amount=line.credit,
                    description=line.description or "",
                )

        # Validate before creating
        errors = builder.validate()
        if errors:
            raise HTTPException(status_code=400, detail="; ".join(errors))

        result = client.create_journal_entry_with_builder(builder)

        # Invalidate accounting cache
        await invalidate_cache("holded:accounting:*")

        return {"success": True, "entry_id": result.get("id"), "entry": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating journal entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/accounting/trial-balance")
@cached(ttl=300, prefix="accounting:trial-balance")
async def get_trial_balance(
    request: Request,
    date_from: Optional[int] = Query(None, description="Unix timestamp start"),
    date_to: Optional[int] = Query(None, description="Unix timestamp end"),
    client: HoldedClient = Depends(get_client)
):
    """Get trial balance (sum of debits and credits per account).

    Returns accounts with their total debits, credits, and balance.
    """
    try:
        # Get all ledger entries in date range
        entries = client.list_daily_ledger_filtered(
            date_from=date_from,
            date_to=date_to,
            limit=10000
        )

        # Get chart of accounts for names
        accounts_list = client.list_chart_of_accounts(limit=1000)
        account_names = {a.get("id"): {"code": a.get("code"), "name": a.get("name")} for a in accounts_list}

        # Aggregate by account
        account_totals: Dict[str, Dict[str, Any]] = {}

        for entry in entries:
            for line in entry.get("lines", []):
                acc_id = line.get("accountId")
                if not acc_id:
                    continue

                if acc_id not in account_totals:
                    acc_info = account_names.get(acc_id, {})
                    account_totals[acc_id] = {
                        "account_id": acc_id,
                        "code": acc_info.get("code", ""),
                        "name": acc_info.get("name", ""),
                        "total_debit": 0.0,
                        "total_credit": 0.0,
                    }

                account_totals[acc_id]["total_debit"] += float(line.get("debit", 0))
                account_totals[acc_id]["total_credit"] += float(line.get("credit", 0))

        # Calculate balances
        result = []
        total_debits = 0.0
        total_credits = 0.0

        for acc_id, totals in sorted(account_totals.items(), key=lambda x: x[1].get("code", "")):
            balance = totals["total_debit"] - totals["total_credit"]
            result.append({
                **totals,
                "balance": round(balance, 2),
                "total_debit": round(totals["total_debit"], 2),
                "total_credit": round(totals["total_credit"], 2),
            })
            total_debits += totals["total_debit"]
            total_credits += totals["total_credit"]

        return {
            "accounts": result,
            "totals": {
                "debit": round(total_debits, 2),
                "credit": round(total_credits, 2),
                "balance": round(total_debits - total_credits, 2),
            },
            "is_balanced": abs(total_debits - total_credits) < 0.01,
            "count": len(result),
        }
    except Exception as e:
        logger.error(f"Error getting trial balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/accounting/account/{account_id}/ledger")
@cached(ttl=300, prefix="accounting:account:ledger")
async def get_account_ledger(
    request: Request,
    account_id: str,
    date_from: Optional[int] = Query(None, description="Unix timestamp start"),
    date_to: Optional[int] = Query(None, description="Unix timestamp end"),
    limit: int = Query(100, le=500),
    client: HoldedClient = Depends(get_client)
):
    """Get ledger entries for a specific account."""
    try:
        entries = client.list_daily_ledger_filtered(
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )

        # Calculate running balance
        total_debit = 0.0
        total_credit = 0.0

        for entry in entries:
            for line in entry.get("lines", []):
                if line.get("accountId") == account_id:
                    total_debit += float(line.get("debit", 0))
                    total_credit += float(line.get("credit", 0))

        return {
            "account_id": account_id,
            "entries": entries,
            "count": len(entries),
            "totals": {
                "debit": round(total_debit, 2),
                "credit": round(total_credit, 2),
                "balance": round(total_debit - total_credit, 2),
            }
        }
    except Exception as e:
        logger.error(f"Error getting account ledger: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONTABLE AGENT (Phase 5)
# ============================================================================

@app.get("/api/v1/contable/dashboard")
@cached(ttl=300, prefix="contable:dashboard")
async def get_contable_dashboard(
    request: Request,
    client: HoldedClient = Depends(get_client)
):
    """Get comprehensive accounting dashboard for Contable agent.

    Returns summary of financial position, pending documents, overdue invoices,
    reconciliation status, and alerts requiring attention.
    """
    try:
        from holded.accounting_reports import AccountingReporter

        reporter = AccountingReporter(client)

        # Get trial balance summary
        trial_balance = reporter.trial_balance()

        # Get pending invoices (unpaid)
        pending_invoices = client.list_documents("invoice", limit=100)
        pending_count = len([d for d in pending_invoices if not d.get("paid")])
        pending_amount = sum(
            float(d.get("total", 0))
            for d in pending_invoices
            if not d.get("paid")
        )

        # Get pending purchases (unpaid)
        pending_purchases = client.list_documents("purchase", limit=100)
        pending_purchase_count = len([d for d in pending_purchases if not d.get("paid")])
        pending_purchase_amount = sum(
            float(d.get("total", 0))
            for d in pending_purchases
            if not d.get("paid")
        )

        # Get overdue invoices
        now = int(datetime.now().timestamp())
        overdue = [
            d for d in pending_invoices
            if not d.get("paid") and d.get("dueDate", now) < now
        ]
        overdue_amount = sum(float(d.get("total", 0)) for d in overdue)

        # Count by overdue buckets
        buckets = {"0-30": 0, "31-60": 0, "61-90": 0, "90+": 0}
        for doc in overdue:
            due_date = doc.get("dueDate", now)
            days = (now - due_date) // 86400
            if days <= 30:
                buckets["0-30"] += 1
            elif days <= 60:
                buckets["31-60"] += 1
            elif days <= 90:
                buckets["61-90"] += 1
            else:
                buckets["90+"] += 1

        return {
            "summary": {
                "trial_balance_status": "balanced" if trial_balance["is_balanced"] else "unbalanced",
                "total_debits": trial_balance["totals"]["debit"],
                "total_credits": trial_balance["totals"]["credit"],
            },
            "receivables": {
                "pending_invoices": pending_count,
                "pending_amount": round(pending_amount, 2),
                "overdue_count": len(overdue),
                "overdue_amount": round(overdue_amount, 2),
                "overdue_buckets": buckets,
            },
            "payables": {
                "pending_purchases": pending_purchase_count,
                "pending_amount": round(pending_purchase_amount, 2),
            },
            "alerts": {
                "legal_escalation_needed": buckets["90+"],
                "trial_balance_issue": not trial_balance["is_balanced"],
            },
            "updated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting contable dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/contable/reports/income-statement")
@cached(ttl=600, prefix="contable:income-statement")
async def get_income_statement(
    request: Request,
    date_from: str = Query(..., description="Start date (YYYY-MM-DD)"),
    date_to: str = Query(..., description="End date (YYYY-MM-DD)"),
    client: HoldedClient = Depends(get_client)
):
    """Get income statement (P&L) for a period.

    Returns revenue, expenses, and net income using the AccountingReporter.
    """
    try:
        from holded.accounting_reports import AccountingReporter

        from_dt = datetime.strptime(date_from, "%Y-%m-%d")
        to_dt = datetime.strptime(date_to, "%Y-%m-%d")

        if from_dt > to_dt:
            raise HTTPException(status_code=400, detail="date_from must be before date_to")

        reporter = AccountingReporter(client)
        return reporter.income_statement(from_dt, to_dt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}. Use YYYY-MM-DD")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting income statement: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/contable/reports/balance-sheet")
@cached(ttl=600, prefix="contable:balance-sheet")
async def get_balance_sheet(
    request: Request,
    as_of_date: str = Query(..., description="As-of date (YYYY-MM-DD)"),
    client: HoldedClient = Depends(get_client)
):
    """Get balance sheet as of a specific date."""
    try:
        from holded.accounting_reports import AccountingReporter

        date = datetime.strptime(as_of_date, "%Y-%m-%d")
        reporter = AccountingReporter(client)
        return reporter.balance_sheet(date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error getting balance sheet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/contable/classify")
async def classify_expense_endpoint(
    description: str = Query(..., description="Expense description"),
    supplier: str = Query("", description="Supplier name (optional)"),
):
    """Classify an expense to a chart of accounts code.

    Uses keyword matching against Spanish PGC (Plan General Contable).
    Returns account code, name, confidence, and classification method.
    """
    result = classify_expense(description, supplier)
    return {
        "account_code": result.account_code,
        "account_name": result.account_name,
        "confidence": result.confidence,
        "method": result.method,
        "alternative_codes": result.alternative_codes,
    }


@app.get("/api/v1/contable/classify")
async def classify_expense_get(
    description: str = Query(..., description="Expense description"),
    supplier: str = Query("", description="Supplier name (optional)"),
):
    """GET version of expense classification for simple queries."""
    result = classify_expense(description, supplier)
    return {
        "account_code": result.account_code,
        "account_name": result.account_name,
        "confidence": result.confidence,
        "method": result.method,
        "alternative_codes": result.alternative_codes,
    }


@app.get("/api/v1/contable/overdue")
@cached(ttl=300, prefix="contable:overdue")
async def get_overdue_invoices(
    request: Request,
    client: HoldedClient = Depends(get_client)
):
    """Get overdue invoices categorized by age bucket.

    Buckets: 0-30 days, 31-60 days, 61-90 days, 90+ days.
    Includes recommended actions for each bucket.
    """
    try:
        invoices = client.list_documents("invoice", limit=500)
        now = int(datetime.now().timestamp())

        buckets = {
            "0-30": {"count": 0, "amount": 0, "items": [], "action": "Monitor"},
            "31-60": {"count": 0, "amount": 0, "items": [], "action": "Friendly reminder"},
            "61-90": {"count": 0, "amount": 0, "items": [], "action": "Formal notice (Ley 3/2004)"},
            "90+": {"count": 0, "amount": 0, "items": [], "action": "Legal escalation"},
        }

        total_overdue = 0

        for doc in invoices:
            if doc.get("paid"):
                continue

            due_date = doc.get("dueDate", now)
            if due_date >= now:
                continue  # Not overdue

            days_overdue = (now - due_date) // 86400
            amount = float(doc.get("total", 0))
            total_overdue += amount

            item = {
                "id": doc.get("id"),
                "number": doc.get("docNumber"),
                "contact_id": doc.get("contactId"),
                "contact_name": doc.get("contactName", ""),
                "amount": round(amount, 2),
                "due_date": datetime.fromtimestamp(due_date).strftime("%Y-%m-%d"),
                "days_overdue": days_overdue,
            }

            if days_overdue <= 30:
                buckets["0-30"]["count"] += 1
                buckets["0-30"]["amount"] += amount
                buckets["0-30"]["items"].append(item)
            elif days_overdue <= 60:
                buckets["31-60"]["count"] += 1
                buckets["31-60"]["amount"] += amount
                buckets["31-60"]["items"].append(item)
            elif days_overdue <= 90:
                buckets["61-90"]["count"] += 1
                buckets["61-90"]["amount"] += amount
                buckets["61-90"]["items"].append(item)
            else:
                buckets["90+"]["count"] += 1
                buckets["90+"]["amount"] += amount
                buckets["90+"]["items"].append(item)

        # Round amounts
        for bucket in buckets.values():
            bucket["amount"] = round(bucket["amount"], 2)
            # Limit items returned
            bucket["items"] = bucket["items"][:10]

        return {
            "buckets": buckets,
            "total_overdue_amount": round(total_overdue, 2),
            "total_overdue_count": sum(b["count"] for b in buckets.values()),
            "action_required": {
                "friendly_reminder": buckets["31-60"]["count"],
                "formal_notice": buckets["61-90"]["count"],
                "legal_escalation": buckets["90+"]["count"],
            },
            "updated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting overdue invoices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/contable/overdue/{invoice_id}/escalate")
async def create_escalation_draft(
    invoice_id: str,
    to_legal: bool = Query(False, description="Escalate to legal department"),
    client: HoldedClient = Depends(get_client)
):
    """Create escalation draft for an overdue invoice.

    Creates a draft action for human approval before execution.
    Returns draft data with recommended action based on overdue days.
    """
    try:
        # Get invoice details
        doc = client.get_document("invoice", invoice_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Invoice not found")

        now = int(datetime.now().timestamp())
        due_date = doc.get("dueDate", now)
        days_overdue = max(0, (now - due_date) // 86400)

        # Determine escalation level
        if to_legal or days_overdue > 90:
            level = 4
            action = "legal_escalation"
            recommended = "Crear ticket en Zoho Desk para departamento Legal"
        elif days_overdue > 60:
            level = 3
            action = "formal_notice"
            recommended = "Enviar notificación formal citando Ley 3/2004"
        elif days_overdue > 30:
            level = 2
            action = "friendly_reminder"
            recommended = "Enviar recordatorio cordial de pago"
        else:
            level = 1
            action = "monitor"
            recommended = "Monitorear - aún dentro del plazo razonable"

        return {
            "status": "draft_created",
            "draft": {
                "invoice_id": invoice_id,
                "invoice_number": doc.get("docNumber"),
                "contact_id": doc.get("contactId"),
                "contact_name": doc.get("contactName", ""),
                "amount": float(doc.get("total", 0)),
                "days_overdue": days_overdue,
                "escalation_level": level,
                "action": action,
                "recommended": recommended,
            },
            "message": f"Draft escalation created for {doc.get('docNumber')}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating escalation draft: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/contable/anomalies")
@cached(ttl=600, prefix="contable:anomalies")
async def check_anomalies(
    request: Request,
    client: HoldedClient = Depends(get_client)
):
    """Run all anomaly detection checks.

    Detects:
    - Duplicate invoice numbers
    - Unusual amounts (>3 standard deviations)
    - Potential duplicate purchases (same supplier+amount+day)

    Returns anomalies with severity levels.
    """
    try:
        import statistics

        anomalies = []

        # Get recent invoices and purchases
        invoices = client.list_documents("invoice", limit=500)
        purchases = client.list_documents("purchase", limit=500)

        # Check for duplicate invoice numbers
        invoice_numbers = {}
        for doc in invoices:
            num = doc.get("docNumber", "")
            if num in invoice_numbers:
                anomalies.append({
                    "type": "duplicate_number",
                    "severity": "high",
                    "document_id": doc.get("id"),
                    "document_number": num,
                    "description": f"Duplicate invoice number: {num}",
                    "details": {
                        "original_id": invoice_numbers[num],
                        "duplicate_id": doc.get("id"),
                    }
                })
            else:
                invoice_numbers[num] = doc.get("id")

        # Check for unusual amounts in invoices
        amounts = [float(d.get("total", 0)) for d in invoices if d.get("total")]
        if len(amounts) > 10:
            mean = statistics.mean(amounts)
            stdev = statistics.stdev(amounts)
            threshold = mean + (3 * stdev)

            for doc in invoices:
                amount = float(doc.get("total", 0))
                if amount > threshold:
                    anomalies.append({
                        "type": "unusual_amount",
                        "severity": "medium",
                        "document_id": doc.get("id"),
                        "document_number": doc.get("docNumber"),
                        "description": f"Unusually high amount: €{amount:,.2f} (avg: €{mean:,.2f})",
                        "details": {
                            "amount": amount,
                            "average": round(mean, 2),
                            "threshold": round(threshold, 2),
                        }
                    })

        # Check for same-day duplicate purchases
        purchase_keys = {}
        for doc in purchases:
            supplier = doc.get("contactId", "")
            amount = round(float(doc.get("total", 0)), 2)
            date = doc.get("date", 0)
            key = f"{supplier}:{amount}:{date}"

            if key in purchase_keys and supplier:
                anomalies.append({
                    "type": "same_day_duplicate",
                    "severity": "medium",
                    "document_id": doc.get("id"),
                    "document_number": doc.get("docNumber"),
                    "description": f"Possible duplicate: same supplier, amount, and date",
                    "details": {
                        "supplier_id": supplier,
                        "amount": amount,
                        "original_id": purchase_keys[key],
                    }
                })
            else:
                purchase_keys[key] = doc.get("id")

        # Count by severity
        by_severity = {"high": 0, "medium": 0, "low": 0}
        for a in anomalies:
            by_severity[a["severity"]] = by_severity.get(a["severity"], 0) + 1

        return {
            "anomalies": anomalies,
            "total_count": len(anomalies),
            "by_severity": by_severity,
            "by_type": {
                "duplicate_number": len([a for a in anomalies if a["type"] == "duplicate_number"]),
                "unusual_amount": len([a for a in anomalies if a["type"] == "unusual_amount"]),
                "same_day_duplicate": len([a for a in anomalies if a["type"] == "same_day_duplicate"]),
            },
            "checked_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error checking anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/contable/agent/queue")
@cached(ttl=120, prefix="contable:agent:queue")
async def get_agent_queue(
    request: Request,
    client: HoldedClient = Depends(get_client)
):
    """Get Contable agent's action queue.

    Returns pending actions that require human approval:
    - Overdue invoice escalations
    - Anomaly resolutions
    """
    try:
        # Get overdue summary
        invoices = client.list_documents("invoice", limit=500)
        now = int(datetime.now().timestamp())

        overdue_90_plus = []
        for doc in invoices:
            if doc.get("paid"):
                continue
            due_date = doc.get("dueDate", now)
            if due_date >= now:
                continue
            days_overdue = (now - due_date) // 86400
            if days_overdue > 90:
                overdue_90_plus.append({
                    "id": doc.get("id"),
                    "number": doc.get("docNumber"),
                    "contact_name": doc.get("contactName", ""),
                    "amount": float(doc.get("total", 0)),
                    "days_overdue": days_overdue,
                })

        # Get high-severity anomalies (simplified check)
        invoice_numbers = {}
        high_anomalies = []
        for doc in invoices:
            num = doc.get("docNumber", "")
            if num in invoice_numbers:
                high_anomalies.append({
                    "type": "duplicate_number",
                    "document_number": num,
                    "document_id": doc.get("id"),
                })
            else:
                invoice_numbers[num] = doc.get("id")

        return {
            "agent": "contable",
            "pending_approvals": {
                "escalations_needed": len(overdue_90_plus),
                "anomalies_high": len(high_anomalies),
            },
            "total_pending": len(overdue_90_plus) + len(high_anomalies),
            "items": {
                "overdue_90_plus": overdue_90_plus[:10],
                "high_anomalies": high_anomalies[:10],
            },
            "updated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting agent queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/contable/summary")
@cached(ttl=300, prefix="contable:summary")
async def get_daily_summary(
    request: Request,
    client: HoldedClient = Depends(get_client)
):
    """Get daily accounting summary for Secretary dashboard.

    Optimized for inclusion in the Secretary Eisenhower matrix.
    """
    try:
        invoices = client.list_documents("invoice", limit=200)
        purchases = client.list_documents("purchase", limit=200)
        now = int(datetime.now().timestamp())

        # Pending invoices
        pending_invoices = [d for d in invoices if not d.get("paid")]
        pending_amount = sum(float(d.get("total", 0)) for d in pending_invoices)

        # Overdue
        overdue = [d for d in pending_invoices if d.get("dueDate", now) < now]
        overdue_amount = sum(float(d.get("total", 0)) for d in overdue)
        overdue_90_plus = len([
            d for d in overdue
            if (now - d.get("dueDate", now)) // 86400 > 90
        ])

        # Pending purchases
        pending_purchases = [d for d in purchases if not d.get("paid")]
        payables_amount = sum(float(d.get("total", 0)) for d in pending_purchases)

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "receivables": {
                "pending_count": len(pending_invoices),
                "pending_amount": round(pending_amount, 2),
                "overdue_count": len(overdue),
                "overdue_amount": round(overdue_amount, 2),
            },
            "payables": {
                "pending_count": len(pending_purchases),
                "pending_amount": round(payables_amount, 2),
            },
            "alerts": {
                "legal_escalation_needed": overdue_90_plus,
            },
            "cash_position": round(pending_amount - payables_amount, 2),
        }
    except Exception as e:
        logger.error(f"Error getting daily summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONTABLE AI AGENT - AUTONOMOUS EXECUTION ENDPOINTS
# ============================================================================

@app.post("/api/v1/contable/agent/execute", response_model=AgentExecuteResponse)
async def execute_agent_action(
    data: AgentExecuteRequest,
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
    client: HoldedClient = Depends(get_client),
):
    """Execute an action via Contable AI Agent.

    This is the main entry point for autonomous accounting operations.
    Actions are processed through OpenAI GPT-4o for intelligent classification
    and risk assessment.

    Security: Requires X-Service-API-Key and X-Service-Name headers.
    Rate limited per service.

    Risk levels:
    - auto_execute: Amount <€500 AND confidence >95% - executed immediately
    - draft: Medium risk - creates draft for human approval
    - reject: High risk or low confidence - rejected

    Args:
        intent: reconcile, mark_paid, classify_expense, create_escalation, check_invoice
        parameters: Intent-specific parameters
        force_draft: Force draft mode even for low-risk actions

    Returns:
        AgentExecuteResponse with action_id, status, and result (if auto-executed)
    """
    service_name, _ = service_auth
    settings = get_settings()

    if not settings.contable_agent_enabled:
        raise HTTPException(status_code=503, detail="Contable AI Agent is disabled")

    try:
        # Process with AI
        ai_result = await process_with_openai(
            intent=data.intent,
            parameters=data.parameters,
            source_service=service_name,
        )

        # Extract values
        risk_level = ai_result["risk_assessment"]["level"]
        confidence = ai_result.get("confidence", 0.5)
        action_data = ai_result.get("action_data", data.parameters)
        amount = action_data.get("amount") or action_data.get("payment_amount")

        # Force draft if requested
        if data.force_draft and risk_level == "auto_execute":
            risk_level = "draft"
            ai_result["risk_assessment"]["reason"] = "Forzado a borrador por solicitud"

        # Generate action ID
        action_id = str(uuid.uuid4())

        # Get database manager
        db_manager = get_db_manager()

        # Handle based on risk level
        if risk_level == "auto_execute":
            # Execute immediately
            result = await _execute_action(
                client=client,
                action_type=ai_result.get("intent", data.intent),
                action_data=action_data,
            )

            # Log to database if available
            if db_manager:
                async with db_manager.session() as session:
                    action_record = ContableActionQueue(
                        id=uuid.UUID(action_id),
                        action_type=ai_result.get("intent", data.intent),
                        action_data=action_data,
                        intent=data.intent,
                        risk_level=risk_level,
                        confidence_score=Decimal(str(round(confidence, 2))),
                        amount_eur=Decimal(str(round(amount, 2))) if amount else None,
                        ai_explanation=ai_result.get("explanation"),
                        ai_model=ai_result.get("ai_model"),
                        status="executed",
                        source_service=service_name,
                        executed_at=datetime.utcnow(),
                        execution_result=result,
                    )
                    session.add(action_record)
                    await session.commit()

            return AgentExecuteResponse(
                action_id=action_id,
                status="auto_executed",
                risk_level=risk_level,
                confidence=confidence,
                result=result,
                explanation=ai_result.get("explanation", ""),
            )

        elif risk_level == "draft":
            # Create draft for approval
            if db_manager:
                async with db_manager.session() as session:
                    action_record = ContableActionQueue(
                        id=uuid.UUID(action_id),
                        action_type=ai_result.get("intent", data.intent),
                        action_data=action_data,
                        intent=data.intent,
                        risk_level=risk_level,
                        confidence_score=Decimal(str(round(confidence, 2))),
                        amount_eur=Decimal(str(round(amount, 2))) if amount else None,
                        ai_explanation=ai_result.get("explanation"),
                        ai_model=ai_result.get("ai_model"),
                        status="pending",
                        source_service=service_name,
                    )
                    session.add(action_record)
                    await session.commit()

            return AgentExecuteResponse(
                action_id=action_id,
                status="draft_created",
                risk_level=risk_level,
                confidence=confidence,
                result=None,
                explanation=ai_result.get("explanation", "") + f" - {ai_result['risk_assessment']['reason']}",
            )

        else:  # reject
            # Log rejection
            if db_manager:
                async with db_manager.session() as session:
                    action_record = ContableActionQueue(
                        id=uuid.UUID(action_id),
                        action_type=ai_result.get("intent", data.intent),
                        action_data=action_data,
                        intent=data.intent,
                        risk_level=risk_level,
                        confidence_score=Decimal(str(round(confidence, 2))),
                        amount_eur=Decimal(str(round(amount, 2))) if amount else None,
                        ai_explanation=ai_result.get("explanation"),
                        ai_model=ai_result.get("ai_model"),
                        status="rejected",
                        source_service=service_name,
                        review_notes=ai_result["risk_assessment"]["reason"],
                    )
                    session.add(action_record)
                    await session.commit()

            return AgentExecuteResponse(
                action_id=action_id,
                status="rejected",
                risk_level=risk_level,
                confidence=confidence,
                result=None,
                explanation=ai_result["risk_assessment"]["reason"],
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent execute error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _execute_action(
    client: HoldedClient,
    action_type: str,
    action_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute an approved action.

    This is called for auto-executed actions or when a draft is approved.
    """
    settings = get_settings()

    if action_type == "classify_expense":
        # Classification is informational - just return the classification
        return {
            "classified": True,
            "account_code": action_data.get("account_code"),
            "account_name": action_data.get("account_name"),
        }

    elif action_type == "reconcile_payment" or action_type == "reconcile":
        # Mark document as paid
        document_id = action_data.get("document_id")
        amount = action_data.get("payment_amount") or action_data.get("amount")

        if document_id and amount:
            payment_data = {
                "amount": amount,
                "date": int(time.time()),
                "bankId": settings.holded_treasury_id,
            }
            result = client.pay_document(document_id, payment_data, doc_type="invoice")
            # Invalidate cache
            await invalidate_cache("holded:docs:invoice:*")
            return {"reconciled": True, "document_id": document_id, "result": result}
        else:
            return {"reconciled": False, "error": "Missing document_id or amount"}

    elif action_type == "mark_paid":
        document_id = action_data.get("document_id")
        amount = action_data.get("amount")
        doc_type = action_data.get("document_type", "invoice")

        if document_id and amount:
            payment_data = {
                "amount": amount,
                "date": int(time.time()),
                "bankId": settings.holded_treasury_id,
            }
            result = client.pay_document(document_id, payment_data, doc_type=doc_type)
            await invalidate_cache(f"holded:docs:{doc_type}:*")
            return {"marked_paid": True, "document_id": document_id, "result": result}
        else:
            return {"marked_paid": False, "error": "Missing document_id or amount"}

    elif action_type == "create_escalation":
        # Escalation creates a draft response - doesn't auto-execute
        return {
            "escalation_created": True,
            "invoice_id": action_data.get("invoice_id"),
            "escalation_level": action_data.get("escalation_level"),
            "note": "Escalación creada como borrador para revisión",
        }

    else:
        return {"executed": False, "error": f"Unknown action type: {action_type}"}


@app.get("/api/v1/contable/agent/drafts")
async def list_draft_actions(
    status: str = Query("pending", description="Filter by status: pending, approved, rejected, executed"),
    limit: int = Query(50, le=200),
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """List draft actions waiting for approval.

    Security: Requires service authentication.
    """
    db_manager = get_db_manager()
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.session() as session:
            from sqlalchemy import select

            query = select(ContableActionQueue).where(
                ContableActionQueue.status == status
            ).order_by(
                ContableActionQueue.created_at.desc()
            ).limit(limit)

            result = await session.execute(query)
            actions = result.scalars().all()

            return {
                "drafts": [
                    DraftAction(
                        id=str(a.id),
                        action_type=a.action_type,
                        action_data=a.action_data,
                        risk_level=a.risk_level,
                        confidence=float(a.confidence_score) if a.confidence_score else 0,
                        amount_eur=float(a.amount_eur) if a.amount_eur else None,
                        ai_explanation=a.ai_explanation or "",
                        source_service=a.source_service,
                        created_at=a.created_at.isoformat(),
                        status=a.status,
                    ).model_dump()
                    for a in actions
                ],
                "count": len(actions),
                "status_filter": status,
            }
    except Exception as e:
        logger.error(f"Error listing drafts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/contable/agent/drafts/{draft_id}/approve")
async def approve_draft_action(
    draft_id: str,
    data: DraftApproveRequest,
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
    client: HoldedClient = Depends(get_client),
):
    """Approve a draft action and execute it.

    Security: Requires service authentication.
    """
    db_manager = get_db_manager()
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.session() as session:
            from sqlalchemy import select

            # Get the draft
            result = await session.execute(
                select(ContableActionQueue).where(
                    ContableActionQueue.id == uuid.UUID(draft_id)
                )
            )
            action = result.scalar_one_or_none()

            if not action:
                raise HTTPException(status_code=404, detail="Draft not found")

            if action.status != "pending":
                raise HTTPException(status_code=400, detail=f"Draft is not pending (status: {action.status})")

            # Execute the action
            exec_result = await _execute_action(
                client=client,
                action_type=action.action_type,
                action_data=action.action_data,
            )

            # Update the record
            action.status = "executed"
            action.reviewed_by = data.reviewed_by
            action.reviewed_at = datetime.utcnow()
            action.review_notes = data.notes
            action.executed_at = datetime.utcnow()
            action.execution_result = exec_result

            await session.commit()

            return {
                "success": True,
                "draft_id": draft_id,
                "status": "executed",
                "result": exec_result,
                "reviewed_by": data.reviewed_by,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving draft: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/contable/agent/drafts/{draft_id}/reject")
async def reject_draft_action(
    draft_id: str,
    data: DraftRejectRequest,
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """Reject a draft action.

    Security: Requires service authentication.
    """
    db_manager = get_db_manager()
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(ContableActionQueue).where(
                    ContableActionQueue.id == uuid.UUID(draft_id)
                )
            )
            action = result.scalar_one_or_none()

            if not action:
                raise HTTPException(status_code=404, detail="Draft not found")

            if action.status != "pending":
                raise HTTPException(status_code=400, detail=f"Draft is not pending (status: {action.status})")

            # Update the record
            action.status = "rejected"
            action.reviewed_by = data.reviewed_by
            action.reviewed_at = datetime.utcnow()
            action.review_notes = data.reason

            await session.commit()

            return {
                "success": True,
                "draft_id": draft_id,
                "status": "rejected",
                "reviewed_by": data.reviewed_by,
                "reason": data.reason,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting draft: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/contable/agent/stats", response_model=AgentStatsResponse)
async def get_agent_stats(
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """Get Contable AI Agent statistics.

    Security: Requires service authentication.
    """
    db_manager = get_db_manager()
    if not db_manager:
        # Return empty stats if no database
        return AgentStatsResponse(
            total_actions=0,
            actions_by_status={},
            actions_by_type={},
            auto_executed_today=0,
            drafts_pending=0,
            avg_confidence=0,
            total_amount_processed=0,
        )

    try:
        async with db_manager.session() as session:
            from sqlalchemy import select, func

            # Total actions
            total_result = await session.execute(
                select(func.count(ContableActionQueue.id))
            )
            total_actions = total_result.scalar() or 0

            # Actions by status
            status_result = await session.execute(
                select(
                    ContableActionQueue.status,
                    func.count(ContableActionQueue.id)
                ).group_by(ContableActionQueue.status)
            )
            actions_by_status = {row[0]: row[1] for row in status_result.fetchall()}

            # Actions by type
            type_result = await session.execute(
                select(
                    ContableActionQueue.action_type,
                    func.count(ContableActionQueue.id)
                ).group_by(ContableActionQueue.action_type)
            )
            actions_by_type = {row[0]: row[1] for row in type_result.fetchall()}

            # Auto-executed today
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            auto_today_result = await session.execute(
                select(func.count(ContableActionQueue.id)).where(
                    ContableActionQueue.status == "executed",
                    ContableActionQueue.risk_level == "auto_execute",
                    ContableActionQueue.executed_at >= today_start,
                )
            )
            auto_executed_today = auto_today_result.scalar() or 0

            # Pending drafts
            pending_result = await session.execute(
                select(func.count(ContableActionQueue.id)).where(
                    ContableActionQueue.status == "pending"
                )
            )
            drafts_pending = pending_result.scalar() or 0

            # Average confidence
            avg_conf_result = await session.execute(
                select(func.avg(ContableActionQueue.confidence_score)).where(
                    ContableActionQueue.confidence_score.isnot(None)
                )
            )
            avg_confidence = float(avg_conf_result.scalar() or 0)

            # Total amount processed
            amount_result = await session.execute(
                select(func.sum(ContableActionQueue.amount_eur)).where(
                    ContableActionQueue.status == "executed"
                )
            )
            total_amount = float(amount_result.scalar() or 0)

            return AgentStatsResponse(
                total_actions=total_actions,
                actions_by_status=actions_by_status,
                actions_by_type=actions_by_type,
                auto_executed_today=auto_executed_today,
                drafts_pending=drafts_pending,
                avg_confidence=round(avg_confidence, 2),
                total_amount_processed=round(total_amount, 2),
            )

    except Exception as e:
        logger.error(f"Error getting agent stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/contable/scheduler/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status():
    """Get background job scheduler status.

    No authentication required - read-only status endpoint.
    """
    settings = get_settings()

    if not SCHEDULER_AVAILABLE:
        return SchedulerStatusResponse(
            enabled=False,
            running=False,
            jobs=[],
            next_runs={},
        )

    if not _scheduler:
        return SchedulerStatusResponse(
            enabled=settings.scheduler_enabled,
            running=False,
            jobs=[],
            next_runs={},
        )

    jobs = []
    next_runs = {}

    for job in _scheduler.get_jobs():
        next_run = job.next_run_time.isoformat() if job.next_run_time else None
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": next_run,
        })
        next_runs[job.id] = next_run

    return SchedulerStatusResponse(
        enabled=settings.scheduler_enabled,
        running=_scheduler.running,
        jobs=jobs,
        next_runs=next_runs,
    )


@app.post("/api/v1/contable/jobs/reconcile")
async def trigger_reconciliation_job(
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """Manually trigger the reconciliation job.

    Security: Requires service authentication.
    """
    try:
        await run_daily_reconciliation()
        return {"success": True, "message": "Reconciliation job completed"}
    except Exception as e:
        logger.error(f"Manual reconciliation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/contable/jobs/anomalies")
async def trigger_anomaly_job(
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """Manually trigger the anomaly detection job.

    Security: Requires service authentication.
    """
    try:
        await run_anomaly_check()
        return {"success": True, "message": "Anomaly check job completed"}
    except Exception as e:
        logger.error(f"Manual anomaly check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/contable/jobs/snapshot")
async def trigger_snapshot_job(
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """Manually trigger the daily snapshot job.

    Security: Requires service authentication.
    """
    try:
        await run_daily_snapshot()
        return {"success": True, "message": "Snapshot job completed"}
    except Exception as e:
        logger.error(f"Manual snapshot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/contable/snapshots")
async def list_snapshots(
    days: int = Query(30, le=365),
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """List daily financial snapshots for trend analysis.

    Security: Requires service authentication.
    """
    db_manager = get_db_manager()
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.session() as session:
            from sqlalchemy import select
            from datetime import timedelta

            cutoff_date = date.today() - timedelta(days=days)

            result = await session.execute(
                select(ContableDailySnapshot).where(
                    ContableDailySnapshot.snapshot_date >= cutoff_date
                ).order_by(ContableDailySnapshot.snapshot_date.desc())
            )
            snapshots = result.scalars().all()

            return {
                "snapshots": [
                    {
                        "date": s.snapshot_date.isoformat(),
                        "receivables": {
                            "total": float(s.total_receivables or 0),
                            "current": float(s.receivables_current or 0),
                            "overdue_30": float(s.receivables_overdue_30 or 0),
                            "overdue_60": float(s.receivables_overdue_60 or 0),
                            "overdue_90": float(s.receivables_overdue_90 or 0),
                            "overdue_90_plus": float(s.receivables_overdue_90_plus or 0),
                        },
                        "payables": {
                            "total": float(s.total_payables or 0),
                            "current": float(s.payables_current or 0),
                            "overdue": float(s.payables_overdue or 0),
                        },
                        "bank_balance": float(s.bank_balance or 0) if s.bank_balance else None,
                        "agent_activity": {
                            "auto_executed": s.actions_auto_executed or 0,
                            "drafted": s.actions_drafted or 0,
                            "approved": s.actions_approved or 0,
                            "rejected": s.actions_rejected or 0,
                            "reconciliations": s.reconciliations_completed or 0,
                        },
                        "anomalies": {
                            "detected": s.anomalies_detected or 0,
                            "resolved": s.anomalies_resolved or 0,
                        },
                    }
                    for s in snapshots
                ],
                "count": len(snapshots),
                "days_requested": days,
            }

    except Exception as e:
        logger.error(f"Error listing snapshots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# QONTO BANKING ENDPOINTS
# ============================================================================

@app.get("/api/v1/qonto/balance")
async def get_qonto_balance(
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """Get Qonto account balance.

    Security: Requires service authentication.
    """
    qonto = get_qonto_client()
    if not qonto:
        raise HTTPException(status_code=503, detail="Qonto client not configured")

    try:
        balance = qonto.get_balance()
        return balance
    except Exception as e:
        logger.error(f"Error getting Qonto balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/qonto/transactions")
async def list_qonto_transactions(
    side: Optional[str] = Query(None, description="Filter: credit (incoming) or debit (outgoing)"),
    days: int = Query(30, le=365),
    limit: int = Query(100, le=500),
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """List Qonto transactions.

    Security: Requires service authentication.
    """
    qonto = get_qonto_client()
    if not qonto:
        raise HTTPException(status_code=503, detail="Qonto client not configured")

    try:
        transactions = qonto.list_transactions(side=side, days=days, per_page=limit)
        return {
            "transactions": [t.to_dict() for t in transactions],
            "count": len(transactions),
            "filters": {
                "side": side,
                "days": days,
            },
        }
    except Exception as e:
        logger.error(f"Error listing Qonto transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/qonto/credits")
async def list_qonto_credits(
    days: int = Query(30, le=365),
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """List incoming payments (credits) from Qonto.

    Security: Requires service authentication.
    """
    qonto = get_qonto_client()
    if not qonto:
        raise HTTPException(status_code=503, detail="Qonto client not configured")

    try:
        credits = qonto.list_credits(days=days)
        return {
            "credits": [t.to_dict() for t in credits],
            "count": len(credits),
            "total_amount": sum(float(t.amount) for t in credits),
            "days": days,
        }
    except Exception as e:
        logger.error(f"Error listing Qonto credits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/qonto/search")
async def search_qonto_transactions(
    query: str = Query(..., description="Search query (label, reference, or note)"),
    days: int = Query(90, le=365),
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """Search Qonto transactions by label or reference.

    Security: Requires service authentication.
    """
    qonto = get_qonto_client()
    if not qonto:
        raise HTTPException(status_code=503, detail="Qonto client not configured")

    try:
        matches = qonto.search_transactions(query=query, days=days)
        return {
            "matches": [t.to_dict() for t in matches],
            "count": len(matches),
            "query": query,
            "days": days,
        }
    except Exception as e:
        logger.error(f"Error searching Qonto transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/qonto/find-payment")
async def find_payment_for_invoice(
    invoice_number: str = Query(..., description="Invoice number to match (e.g., FA-2026-0123)"),
    amount: Optional[float] = Query(None, description="Expected amount (for better matching)"),
    days: int = Query(90, le=365),
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """Find potential payment matches for an invoice.

    Searches Qonto credits for transactions matching the invoice number
    and/or amount.

    Security: Requires service authentication.
    """
    qonto = get_qonto_client()
    if not qonto:
        raise HTTPException(status_code=503, detail="Qonto client not configured")

    try:
        matches = qonto.find_payment_for_invoice(
            invoice_number=invoice_number,
            amount=amount,
            days=days,
        )
        return {
            "matches": [t.to_dict() for t in matches],
            "count": len(matches),
            "invoice_number": invoice_number,
            "expected_amount": amount,
            "best_match": matches[0].to_dict() if matches else None,
        }
    except Exception as e:
        logger.error(f"Error finding payment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/qonto/reconciliation/pending")
async def list_pending_reconciliation_matches(
    limit: int = Query(50, le=200),
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """List pending reconciliation matches awaiting approval.

    Security: Requires service authentication.
    """
    db_manager = get_db_manager()
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(ContableReconciliationMatch).where(
                    ContableReconciliationMatch.status == "pending"
                ).order_by(
                    ContableReconciliationMatch.match_confidence.desc()
                ).limit(limit)
            )
            matches = result.scalars().all()

            return {
                "matches": [
                    {
                        "id": str(m.id),
                        "qonto": {
                            "transaction_id": m.qonto_transaction_id,
                            "amount": float(m.qonto_amount),
                            "date": m.qonto_date.isoformat() if m.qonto_date else None,
                            "counterparty": m.qonto_counterparty,
                            "reference": m.qonto_reference,
                        },
                        "holded": {
                            "document_id": m.holded_document_id,
                            "document_type": m.holded_document_type,
                            "document_number": m.holded_document_number,
                            "amount": float(m.holded_amount) if m.holded_amount else None,
                            "contact_name": m.holded_contact_name,
                        },
                        "match": {
                            "confidence": float(m.match_confidence),
                            "type": m.match_type,
                            "details": m.match_details,
                        },
                        "status": m.status,
                        "created_at": m.created_at.isoformat(),
                    }
                    for m in matches
                ],
                "count": len(matches),
            }
    except Exception as e:
        logger.error(f"Error listing pending matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/qonto/reconciliation/{match_id}/approve")
async def approve_reconciliation_match(
    match_id: str,
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
    client: HoldedClient = Depends(get_client),
):
    """Approve a reconciliation match and mark the document as paid.

    This will:
    1. Mark the Holded document as paid
    2. Update the reconciliation match status

    Security: Requires service authentication.
    """
    db_manager = get_db_manager()
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")

    settings = get_settings()

    try:
        async with db_manager.session() as session:
            from sqlalchemy import select

            # Get the match
            result = await session.execute(
                select(ContableReconciliationMatch).where(
                    ContableReconciliationMatch.id == uuid.UUID(match_id)
                )
            )
            match = result.scalar_one_or_none()

            if not match:
                raise HTTPException(status_code=404, detail="Match not found")

            if match.status != "pending":
                raise HTTPException(
                    status_code=400,
                    detail=f"Match is not pending (status: {match.status})"
                )

            # Mark document as paid in Holded
            payment_data = {
                "amount": float(match.qonto_amount),
                "date": int(match.qonto_date.timestamp()) if match.qonto_date else int(time.time()),
                "bankId": settings.holded_treasury_id,
            }

            try:
                result = client.pay_document(
                    match.holded_document_id,
                    payment_data,
                    doc_type=match.holded_document_type,
                )
            except Exception as e:
                logger.error(f"Failed to mark document as paid: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to mark as paid: {e}")

            # Update match status
            match.status = "executed"
            match.reconciled_at = datetime.utcnow()

            await session.commit()

            # Invalidate cache
            await invalidate_cache(f"holded:docs:{match.holded_document_type}:*")

            return {
                "success": True,
                "match_id": match_id,
                "document_id": match.holded_document_id,
                "document_number": match.holded_document_number,
                "amount": float(match.qonto_amount),
                "status": "executed",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving match: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/qonto/reconciliation/{match_id}/reject")
async def reject_reconciliation_match(
    match_id: str,
    reason: str = Query(..., description="Reason for rejection"),
    service_auth: Tuple[str, str] = Depends(verify_service_auth),
):
    """Reject a reconciliation match.

    Security: Requires service authentication.
    """
    db_manager = get_db_manager()
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_manager.session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(ContableReconciliationMatch).where(
                    ContableReconciliationMatch.id == uuid.UUID(match_id)
                )
            )
            match = result.scalar_one_or_none()

            if not match:
                raise HTTPException(status_code=404, detail="Match not found")

            if match.status != "pending":
                raise HTTPException(
                    status_code=400,
                    detail=f"Match is not pending (status: {match.status})"
                )

            # Update match status
            match.status = "rejected"
            match.match_details = {
                **(match.match_details or {}),
                "rejection_reason": reason,
                "rejected_at": datetime.utcnow().isoformat(),
            }

            await session.commit()

            return {
                "success": True,
                "match_id": match_id,
                "status": "rejected",
                "reason": reason,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting match: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
