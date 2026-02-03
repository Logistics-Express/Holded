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
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("holded-api")

# Global client instance
_client: Optional[HoldedClient] = None


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
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

    yield

    # Cleanup
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
