"""
Holded API Service
Internal microservice for Holded ERP operations.

Endpoints:
- /health - Health check
- /api/v1/contacts - Contact management
- /api/v1/documents - Document operations (invoices, estimates, proformas)
- /api/v1/debt - Debt/outstanding invoice checking
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add lib directory to path for HoldedClient
# Supports both local development (claude-tools) and production (bundled)
LIB_PATHS = [
    Path(__file__).parent / "lib",  # Bundled in repo
    Path("/app/lib"),  # Docker container
    Path.home() / "Desarrollos/claude-tools/lib",  # Local development
]

# Debug: Print directory contents at startup
print(f"DEBUG: __file__ = {__file__}")
print(f"DEBUG: parent = {Path(__file__).parent}")
print(f"DEBUG: /app contents = {list(Path('/app').iterdir()) if Path('/app').exists() else 'NOT FOUND'}")
print(f"DEBUG: /app/lib exists = {Path('/app/lib').exists()}")
if Path('/app/lib').exists():
    print(f"DEBUG: /app/lib contents = {list(Path('/app/lib').iterdir())}")

lib_loaded = False
for lib_path in LIB_PATHS:
    print(f"DEBUG: Checking {lib_path}, exists={lib_path.exists()}")
    if lib_path.exists() and str(lib_path) not in sys.path:
        sys.path.insert(0, str(lib_path))
        lib_loaded = True
        print(f"DEBUG: Loaded lib from {lib_path}")
        break

if not lib_loaded:
    raise RuntimeError(f"Could not find holded library in any of: {LIB_PATHS}")

from holded.holded_client import HoldedClient, DocumentBuilder

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
        api_key = os.getenv("HOLDED_API_KEY")
        if api_key:
            _client = HoldedClient(api_key)
        else:
            _client = HoldedClient.from_credentials()
    return _client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Holded API service starting...")
    yield
    logger.info("Holded API service shutting down...")


app = FastAPI(
    title="Holded API Service",
    description="Internal microservice for Holded ERP operations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for internal services
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Internal service
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ============================================================================
# HEALTH
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "holded-api"}


# ============================================================================
# CONTACTS
# ============================================================================

@app.get("/api/v1/contacts")
async def list_contacts(
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
async def get_contact(
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
async def search_contacts(
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
        return {"success": True, "contact_id": result.get("id"), "contact": result}
    except Exception as e:
        logger.error(f"Error creating contact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DOCUMENTS
# ============================================================================

@app.get("/api/v1/documents/invoices")
async def list_invoices(
    limit: int = Query(50, le=500),
    contact_id: Optional[str] = None,
    client: HoldedClient = Depends(get_client)
):
    """List invoices."""
    try:
        if contact_id:
            invoices = client.list_documents_filtered(
                doc_type="invoice",
                contact_id=contact_id,
                limit=limit
            )
        else:
            invoices = client.list_documents(doc_type="invoice", limit=limit)
        return {"invoices": invoices, "count": len(invoices)}
    except Exception as e:
        logger.error(f"Error listing invoices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/estimates")
async def list_estimates(
    limit: int = Query(50, le=500),
    contact_id: Optional[str] = None,
    client: HoldedClient = Depends(get_client)
):
    """List estimates/quotations."""
    try:
        if contact_id:
            estimates = client.list_documents_filtered(
                doc_type="estimate",
                contact_id=contact_id,
                limit=limit
            )
        else:
            estimates = client.list_documents(doc_type="estimate", limit=limit)
        return {"estimates": estimates, "count": len(estimates)}
    except Exception as e:
        logger.error(f"Error listing estimates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents/{doc_type}/{document_id}")
async def get_document(
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
async def lookup_invoice(
    data: InvoiceLookupRequest,
    client: HoldedClient = Depends(get_client)
):
    """Look up invoice by number."""
    try:
        # Normalize invoice number
        invoice_number = data.invoice_number.upper().strip()

        # Remove common prefixes if partial
        for prefix in ["FA-", "FAC-", "F-"]:
            if invoice_number.startswith(prefix):
                invoice_number = invoice_number[len(prefix):]

        # Search invoices
        invoices = client.list_documents(doc_type="invoice", limit=500)

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
async def get_invoice_by_number(
    invoice_number: str,
    client: HoldedClient = Depends(get_client)
):
    """Get invoice by number (convenience endpoint)."""
    from pydantic import BaseModel as BM

    class Req(BM):
        invoice_number: str

    return await lookup_invoice(Req(invoice_number=invoice_number), client)


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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
