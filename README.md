# Holded API Service

Internal microservice providing a REST API for [Holded ERP](https://www.holded.com/) operations with Redis caching and PostgreSQL audit logging.

**Version:** 2.0.0

## Features

- **45 REST endpoints** covering contacts, documents, products, services, payments, and accounting
- **Redis caching** with configurable TTLs and pattern-based invalidation
- **PostgreSQL audit logging** for all requests
- **Document workflows** - Clone estimates to invoices, download PDFs, mark as paid
- **Accounting module** - Trial balance, journal entries, chart of accounts
- **Rate limiting** - Automatic retry with exponential backoff

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/Logistics-Express/Holded.git
cd Holded

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Holded API key

# Run server
uvicorn main:app --reload --port 8001
```

### Docker

```bash
docker build -t holded-api .
docker run -p 8001:8001 --env-file .env holded-api
```

### Railway Deployment

The service is configured for Railway deployment. Connect your GitHub repo and Railway will auto-deploy on push.

Required Railway services:
- **PostgreSQL** - For audit logging
- **Redis** - For caching

Environment variables are automatically linked via `${{Postgres.DATABASE_URL}}` and `${{Redis.REDIS_URL}}`.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOLDED_API_KEY` | Holded API key (Settings > Developers) | Required |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://localhost/holded_audit` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `CACHE_ENABLED` | Enable Redis caching | `true` |
| `AUDIT_ENABLED` | Enable request audit logging | `true` |
| `PORT` | Server port | `8001` |

### Cache TTLs

| Resource | TTL | Invalidation |
|----------|-----|--------------|
| Contacts | 5-10 min | On create/update/delete |
| Documents | 10-15 min | On create/update/delete/pay |
| Products | 15 min | On create/update/delete |
| Services | 15 min | On create/update/delete |
| Payments | 5 min | On create/delete |
| Treasuries | 30 min | Rarely changes |
| Accounting | 5-30 min | On journal entry create |

---

## API Reference

### Health Check

```
GET /health
```

Returns service status including Redis and PostgreSQL health.

---

### Contacts

#### List Contacts
```
GET /api/v1/contacts?limit=50
```

#### Get Contact
```
GET /api/v1/contacts/{contact_id}
```

#### Search Contacts
```
GET /api/v1/contacts/search/{query}
```
Searches by name, email, or NIF.

#### Create Contact
```
POST /api/v1/contacts
Content-Type: application/json

{
  "name": "Acme Corp",
  "email": "contact@acme.com",
  "phone": "+34612345678",
  "nif": "B12345678",
  "address": "C/ Principal 123"
}
```

#### Update Contact
```
PUT /api/v1/contacts/{contact_id}
Content-Type: application/json

{
  "name": "Acme Corporation",
  "email": "new@acme.com",
  "phone": "+34612345678",
  "nif": "B12345678",
  "address": "C/ Nueva 456",
  "city": "Madrid",
  "province": "Madrid",
  "postal_code": "28001",
  "country": "ES"
}
```

#### Delete Contact
```
DELETE /api/v1/contacts/{contact_id}
```

---

### Documents

Supported document types: `invoice`, `estimate`, `proform`, `salesorder`, `creditnote`, `salesreceipt`, `waybill`, `purchase`, `purchaseorder`, `purchaserefund`

#### List Invoices
```
GET /api/v1/documents/invoices?limit=50&year=2024&contact_id=xxx
```

#### List Estimates
```
GET /api/v1/documents/estimates?limit=50&year=2024
```

#### List Documents by Year
```
GET /api/v1/documents/{doc_type}/year/{year}?limit=500
```

#### Get Document
```
GET /api/v1/documents/{doc_type}/{document_id}
```

#### Download PDF
```
GET /api/v1/documents/{doc_type}/{document_id}/pdf
```
Returns binary PDF content with `Content-Type: application/pdf`.

#### Send Document by Email
```
POST /api/v1/documents/{doc_type}/{document_id}/send
Content-Type: application/json

{
  "emails": ["client@example.com"]  // Optional, uses contact email if omitted
}
```

#### Clone Document
```
POST /api/v1/documents/{document_id}/clone
Content-Type: application/json

{
  "target_type": "invoice",      // Required: invoice, salesorder, proform, etc.
  "new_contact_id": "xxx",       // Optional: override contact
  "new_date": 1704067200         // Optional: Unix timestamp, defaults to now
}
```

**Common use case:** Convert estimate to invoice:
```bash
curl -X POST https://api/documents/abc123/clone \
  -H "Content-Type: application/json" \
  -d '{"target_type": "invoice"}'
```

#### Mark Document as Paid
```
POST /api/v1/documents/{doc_type}/{document_id}/pay
Content-Type: application/json

{
  "amount": 121.00,
  "date": 1704067200,           // Optional: Unix timestamp
  "treasury_id": "xxx"          // Optional: bank account ID
}
```

#### Update Document
```
PUT /api/v1/documents/{doc_type}/{document_id}
Content-Type: application/json

{
  "notes": "Updated notes",
  "due_date": 1706745600,
  "contact_id": "xxx"
}
```

#### Delete Document
```
DELETE /api/v1/documents/{doc_type}/{document_id}
```

---

### Estimates (Shortcuts)

#### Create Shipping Estimate
```
POST /api/v1/estimates/shipping
Content-Type: application/json

{
  "contact_name": "Acme Corp",
  "contact_email": "logistics@acme.com",
  "destination": "Melilla",
  "pallets": 2,
  "price": 250.00,
  "origin": "Málaga",
  "weight_kg": 450,
  "notes": "Fragile cargo",
  "send_email": true
}
```

#### Create Generic Estimate
```
POST /api/v1/estimates
Content-Type: application/json

{
  "contact_name": "Acme Corp",
  "contact_email": "billing@acme.com",
  "items": [
    {"description": "Consulting", "quantity": 8, "price": 75, "tax": 21},
    {"description": "Travel expenses", "quantity": 1, "price": 150, "tax": 21}
  ],
  "notes": "Payment due in 30 days",
  "send_email": false
}
```

---

### Invoice Lookup

#### Lookup by Number
```
POST /api/v1/invoices/lookup
Content-Type: application/json

{
  "invoice_number": "FA-2024-001"
}
```

#### Get by Number (Shortcut)
```
GET /api/v1/invoices/{invoice_number}?year=2024
```

---

### Debt Checking

```
POST /api/v1/debt/check
Content-Type: application/json

{
  "contact_name": "Acme Corp",  // OR
  "contact_id": "xxx",          // OR
  "nif": "B12345678"
}
```

Response:
```json
{
  "found": true,
  "contact_id": "xxx",
  "has_debt": true,
  "total_outstanding": 1250.50,
  "pending_count": 3,
  "overdue_count": 1,
  "oldest_overdue_days": 45,
  "documents": [...]
}
```

---

### Products

#### List Products
```
GET /api/v1/products?limit=50
```

#### Get Product
```
GET /api/v1/products/{product_id}
```

#### Create Product
```
POST /api/v1/products
Content-Type: application/json

{
  "name": "Europallet",
  "price": 25.00,
  "cost_price": 18.00,
  "tax": 21,
  "sku": "PAL-001",
  "desc": "Standard europallet 120x80cm",
  "stock": 100
}
```

#### Update Product
```
PUT /api/v1/products/{product_id}
Content-Type: application/json

{
  "price": 27.00,
  "stock": 85
}
```

#### Delete Product
```
DELETE /api/v1/products/{product_id}
```

---

### Services

#### List Services
```
GET /api/v1/services?limit=50
```

#### Get Service
```
GET /api/v1/services/{service_id}
```

#### Create Service
```
POST /api/v1/services
Content-Type: application/json

{
  "name": "Logistics Consulting",
  "price": 75.00,
  "cost_price": 45.00,
  "tax": 21,
  "desc": "Per hour rate"
}
```

#### Update Service
```
PUT /api/v1/services/{service_id}
Content-Type: application/json

{
  "price": 80.00
}
```

#### Delete Service
```
DELETE /api/v1/services/{service_id}
```

---

### Payments

#### List Payments
```
GET /api/v1/payments?limit=50
```

#### Get Payment
```
GET /api/v1/payments/{payment_id}
```

#### Create Payment
```
POST /api/v1/payments
Content-Type: application/json

{
  "contact_id": "xxx",
  "amount": 500.00,
  "date": 1704067200,
  "treasury_id": "xxx",
  "notes": "Bank transfer",
  "document_id": "xxx"
}
```

#### Delete Payment
```
DELETE /api/v1/payments/{payment_id}
```

---

### Treasuries (Bank Accounts)

#### List Treasuries
```
GET /api/v1/treasuries
```

---

### Accounting

#### Chart of Accounts
```
GET /api/v1/accounting/accounts?limit=500
```

#### Get Account by ID
```
GET /api/v1/accounting/accounts/{account_id}
```

#### Get Account by Code
```
GET /api/v1/accounting/accounts/code/{code}
```
Example: `/api/v1/accounting/accounts/code/4300` for receivables.

#### Daily Ledger
```
GET /api/v1/accounting/ledger?account_id=xxx&date_from=1704067200&date_to=1706745600&limit=100
```

#### Get Ledger Entry
```
GET /api/v1/accounting/ledger/{entry_id}
```

#### Create Journal Entry
```
POST /api/v1/accounting/journal-entry
Content-Type: application/json

{
  "description": "Invoice F-2024-001 payment received",
  "reference": "BANK-001",
  "date": 1704067200,
  "lines": [
    {"account_id": "bank_account_id", "debit": 1210.00, "credit": 0, "description": "Bank deposit"},
    {"account_id": "receivable_id", "debit": 0, "credit": 1210.00, "description": "Customer payment"}
  ]
}
```

**Note:** Journal entries must be balanced (total debits = total credits).

#### Trial Balance
```
GET /api/v1/accounting/trial-balance?date_from=1704067200&date_to=1706745600
```

Response:
```json
{
  "accounts": [
    {
      "account_id": "xxx",
      "code": "4300",
      "name": "Clientes",
      "total_debit": 50000.00,
      "total_credit": 45000.00,
      "balance": 5000.00
    }
  ],
  "totals": {
    "debit": 150000.00,
    "credit": 150000.00,
    "balance": 0.00
  },
  "is_balanced": true,
  "count": 25
}
```

#### Account Ledger
```
GET /api/v1/accounting/account/{account_id}/ledger?date_from=xxx&date_to=xxx&limit=100
```

---

## Error Handling

All endpoints return standard HTTP status codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (validation error) |
| 404 | Resource not found |
| 500 | Internal server error |

Error response format:
```json
{
  "detail": "Error message description"
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Service                         │
├─────────────────────────────────────────────────────────────┤
│  Middleware: CORS, Audit Logging                            │
├─────────────────────────────────────────────────────────────┤
│  Endpoints: /contacts, /documents, /products, /accounting   │
├─────────────────────────────────────────────────────────────┤
│  Cache Layer: Redis with TTL & Pattern Invalidation         │
├─────────────────────────────────────────────────────────────┤
│  HoldedClient: Retry logic, Rate limiting, Pagination       │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │  Holded  │        │  Redis   │        │ Postgres │
   │   API    │        │  Cache   │        │  Audit   │
   └──────────┘        └──────────┘        └──────────┘
```

### File Structure

```
holded-api/
├── main.py                 # FastAPI application & endpoints
├── app/
│   ├── config.py           # Settings from environment
│   ├── cache/
│   │   ├── decorators.py   # @cached decorator
│   │   └── redis_client.py # Async Redis client
│   ├── db/
│   │   ├── database.py     # Async PostgreSQL manager
│   │   └── models.py       # SQLAlchemy models (ApiAuditLog)
│   └── middleware/
│       └── audit.py        # Request audit middleware
├── lib/
│   └── holded/
│       └── holded_client.py # Holded API client library
├── requirements.txt
├── Dockerfile
├── railway.toml
└── .env.example
```

---

## HoldedClient Library

The `lib/holded/holded_client.py` provides a full-featured client for the Holded API:

### Features
- **Automatic retry** with exponential backoff for 429/5xx errors
- **Pagination** support for all list endpoints
- **DocumentBuilder** for creating complex invoices with margin tracking
- **JournalEntryBuilder** for double-entry accounting
- **TemplateManager** for reusable document templates

### Direct Usage

```python
from holded.holded_client import HoldedClient, DocumentBuilder

# Initialize
client = HoldedClient.from_credentials()  # Uses ~/.holded_credentials.json
# OR
client = HoldedClient("your-api-key")

# List contacts
contacts = client.list_contacts(limit=100)

# Create invoice with builder
builder = DocumentBuilder(contact_id="abc123", doc_type="invoice")
builder.add_item("Service", units=8, price=75, cost_price=45, tax=21)
builder.add_suplido("Travel", amount=150)  # Disbursement at cost
result = client.create_document_with_builder(builder)

# Clone estimate to invoice
new_invoice = client.clone_document("estimate_id", "invoice")

# Mark as paid
client.pay_document("invoice_id", {"amount": 1210.00}, doc_type="invoice")
```

---

## Audit Logging

All requests are logged to PostgreSQL with:
- Timestamp
- HTTP method and path
- Query parameters
- Status code
- Response time (ms)
- Cache hit/miss
- Client IP
- Error message (if any)

Query audit logs:
```sql
SELECT * FROM api_audit_logs
WHERE path LIKE '/api/v1/documents%'
  AND status_code >= 400
ORDER BY timestamp DESC
LIMIT 100;
```

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### Code Style

```bash
pip install black isort
black .
isort .
```

---

## License

Internal use only - Logistics Express

## Changelog

### v2.0.0 (2025-02-03)
- Added 33 new endpoints (45 total)
- Phase 1: Contact update/delete, PDF download, document clone/pay/update/delete
- Phase 2: Full CRUD for products and services
- Phase 3: Payments CRUD, treasuries list
- Phase 4: Accounting - chart of accounts, ledger, journal entries, trial balance

### v1.1.0 (2025-02-03)
- Added Redis caching with TTL
- Added PostgreSQL audit logging
- Added cache invalidation on mutations

### v1.0.0 (2025-02-03)
- Initial release with contacts, documents, estimates, debt checking
