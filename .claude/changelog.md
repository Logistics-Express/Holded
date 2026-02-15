# Changelog

> Auto-extracted from Claude Code session history.


> 65 commits, 303 files changed across 36 sessions

## 2026-02-11

- Files modified: 2
  - `/Users/mj/.qonto_credentials.json`
  - `claude-tools/bin/qonto-turbo`

## 2026-02-10

- Session: aqui tienes la carpeta de una empresa que quiero preparar para preconcurso para poder frenar el posible embargo de una embarcacion que tneemos judicia

## 2026-02-09

- Files modified: 1
  - `/Users/mj/.claude-profiles/sub1/plans/atomic-moseying-rain.md`

## 2026-02-08

- docs: add research output from 4 parallel researchers  - STACK.md: Python/FastAPI/PostgreSQL/Redis recommendation - FEATURES.md: Competitor analysis and feature prioritization - ARCHITECTURE.md: Multi-tenant event sourcing with CQRS - PITFALLS.md: 10 critical risks with prevention strategies
- docs: add repo reference to PROJECT.md
- feat: scaffold project structure  - README.md with overview, setup instructions, and roadmap summary - pyproject.toml with FastAPI/SQLAlchemy/Pydantic/WeasyPrint deps - Dockerfile + docker-compose.yml (app, PostgreSQL 15, Redis 7, Celery worker) - Alembic migrations setup with async PostgreSQL suppo
- feat(01-04): implement company management API with PGC seeding  Company CRUD with full automated onboarding pipeline: - NIF/CIF/NIE format validation with check digit algorithm - IVA/IGIC tax regime auto-detection from postal code/province - PGC chart of accounts seeding (3 templates: PyME, Full, Fo
- feat(01-05): implement accounting engine with journal entries, ledger, trial balance  Double-entry accounting engine with balance validation at 3 levels: - Pydantic schema validators (JournalEntryCreate) - Service layer double-check - Database CHECK constraint (ck_entry_balanced)  Features: journal
- docs(01-04,01-05): complete plan execution summaries and update state  Plans 01-04 (Company API) and 01-05 (Accounting Engine) complete. 5/7 plans done in Phase 1 (71%). 60 total tests passing.
- feat(01-06): implement frontend dashboard with company switching and accounting views  Full dashboard UI: company cards grid, Cmd+K switcher, sidebar navigation. Accounting pages: journal entries with expand/collapse, chart of accounts, ledger with running balances, trial balance with CSV export, se
- docs(01-06): complete frontend dashboard execution summary and update state
- feat(01-07): add integration tests, fix RLS tenant isolation, and resolve schema bugs  - Create 18 integration tests: 8 full-flow (register→company→entries→reports)   and 10 RLS isolation (cross-tenant access denied, DB-level RLS, superadmin   bypass, role-based access) - Fix selectinload bug in aut
- docs(01-07): complete Phase 1 execution summary and mark phase complete
- fix: Dockerfile build order and module path for Railway deploy  - Move COPY src/ before pip install (setuptools needs source to build) - Fix uvicorn module path: contacloud.main:app (not src.contacloud.main:app) - Add .dockerignore to exclude frontend, tests, docs from image
- fix: Dockerfile libgdk-pixbuf package name for Debian Trixie  Package renamed from libgdk-pixbuf2.0-0 to libgdk-pixbuf-2.0-0 in Trixie.
- docs(phase-2): fix plan checker blockers - cert alerting, VIES retry, API-05 scope  - Add certificate expiry background task to Plan 02-06 (COMP-14 alerting) - Fix misleading VIES retry note in Plan 02-03 (deferred to Phase 4) - Move API-05 (bank CSV import) from Phase 2 to Phase 4 (needs reconcilia

## 2026-02-06

- Files modified: 10
  - `/private/tmp/claude-501/-Users-mj/1f536c9f-68a3-4ddc-81b2-e03f4d5e6b9c/scratchpad/email_body.txt`
  - `/private/tmp/claude-501/-Users-mj/1f536c9f-68a3-4ddc-81b2-e03f4d5e6b9c/scratchpad/email_body_v2.txt`
  - `/private/tmp/claude-501/-Users-mj/1f536c9f-68a3-4ddc-81b2-e03f4d5e6b9c/scratchpad/phase4_cleanup.py`
  - `/private/tmp/claude-501/-Users-mj/1f536c9f-68a3-4ddc-81b2-e03f4d5e6b9c/scratchpad/plan_verification.md`
  - `/private/tmp/claude-501/-Users-mj/1f536c9f-68a3-4ddc-81b2-e03f4d5e6b9c/scratchpad/plan_zoho_audit.md`
  - ... and 5 more

## 2026-02-05

- feat: add editable dashboard tasks with Quick Actions modal  Add editing capabilities to dashboard action items through a Quick Actions modal that allows users to manage tags, add notes, and archive items.  Backend: - Add ActionMetadata model for notes and archive status - Add /actions/{source}/{id}
- docs: document Quick Actions modal and dashboard features  - Add Dashboard Features section with Quick Actions Modal details - Document Archive System and Tag Management features - Add action_metadata, tags, action_tags to Database Tables - Add actions.py and tags.py to Files Structure
- fix: Add security hardening to Stripe payment links  - Add credentials file permission validation (warn if not 0600) - Validate Stripe secret key format on load - Add SSRF protection with URL whitelist for redirect URLs - Add amount validation (50 cents - 50,000 EUR) - Add currency validation (ISO 4
- fix: Add security hardening to Stripe webhook  - Add idempotency tracking to prevent duplicate event processing - Add rate limiting (60 requests/minute per IP) - Fix webhook timestamp tolerance check (reject old AND future) - Add webhook secret format validation - Add event_id format validation - Sa
- fix: Add authentication and validation to Stripe endpoints  Security fixes for all 5 Stripe reconciliation endpoints: - Add verify_service_auth to require X-Service-API-Key + X-Service-Name - Add document_id format validation (24 char hex) - Add currency validation (ISO 4217) - Add payment_intent_id
- feat: WordPress Stripe payment plugin with security hardening  Logistics Stripe Payments plugin features: - Stripe Elements integration for card payments - Stripe Checkout session support - Webhook handler for payment events - Holded ERP integration for reconciliation - Payment tracking database  Se
- docs: Add comprehensive Stripe integration documentation  Central documentation for all Stripe components: - Architecture overview with flow diagram - WordPress plugin usage and configuration - CLI tool commands and library API - Webhook handler details - Reconciliation API endpoints - Security feat
- docs: Add security features and link to central Stripe docs  - Link to Logistics-Express/stripe for central documentation - Document security features (validation, escaping, permissions)
- docs: Update CLAUDE.md with Stripe integration section  - Add Stripe Integration section with components, commands, webhooks - Update tool mapping to reference central docs - Add stripe project to Authorized Working Directories - Add lib/stripe_links/__init__.py for package exports

## 2026-02-04

- Files modified: 38
  - `/Users/mj/.claude-profiles/sub2/plans/fluffy-inventing-toast.md`
  - `/Users/mj/.claude-profiles/sub2/plans/tingly-nibbling-scone.md`
  - `/Users/mj/.claude/CLAUDE.md`
  - `/Users/mj/.claude/lessons/HISTORY.md`
  - `/private/tmp/claude-501/-Users-mj/94c7a9b0-268b-454e-a87b-87882b042139/scratchpad/cruce_deudores.py`
  - ... and 33 more

## 2026-02-03

- feat: Add Contable AI Agent with Qonto banking integration  - Add OpenAI GPT-4o integration for intelligent accounting decisions - Implement draft-approve-execute pattern for supervised autonomy - Add service-to-service authentication with API keys and rate limiting - Create Qonto banking client for
- fix: Add session() async context manager to DatabaseManager
- fix: Strip markdown code blocks from OpenAI JSON responses
- fix: Remove startCommand to fix PORT expansion issue  Railway auto-detects Dockerfile but startCommand was overriding it, causing $PORT to not expand properly. Let Dockerfile CMD handle the port expansion with shell form: ${PORT:-8001}
- fix: Use shell form CMD to enable PORT env var expansion  Changed from exec form (["uvicorn",...]) to shell form which allows ${PORT:-8001} to be expanded at runtime. Also cleaned up redundant COPY commands.
- fix: Add explicit error when holded library not found  Helps debug deployment issues by showing which paths were checked.
- fix: Use COPY . . to ensure all files including lib/ are copied  The selective COPY commands were missing files during Railway upload.
- debug: Add directory listing to diagnose lib path issue  Temporary debug output to understand why lib/ isn't being found in production.
- fix: Set lib_loaded=True when path exists, even if already in sys.path  The PYTHONPATH env var already includes /app/lib, so the check for "not in sys.path" was preventing lib_loaded from being set to True.
- feat: Expand API with 33 new endpoints (v2.0.0)  Phase 1 - Critical Operations: - PUT/DELETE /contacts/{id} - Update and delete contacts - GET /documents/{type}/{id}/pdf - Download document as PDF - POST /documents/{id}/clone - Clone/convert documents (estimate → invoice) - POST /documents/{type}/{i
- docs: Add comprehensive API documentation  - Full endpoint reference for all 45 endpoints - Request/response examples for every operation - Configuration guide (env vars, cache TTLs) - Architecture overview with file structure - HoldedClient library usage examples - Deployment instructions (local, D
- feat: Add Redis caching and PostgreSQL audit trail  - Add Redis caching for hot endpoints (contacts, documents, lookups) - Add audit middleware logging requests to PostgreSQL - Cache TTLs: contacts 5-10min, documents 15min, lookups 30min - Auto-invalidate cache on mutations (contact/estimate creatio
- Add year/date filtering to document endpoints  - Add year, date_from, date_to params to /api/v1/documents/invoices - Add year, date_from, date_to params to /api/v1/documents/estimates - Add year param to /api/v1/invoices/lookup and GET /api/v1/invoices/{number} - New convenience endpoint: GET /api/v
- Fix route order: year endpoint before generic document_id
- Fix Holded API param names: starttmp/endtmp instead of dateFrom/dateTo

## 2026-01-31

- feat(reconcile): Add outgoing payment reconciliation and multi-source search  - Add lib/reconcile/models.py with UnmatchedPayment, SearchResult dataclasses - Extend bin/reconcile-turbo with outgoing, full, and search commands - Support matching supplier payments with Holded purchases - Add search fa
- feat(reconcile): Add daily reconciliation workflow with outgoing support  - Add outgoing payment reconciliation (matching with supplier purchases) - Add InvoiceSearchService for multi-source search (Holded → Gmail → Zoho) - Add NotificationService for email + Telegram alerts on unmatched payments -

## 2026-01-30

- Files modified: 1
  - `/Users/mj/.claude-profiles/sub2/plans/buzzing-marinating-clarke.md`

## 2026-01-29

- feat: Add qonto-turbo CLI for Qonto banking API  - Organization and account info - Transaction listing and search - Bank statement management with PDF download - Beneficiary listing - JSON output support for all commands
- fix(holded): Fix pay_document endpoint path  The pay endpoint requires document type in path: /invoicing/v1/documents/{type}/{id}/pay
- feat: Add reconcile-turbo for Qonto-Holded payment reconciliation  Automatically matches incoming Qonto payments with pending Holded invoices based on: - Amount matching (±2 cents tolerance) - Invoice reference detection (F261277, FA-2024-001, etc.) - Company name similarity  Commands: - reconcile-t
- feat(reconcile-turbo): Support estimates + 18 months history  - Check ALL pending invoices AND estimates (not just current month) - Look back 18 months of Qonto transactions (was 90 days) - Match payments against estimates (presupuestos) too - Improved name matching: check both label AND reference f
- feat: Add 3-step debt collection workflow  Add step1, step2, step3 commands to legal-turbo for structured debt collection:  - step1: Gentle reminder with proformas + original invoices (status: PENDING) - step2: Formal burofax with all attachments (status: BUROFAX_SENT) - step3: Demanda notification

## 2026-01-26

- feat: Add Zoho CRM sales funnel & partner management system  - Extend ZohoCRMClient with Quotes, Sales_Orders, Vendors, Products,    Purchase_Orders modules (full CRUD operations) - Add partner library for managing logistics partners as Zoho Accounts   with tiers (Gold/Silver/Bronze), capabilities,

## 2026-01-23

- feat(holded): add full document management with margins and templates  - Add LineItem, DocumentBuilder dataclasses for document creation with   real cost tracking and margin calculation per line - Add ItemKind enum (line, title, subtitle) and TaxRate enum - Support suplidos (disbursements) as pass-t
- feat(holded): add complete document creation with addresses and payment  - Add pickup/delivery address, cargo description, payment terms to DocumentBuilder - Build structured notes automatically from address/cargo info - Default payment method: Transferencia Qonto (configurable) - Include bank IBAN
- fix: read contact field from API response in DocumentBuilder.from_document  API returns 'contact' not 'contactId' - fixes clone_document failing with empty contact_id when converting estimates to invoices.
- feat: add TemplateManager and services batch-update  - Add TemplateManager class for reusable document templates - Add DocumentBuilder.to_template() and from_template() methods - Add services batch-update command for bulk costPrice updates - Update template commands to use TemplateManager - Export T
- feat: add Canarias territories and m³ pricing  - Add 7 Canarias islands with volume-based pricing (€/m³) - Add transport_cost_per_m3 field to TerritoryConfig - Add pricing_type field ('pallet' or 'volume') - Update MIN_SHIPMENT_M3 to 37.50€ for volume pricing - Update pallet costs: Melilla/Ceuta ame
- feat: add gmail-send tool and attachment commands to gmail-turbo  gmail-send: - Send drafts directly - Send emails immediately - Schedule emails (creates draft for Gmail web scheduling) - List and delete drafts  gmail-turbo: - attachments list <message_id> - attachments download <message_id> --outpu
- feat: add display name support to gmail-send and gmail-draft  - Read display names from ~/.claude_tools/config.yaml - Format From header as "Name <email>" instead of just "email" - Add email_aliases section to config with names for all accounts - Add pyyaml dependency
- feat: add gmail-draft-alias for drafts with sender aliases  Creates drafts in your account (--account) but with a different From address (--from). Useful for managing drafts in one mailbox while sending as different aliases.  Defaults: - account: mj@logisticsexpress.es - from: lmartin@logisticsexpre
- fix: use Gmail SendAs settings for correct From header  Now fetches the exact display name from Gmail SendAs aliases instead of config file, ensuring the From header is accepted.
- fix: auto-remove trailing greetings when signature is used  The signature already includes "Un saludo." so the tool now automatically removes duplicate greetings from body text: - Un saludo, / Un saludo. - Saludos, / Atentamente, / Cordialmente, - Best regards, / Kind regards, / Thanks,  Works for b
- feat: Add navieras API for ferry quotes  - Add navieras.yaml config with 9 ferry companies - Add /api/navieras endpoints for quotes, routes, ports - Includes: Trasmed/Armas, Baleària, Servimad, FRS/DFDS,   Fred Olsen, GNV, Grimaldi, AML, Inter Shipping - Tariffs pending - structure ready for when re
- security: Add JWT authentication to navieras API endpoints  All /api/navieras/* endpoints now require valid JWT token. Uses @api_login_required decorator from existing auth system.
- temp: Add bootstrap endpoint for token creation  Temporary endpoint to create API token in Railway database. Will be removed after token is created.
- fix: Use raw SQL for bootstrap token creation
- feat: Add navieras (ferry companies) API endpoints  - Add /api/v1/navieras endpoints for ferry operations - List navieras, routes, ports, vehicle types - Quote endpoint for ferry crossings - System status endpoint - navieras.yaml config with 9 ferry companies - Add pyyaml dependency
- feat: Add token authentication to all endpoints  - Create app/core/deps.py with require_auth dependency - Add HMAC-SHA256 token verification - Support both Bearer token and X-API-KEY header - Protect all endpoints except /health  Token format: prefix (8 chars) + HMAC signature (64 chars)
- feat: add customs agents API with tariff pricing  - Add customs_agents.yaml with Cabrero, AGSA, Quiles agents - Add tariffs per route with vehicle and complexity surcharges - Create /api/v1/customs-agents endpoints:   - GET / - list agents   - GET /{code} - agent details   - GET /{code}/tariffs - ag
- feat: add Andorra territory and transitarios for Gibraltar/Andorra  - Add ANDORRA to TerritoryCode enum in schemas and services - Add Andorra territory config (La Seu d'Urgell warehouse, 55€/pallet) - Add transitarios to customs_agents.yaml:   - gibraltar-logistics: Algeciras → Gibraltar (status: pe
- feat: add Medrano / Red & Pallets national pallet pricing  - Add medrano_tariffs.yaml with 56 destination zones - Add MedranoPricingService for quote calculation - Add API endpoints: /medrano/quote, /zones, /pallet-types, etc. - Support P200-P1200 pallet types - Coverage: Spain + Portugal + Baleares
- feat: add Canarias volume pricing and public estimate endpoint  - Add Canarias islands to TerritoryCode enum (tenerife, gran_canaria, etc.) - Implement volume-based pricing (€/m³) for Canarias - Add dimension fields (length_cm, width_cm, height_cm) to FullTerritoryQuoteRequest - Calculate port fees

## 2026-01-21

- Files modified: 5
  - `/Users/mj/.claude-profiles/sub2/plans/streamed-baking-nebula.md`
  - `/Users/mj/.holded_credentials.json`
  - `claude-tools/bin/holded-turbo`
  - `claude-tools/lib/holded/__init__.py`
  - `claude-tools/lib/holded/holded_client.py`

## 2026-01-17

- Files modified: 2
  - `/Users/mj/.claude/plans/memoized-percolating-candy.md`
  - `/Users/mj/bin/gmail-filters`
