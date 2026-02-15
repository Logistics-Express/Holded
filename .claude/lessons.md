# Lessons & Gotchas

> Auto-extracted from Claude Code session history.



## Key Decisions

### [2026-02-11]
Good, I can see the full file. I need to:

1. Adjust SUNSEEKER amortization for second-hand (coefficient duplicado = 20% instead of 10%)
2.

### [2026-02-09]
Now let me read the existing Phase 3 plans and the source code models to understand the current architecture.

### [2026-02-08]
```
◆ Spawning 4 researchers in parallel... → Stack research
  → Features research
  → Architecture research
  → Pitfalls research
```

### [2026-02-08]
The project initialization is complete. All artifacts are committed in `/Users/mj/Desarrollos/contacloud/`:

- `fe351ee` - PROJECT.md
- `45bf2d0` - config.json
- `668a194` - Research files (STACK, FEATURES, ARCHITECTURE, PITFALLS)
- `32a54ac` - SUMMARY.md (synthesized research)
- `77d5068` - REQUIRE

### [2026-02-08]
All research agents have completed. The project is fully initialized and ready to go. To recap the final state:

**ContaCloud** - `/Users/mj/Desarrollos/contacloud/`

| Commit | Artifact |
|--------|----------|
| `fe351ee` | PROJECT.md |
| `45bf2d0` | config.json |
| `668a194` | Research (STACK, FEA

### [2026-02-08]
✓ Researcher complete: `01-RESEARCH.md` written (commit `b08364a`)

Research covered: PostgreSQL RLS patterns, FastAPI async architecture, NIF/CIF validation, PGC chart structure, Next.js App Router auth, Celery task patterns, and 8 pitfalls to avoid. Now let me check for existing plans and move to 

### [2026-02-08]
One test failure — the NIE validator returns the converted number instead of the original format. Let me fix.

### [2026-02-08]
All 60 tests pass but the autouse cleanup fixture causes asyncpg connection conflicts on teardown. Unit tests don't need DB cleanup — let me fix the approach:

### [2026-02-08]
The issue is clear: `TenantMixin` requires `company_id` to be set on model instances, but the service functions don't set it. Let me check `TenantMixin` and then fix the approach.

### [2026-02-08]
## Phase 1 Complete — Human Checkpoint

**Phase 1: Foundation & Multi-Tenancy** is fully executed. Here's the summary:

### What Was Built (7 Plans, 21 commits)

| Plan | What | Key Files |
|------|------|-----------|
| 01-01 | Models, RLS, migrations | 12 models, Alembic migration with RLS policies

### [2026-02-06]
The SPF fix was necessary but this is a different problem - Barracuda is actively **blocking** the connection from Zoho's mail servers, not just rejecting for SPF. This could be:

1. Zoho's sending IPs are on Barracuda's blocklist
2.

### [2026-02-06]
All Zoho sending IPs are **clean** on Barracuda's global RBL. None are blocklisted. This means the block is **not global** - it's a **local policy** at benito.com's Barracuda Essentials instance (confirmed by the `.ess.` in `d315805.a.ess.de.barracudanetworks.com`).

### [2026-02-05]
Done! Here's what I created:

## Documentation Structure

### Central Hub: `Logistics-Express/stripe`
https://github.com/Logistics-Express/stripe

Contains comprehensive README with:
- Architecture diagram showing payment flow
- All 4 components documented
- Configuration for each component
- Securi

### [2026-02-04]
I'll help you plan the Stripe integration. First, I need to understand your current setup and requirements better. **Important security note:** You've shared live API keys in the chat.

### [2026-02-03]
The Holded API is now deployed and healthy. ## Summary

The original plan addressed the PORT expansion issue, but there were two problems:

1. **PORT expansion issue (from plan):** The `startCommand` in railway.toml was passing `$PORT` literally instead of expanding it.


## Insights

- **[2026-02-06]** Now I have the full picture. Here's what's wrong and the fix:

### Root Cause

The **main SPF record** for `logisticsexpress.es` is:
```
v=spf1 include:_spf.google.com ~all
```

But the bounced email was sent via **Zoho Desk** (`mail.zoho.eu`). Zoho's servers are **not authorized** in SPF, so Barrac
- **[2026-02-03]** I've identified the root cause of the build failure.

## Issue Analysis

**Error:** `Error: Invalid value for '--port': '$PORT' is not a valid integer.`

**Root Cause:** There's a conflict between the Dockerfile and railway.toml:

1. Railway detects the Dockerfile and uses it for building (seen in l
- **[2026-01-17]** Remaining breakdown:
- **Maribel (abmconsultores)**: 62 - asking about web repeatedly
- **Government (correo.gob.es)**: 62 - IMPORTANT
- **Social Security**: 27 - IMPORTANT
- **Ministry of Justice**: 17 - IMPORTANT
- **Gmail.com**: 12 - mixed
- **Other**: ~86 business emails

Let me clean Maribel's 
