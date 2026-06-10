# Memlord — Design Document

## Overview

MCP server for storing and searching memory. Hybrid search: BM25 (full-text) + vector (semantic similarity) combined via
Reciprocal Rank Fusion.

## Stack

| Component        | Library                                                                 |
|------------------|-------------------------------------------------------------------------|
| MCP framework    | `fastmcp >= 3.1.0` — standalone server                                  |
| UI               | `fastapi[all]`                                                          |
| Database         | PostgreSQL (`asyncpg` + SQLAlchemy async)                               |
| Migrations       | `alembic` + `alembic-autogen-check` (dev)                               |
| Vector store     | `pgvector` — `vector(384)` column in `memories`                         |
| Full-text search | PostgreSQL `tsvector GENERATED ALWAYS AS` + `websearch_to_tsquery`      |
| Embeddings       | `onnxruntime` + `paraphrase-multilingual-MiniLM-L12-v2.onnx` (384 dims) |
| Tokenization     | `tokenizers`                                                            |
| Time parsing     | `dateparser`                                                            |
| Model            | ONNX files excluded from git, downloaded via script                     |
| Configuration    | `pydantic-settings`                                                     |
| Auth             | OAuth 2.1 — custom `OAuthProvider` (fastmcp)                            |
| Password hashing | `bcrypt`                                                                |
| Deployment       | Docker + docker-compose                                                 |

## Dependencies

**production:** `asyncpg`, `alembic`, `authlib`, `bcrypt`, `email-validator`, `fastapi[all]`, `fastmcp`, `greenlet`,
`onnxruntime`, `pgvector`, `pydantic-settings`, `dateparser`, `sqlalchemy[asyncio]`, `tokenizers`

**dev:** `alembic-autogen-check`, `black`, `httpx`, `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-mock`,
`pytest-xdist`

## Project Structure

```
src/memlord/
├── __init__.py
├── main.py                # entrypoint: FastAPI app + uvicorn
├── server.py              # FastMCP("Memlord") + mcp.mount() per tool
├── config.py              # pydantic-settings (MEMLORD_* prefix)
├── db.py                  # async SQLAlchemy engine (asyncpg) + SessionDep
├── embeddings.py          # ONNX session, tokenization, mean pooling, L2 norm
├── search.py              # hybrid BM25 + vector KNN + RRF fusion
├── oauth.py               # custom OAuthProvider (fastmcp.server.auth)
├── auth.py                # hash_password, verify_password, UserDep
├── models/
│   ├── __init__.py        # re-exports only
│   ├── base.py            # SQLAlchemy Base + naming convention
│   ├── memory.py          # Memory table (embedding, search_vector, created_by, workspace_id)
│   ├── tag.py             # Tag table
│   ├── memory_tag.py      # MemoryTag M2M table
│   ├── oauth_client.py    # OAuthClient table
│   ├── user.py            # User table (email, hashed_password, display_name)
│   └── workspace.py       # Workspace, WorkspaceMember, WorkspaceInvite tables
├── schemas/
│   ├── __init__.py        # re-exports only
│   ├── memory_type.py     # MemoryType StrEnum
│   ├── search.py          # SearchResult, MemoryResult
│   ├── store.py           # StoreResult
│   ├── recall.py          # RecallResult
│   ├── list_memories.py   # MemoryListItem, MemoryPage
│   ├── delete.py          # DeleteResult
│   ├── update.py          # UpdateMemoryRequest
│   ├── health.py          # HealthResult
│   └── workspace.py       # workspace-related schemas
├── dao/
│   ├── __init__.py        # re-exports MemoryDao, UserDao, WorkspaceDao
│   ├── memory.py          # MemoryDao — DB access layer for memories
│   ├── user.py            # UserDao — DB access layer for users
│   └── workspace.py       # WorkspaceDao — DB access layer for workspaces
├── utils/
│   ├── __init__.py
│   └── dt.py              # date/time helpers
├── templates/             # Jinja2 templates (base, index, search, memory)
├── onnx/
│   ├── model.onnx         # all-MiniLM-L6-v2 (excluded from git, see scripts/)
│   └── tokenizer.json     # (excluded from git, see scripts/)
├── tools/
│   ├── __init__.py        # re-exports: mcp instances as named aliases
│   ├── store.py           # store_memory → StoreResult
│   ├── retrieve.py        # retrieve_memory → list[MemoryResult]
│   ├── recall.py          # recall_memory → list[RecallResult]
│   ├── list_memories.py   # list_memories → MemoryPage
│   ├── get_memory.py      # get_memory → MemoryListItem
│   ├── search_by_tag.py   # search_by_tag → list[MemoryListItem]
│   ├── delete.py          # delete_memory → DeleteResult
│   ├── update.py          # update_memory → MemoryListItem
│   ├── health.py          # check_database_health → HealthResult
│   └── workspaces.py      # workspace tools (create, list, invite, join, leave)
└── ui/
    ├── __init__.py        # assembles ui_router from sub-routers
    ├── base.py            # pages: index, search, memory detail, update, delete
    ├── data.py            # export/import JSON
    ├── login.py           # login form (GET/POST /ui/login)
    ├── utils.py           # templates, session_token, require_auth
    └── workspaces.py      # workspace UI pages
scripts/
└── download_model.py      # download model.onnx + tokenizer.json from HuggingFace
migrations/                # Alembic
├── env.py                 # async engine (asyncpg), asyncio.run()
├── script.py.mako
└── versions/
alembic.ini
Dockerfile
docker-compose.yml
.env.example
```

---

## Configuration

Via `pydantic-settings`. Sources in priority order: environment variables (prefix `MEMLORD_`) → `.env` file → defaults.

| Variable                   | Default                                                    | Description                                                       |
|----------------------------|------------------------------------------------------------|-------------------------------------------------------------------|
| `MEMLORD_DB_URL`           | `postgresql+asyncpg://postgres:postgres@localhost/memlord` | PostgreSQL connection URL                                         |
| `MEMLORD_DB_ECHO`          | `false`                                                    | SQLAlchemy query logging                                          |
| `MEMLORD_MODEL_DIR`        | `/app/src/memlord/onnx`                                    | Directory containing ONNX model                                   |
| `MEMLORD_HOST`             | `0.0.0.0`                                                  | uvicorn host                                                      |
| `MEMLORD_PORT`             | `8000`                                                     | uvicorn port                                                      |
| `MEMLORD_BASE_URL`         | —                                                          | Public server URL (enables OAuth)                                 |
| `MEMLORD_RRF_K`            | `60`                                                       | RRF fusion k parameter                                            |
| `MEMLORD_DEFAULT_LIMIT`    | `10`                                                       | Default result limit                                              |
| `MEMLORD_SIM_THRESHOLD`    | `0.25`                                                      | Default cosine similarity threshold                               |
| `MEMLORD_DEDUP_THRESHOLD`  | `0.85`                                                     | Cosine similarity threshold for near-duplicate detection on write |
| `MEMLORD_OAUTH_JWT_SECRET` | `memlord-dev-secret-please-change`                         | JWT signing secret                                                |

---

## Embeddings Pipeline

`content` → tokenization (`tokenizer.json`) → ONNX inference → mean pooling (with attention mask) → L2 normalize →
`float32[384]` → `memories.embedding` (`vector(384)`, pgvector)

Model files: `src/memlord/onnx/model.onnx`, `src/memlord/onnx/tokenizer.json` — excluded from git. Download before
running: `uv run python scripts/download_model.py` (source: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`).

---

## Database Schema

**users** — `id` (PK), `email` (UNIQUE), `display_name`, `hashed_password` (bcrypt), `created_at`

**oauth_clients** — OAuth client registrations: `client_id` (PK), `data` (JSONB), `user_id` (FK → `users.id`, nullable),
`created_at`

**memories** — main table: `id` (PK), `content` (TEXT), `created_by` (FK → `users.id`), `memory_type`, `metadata` (
JSONB), `workspace_id` (FK → `workspaces.id` NOT NULL, ON DELETE CASCADE), `embedding` (`vector(384)` —
pgvector), `search_vector` (`TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED`), `created_at`

Unique constraint: `uq_memories_content_workspace (content, workspace_id)` — idempotency per workspace.

Indexes: GIN on `search_vector`, HNSW on `embedding` (cosine ops, m=16, ef_construction=64).

**tags** — `id`, `name` (UNIQUE)

**memory_tags** — M2M: `memory_id` → `memories.id` (CASCADE), `tag_id` → `tags.id` (CASCADE)

**workspaces** — `id` (PK), `name` (UNIQUE), `created_by` (FK → `users.id`), `is_personal` (BOOLEAN NOT NULL DEFAULT
FALSE), `created_at`

Partial unique index: `uq_workspaces_personal_per_user ON workspaces (created_by) WHERE is_personal = TRUE` — enforces
one personal workspace per user.

**workspace_members** — `workspace_id` (FK → `workspaces.id`, CASCADE) + `user_id` (FK → `users.id`, CASCADE) —
composite PK; `role` (default `member`), `joined_at`

**workspace_invites** — `id` (UUID string, PK), `workspace_id` (FK → `workspaces.id`, CASCADE), `created_by` (FK →
`users.id`), `expires_at`, `used_by` (FK → `users.id`, nullable), `used_at` (nullable)

---

## Hybrid Search: BM25 + Vector + RRF

### Algorithm

`hybrid_search(session, query, workspace_ids, ...)` in `search.py`:

1. Build **access filter**: `Memory.workspace_id.in_(workspace_ids)`.
2. Two sequential queries:
    - **BM25**: `Memory.search_vector @@ websearch_to_tsquery('simple', query)` + tag match (
      `to_tsvector('simple', tag.name) @@ tsquery`). Ranked via `row_number() OVER (ORDER BY ts_rank DESC)`. Fetches
      `limit * 4` rows.
    - **Vector KNN**: `Memory.embedding <=> :vec` (pgvector cosine distance). Ranked via
      `row_number() OVER (ORDER BY distance)`. Fetches `limit * 4` rows.
3. **RRF fusion** (in Python): `rrf(d) = 1/(k + rank_bm25) + 1/(k + rank_vec)`, where `k = 60`. If a document appears in
   only one list, the other term is 0.
4. **Similarity filter**: `similarity = 1 - cosine_distance`. BM25 hits always pass; pure vector hits are filtered out
   if `similarity < threshold`.
5. Sort by `rrf_score DESC`, return top-`limit`.

### Parameters

| Parameter               | Description                                                       |
|-------------------------|-------------------------------------------------------------------|
| `workspace_ids`         | List of accessible workspace IDs                                  |
| `limit`                 | Max results (default: `MEMLORD_DEFAULT_LIMIT`)                    |
| `similarity_threshold`  | Threshold for pure vector hits (default: `MEMLORD_SIM_THRESHOLD`) |
| `date_from` / `date_to` | Filter by `created_at`                                            |
| `memory_type`           | Filter by type                                                    |

### Usage in tools

| Tool              | BM25 | Vector | RRF                                     |
|-------------------|------|--------|-----------------------------------------|
| `retrieve_memory` | ✅    | ✅      | ✅                                       |
| `recall_memory`   | ✅    | ✅      | ✅ (+ date filter from natural language) |
| `search_by_tag`   | —    | —      | — (tag filter only)                     |
| `list_memories`   | —    | —      | — (date sort only)                      |
| `get_memory`      | —    | —      | — (name lookup only)                    |

---

## MCP Tools

All tools require authentication. `UserDep` resolves `user_id` from the access token via
`client_id → oauth_clients.user_id`. `SessionDep` manages the transaction — commit on success, rollback on exception.

### `store_memory`

Save a memory entry.

| Field         | Type     | Required | Description                                                          |
|---------------|----------|----------|----------------------------------------------------------------------|
| `content`     | string   | ✅        | Memory text                                                          |
| `memory_type` | string   | ✅        | Type: `fact`, `preference`, `instruction`, `feedback`                |
| `tags`        | string[] | ❌        | Tags                                                                 |
| `metadata`    | object   | ❌        | Arbitrary metadata (JSON)                                            |
| `workspace`   | string   | ❌        | Workspace name (must have write access). `null` → personal workspace |

**Logic:** if `workspace` is provided — name lookup, `can_write` check → `workspace_id`. If `null` → personal
workspace resolved via `get_personal(uid)`. Unique constraint `(content, workspace_id)` — idempotent per workspace.
`MemoryDao.create` → embedding → near-duplicate check → tags.

**Near-duplicate check:** before inserting, computes embedding and finds the closest existing memory in the workspace
via `<=>` (cosine distance). If `1 - distance >= dedup_threshold` — raises `ValueError` with the duplicate's `id` and
similarity score. Skipped when `force=True`.

**Returns:** `StoreResult` — `id`, `created` (bool).

---

### `retrieve_memory`

Hybrid semantic + full-text search.

| Field                  | Type    | Default | Description                     |
|------------------------|---------|---------|---------------------------------|
| `query`                | string  | —       | Search query                    |
| `limit`                | integer | 10      | Max results                     |
| `similarity_threshold` | float   | 0.7     | Min cosine similarity (0.0–1.0) |
| `memory_type`          | string  | ❌       | Filter by type                  |

**Logic:** fetch user's `workspace_ids` → `hybrid_search` (personal + workspaces) → enrich with tags and metadata.

**Returns:** `list[MemoryResult]` — `id`, `content`, `memory_type`, `tags`, `metadata`, `created_at`, `rrf_score`,
`workspace_id`.

---

### `recall_memory`

Time-based + semantic search in natural language.

| Field         | Type    | Default | Description                                                      |
|---------------|---------|---------|------------------------------------------------------------------|
| `query`       | string  | —       | Query: `"last week"`, `"yesterday"`, `"about Python last month"` |
| `n_results`   | integer | 5       | Max results                                                      |
| `memory_type` | string  | ❌       | Filter by type                                                   |

**Logic:** `dateparser.search_dates` extracts temporal expressions → `date_from`/`date_to` → `hybrid_search` with
`similarity_threshold=0.0` and date filter. If no dates found — plain hybrid search.

**Returns:** `list[RecallResult]` — `id`, `content`, `memory_type`, `tags`, `created_at`, `workspace_id`.

---

### `update_memory`

Update an existing memory identified by name. Only provided fields are changed.

| Field         | Type     | Required | Description                              |
|---------------|----------|----------|------------------------------------------|
| `name`        | string   | ✅        | Memory name                              |
| `memory_type` | string   | ✅        | New type                                 |
| `content`     | string   | ❌        | New text (if changing)                   |
| `new_name`    | string   | ❌        | Rename the memory to this name           |
| `tags`        | string[] | ❌        | New tags (full replacement)              |
| `metadata`    | object   | ❌        | New metadata (full replacement)          |
| `workspace`   | string   | ❌        | Disambiguate if name exists in multiple  |

**Logic:** resolve name (+ optional workspace) → access check → update fields → regenerate embedding if `content` changed.

**Returns:** `StoreResult` — `name`, `created=False`.

---

### `list_memories`

Paginated list with filtering.

| Field         | Type    | Default | Description           |
|---------------|---------|---------|-----------------------|
| `page`        | integer | 1       | Page number (1-based) |
| `page_size`   | integer | 10      | Page size             |
| `memory_type` | string  | ❌       | Filter by type        |
| `tag`         | string  | ❌       | Filter by tag         |

**Logic:** SELECT with LIMIT/OFFSET, access filter (personal + workspaces), sorted by `created_at DESC`.

**Returns:** `MemoryPage` — `items: list[MemoryItem]`, `total`, `page`, `page_size`, `total_pages`. Each `MemoryItem`:
`name`, `memory_type`, `metadata`, `tags`, `created_at`, `workspace` (no `id`, no `content`).

---

### `search_by_tag`

Tag search with boolean logic.

| Field       | Type         | Default | Description           |
|-------------|--------------|---------|-----------------------|
| `tags`      | string[]     | —       | Tags to search        |
| `operation` | `AND` / `OR` | `AND`   | Tag combination logic |

**Logic:** `AND` — all tags present; `OR` — at least one. Access filter applied.

**Returns:** `MemoryPage` — `items: list[MemoryItem]` (`name`, `memory_type`, `metadata`, `tags`, `created_at`, `workspace`).

---

### `get_memory`

Fetch a single entry by name.

| Field       | Type   | Required | Description                             |
|-------------|--------|----------|------------------------------------------|
| `name`      | string | ✅        | Memory name                             |
| `workspace` | string | ❌        | Disambiguate if name exists in multiple |

**Logic:** resolve name (+ optional workspace) → access check (personal or in user's workspace).

**Returns:** `MemoryDetail` — `name`, `content`, `memory_type`, `metadata`, `tags`, `created_at`.

---

### `delete_memory`

Delete an entry by name.

| Field       | Type   | Required | Description                             |
|-------------|--------|----------|------------------------------------------|
| `name`      | string | ✅        | Memory name                             |
| `workspace` | string | ❌        | Disambiguate if name exists in multiple |

**Logic:** resolve name (+ optional workspace) → access check → DELETE from `memories` (CASCADE → `memory_tags`);
`embedding` and `search_vector` are removed with the row.

**Returns:** `DeleteResult` — `success`, `name`.

**Returns:** `DeleteResult` — `success: bool`, `id`.

---

### `list_workspaces`

List all workspaces the user is a member of, including the personal workspace (`is_personal=true`).

No parameters.

**Returns:** `list[WorkspaceInfo]` — `id`, `name`, `role`, `member_count`, `is_personal`.

> Workspace management (create shared workspace, generate invite link, join via token, leave, delete) is handled
> exclusively through the Web UI.

---

## Deployment

Two containers: PostgreSQL + Memlord server. Start with `docker compose up`.

**Architecture:** FastAPI is the main ASGI app. UI routers are registered first (`app.include_router`), then the MCP app
is mounted at `/` (`app.mount("/", mcp_app)`). Single port, single uvicorn process.

```
FastAPI (/)
├── /ui/...    — Web UI (list, search, view, edit, delete, workspaces)
└── /mcp       — MCP HTTP (fastmcp)
/.well-known/oauth-authorization-server   — OAuth metadata
/.well-known/oauth-protected-resource     — resource metadata (RFC 9728)
/login                                    — OAuth login / registration form
/authorize                                — OAuth 2.1 authorization endpoint
/token                                    — OAuth 2.1 token endpoint
/register                                 — Dynamic Client Registration
/revoke                                   — Token revocation
```

**Dockerfile:** multi-stage build (`python:3.12-slim`, uv). Stage 1: dependencies + ONNX model download (
`scripts/download_model.py`). Stage 2: runtime with `libgomp1` (required by onnxruntime). Entrypoint:
`alembic upgrade head && memlord`.

## Authentication

OAuth 2.1 — `MemlordOAuthProvider(OAuthProvider)` in `oauth.py`, full in-process Authorization Server.

**Mechanism:**

- `authorize()` → saves `_PendingAuth` → redirects to `/login?id=<pending_id>`
- `/login` GET — HTML form (email + password); POST:
    - user not found → registration form (email, display_name, password)
    - wrong password → form with error
    - success → auth code → redirect to `redirect_uri`, `oauth_clients.user_id` linked to the user
- JWT access + refresh tokens: `JWTIssuer` (HS256, key derived via HKDF from `oauth_jwt_secret`)
    - access: 1 hour, stored in memory by JTI
    - refresh: 30 days, stored in memory by token string
    - after server restart: tokens are reconstructed from JWT claims (fallback)
- Token rotation on `exchange_refresh_token` — old pair revoked via `_revoke_pair`
- Dynamic Client Registration enabled (scope: `mcp`)
- Revocation: `_revoke_pair` removes both sides of the access ↔ refresh pair

**Enabled** when `MEMLORD_BASE_URL` is set in config (JWT secret always has a default). Without `MEMLORD_BASE_URL` —
server starts without authentication.

---

## Web UI (FastAPI)

Web interface over the same database. Stack: FastAPI + Jinja2 (SSR) + HTMX. All routes under `/ui` prefix, require
authentication via session cookie (`get_current_user`).

| Section    | Description                                                                                   |
|------------|-----------------------------------------------------------------------------------------------|
| List       | `GET /` — paginated list, filter by type, tag, workspace                                      |
| Search     | `GET /search?q=...` — hybrid search (BM25 + vector)                                           |
| View       | `GET /memory/{id}` — detail card: content, tags, metadata, workspace, date                    |
| Edit       | `PUT /memory/{id}` — update content, memory_type, tags, metadata                              |
| Move       | `POST /memory/{id}/move` — move memory to another workspace (write access required on target) |
| Delete     | `DELETE /memory/{id}` — delete entry                                                          |
| Workspaces | `GET /ui/workspaces` — list user's workspaces (personal shown first)                          |
|            | `GET/POST /ui/workspaces/new` — create shared workspace                                       |
|            | `GET /ui/workspaces/{id}` — workspace detail, member list                                     |
|            | `POST /ui/workspaces/{id}/invite` — generate invite link (non-personal only)                  |
|            | `POST /ui/workspaces/{id}/leave` — leave workspace (non-personal only)                        |
|            | `POST /ui/workspaces/{id}/delete` — delete workspace (owner, non-personal only)               |
|            | `GET/POST /ui/join/{token}` — accept invite via token                                         |
| Login      | `GET/POST /ui/login` — web UI login                                                           |
| Data       | `GET /ui/export` / `POST /ui/import` — export/import JSON                                     |

---

## FastMCP Tool Conventions

- Each tool file has its own `mcp = FastMCP()`, tool registered via `@mcp.tool`
- `server.py` mounts all sub-servers via `mcp.mount()`
- Sessions: `MCPSessionDep` (MCP tools) and `APISessionDep` (FastAPI routes) from `memlord.db`; typed as
  `s: AsyncSession = MCPSessionDep  # type: ignore[assignment]`
- `UserDep` from `memlord.auth` — resolves `user_id` from OAuth access token:
  `uid: int = UserDep  # type: ignore[assignment]`
- Commit/rollback managed by `session_dep` — never call manually
- DB access via DAO (`MemoryDao`, `WorkspaceDao`, `UserDao`) — tools do not execute queries directly; `hybrid_search` is
  the exception (used as a utility function)
- `output_schema=Model.model_json_schema()` — only for tools returning an object; list-returning tools use bare
  `@mcp.tool` (MCP spec does not allow array as output_schema)
- SQLAlchemy Core: `select()`, `insert()`, `delete()`, `update()`. `sa.text()` only for PostgreSQL-specific syntax.
  `.mappings().all()` for multi-column results
