# kiki-syn-agent

A conversational AI agent that bridges natural language with **Synera CAD** software and the **KIKI ML model**. Users describe what they want in plain language; a multi-agent system routes the request to the right specialist, calls the appropriate tool, and reports back the result.

**Part of:** KIKI project — WP 300 AI Optimization / WP 320 Dissemination

---

## What it does

```
User: "Recommend a lattice for an elderly patient, 60 kg"

  → Supervisor routes to KIKI agent
  → KIKI agent collects inputs, calls KIKI ML API (with HITL confirmation)
  → KIKI: "GYR lattice recommended — vol. fraction 0.25, rotation 17°/18°/10°"

  → Supervisor routes to Synera agent
  → Synera agent confirms parameters (HITL), calls reconstruct_lattice tool
  → syneraheadless.exe runs reconstruct_lattice.syn, exports lattice_image.png
  → Synera: "GYR lattice created. Ready in Synera."

User: "What hip contact forces occur during walking?"

  → Supervisor routes to Knowledge agent
  → Knowledge agent calls rag_search tool
  → rag_search retrieves relevant chunks from indexed research papers
  → Agent: "According to Bergmann et al. (2001), peak hip contact force during
             normal walking is 238% of body weight..."
```

---

## Architecture

Four specialised agents coordinated by a supervisor:

| Agent | Role | Tools |
|---|---|---|
| **Supervisor** | Reads conversation, routes to correct agent | None (structured output routing) |
| **KIKI agent** | Biomechanical optimisation — collects patient data and calls KIKI ML API | `kiki_recommend` |
| **Synera agent** | CAD geometry creation — runs Synera workflows | `reconstruct_lattice`, `create_sphere`, `create_pattern` |
| **Knowledge agent** | Answers questions from indexed research papers | `rag_search` |
| **Responder** | General questions and clarifications | None |

**UI:** Chainlit — conversational chat with history sidebar and inline image display.  
**LLM:** OpenAI `gpt-4o-mini` (primary) with Google Gemini `gemini-2.5-flash` automatic fallback on rate limits or connection errors.  
**HITL:** Human-in-the-loop confirmation before every tool call — the agent pauses and shows ✓ Yes / ✗ No / ✏ Change buttons.  
**Observability:** LangSmith tracing enabled via `.env`.  
**Conversation persistence:** PostgreSQL checkpointer (LangGraph) — full conversation history survives restarts.  
**History sidebar:** Chainlit SQLAlchemy data layer — past conversations listed and resumable from the sidebar.  
**Knowledge base:** PostgreSQL + pgvector — semantic search over indexed research papers.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Tested on 3.11 |
| Docker Desktop | Runs PostgreSQL + pgvector for conversation history and knowledge base |
| Synera with headless license | `syneraheadless.exe` must be on PATH or set via `SYNERA_EXE` in `.env` |
| OpenAI API key | Primary LLM — get one at platform.openai.com |
| Google AI API key | Fallback LLM — free tier at aistudio.google.com |
| KIKI ML API running | Separate local FastAPI server at `http://127.0.0.1:8000` |

---

## Setup

**1. Create and activate a virtual environment**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**2. Install dependencies**

```powershell
pip install -r requirements.txt
```

**3. Configure `.env`**

```powershell
copy .env.example .env
```

Open `.env` and fill in your keys. At minimum you need:

```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
POSTGRES_URI=postgresql://postgres:yourpassword@localhost:5432/kiki_agent
POSTGRES_PASSWORD=yourpassword
CHAINLIT_AUTH_SECRET=your-random-secret-string
```

Generate a value for `CHAINLIT_AUTH_SECRET` with any random string (e.g. `python -c "import secrets; print(secrets.token_hex(32))"`). It is required for the conversation history sidebar to work.

**4. Start the database container**

```powershell
docker compose up -d
```

**5. Set up the database (first time only)**

```powershell
python scripts/db_setup.py
```

Creates all required tables: LangGraph checkpointer tables, Chainlit data-layer tables, RAG tracking table, and conversation metadata tables.

**6. Index your research documents (first time only)**

Add PDFs to `tools/rag_search/documents/` then run:

```powershell
python tools/rag_search/ingest.py
```

Only new or changed files are re-embedded on subsequent runs (MD5 hash tracking).

**7. Run the app**

```powershell
.\start.ps1
```

Then open `http://localhost:8080` in your browser.

---

## Daily usage

```powershell
.\start.ps1
```

Starts the `rag_postgres` Docker container, checks whether the KIKI API is reachable, and launches the Chainlit UI on port 8080. The container reconnects to the existing database volume — no re-ingestion needed.

---

## Project structure

```
kiki-syn-agent-chainlit/
│
├── agent.py                    # Multi-agent supervisor with HITL and LLM fallback
├── app.py                      # Chainlit browser UI
├── docker-compose.yml          # PostgreSQL 16 + pgvector container definition
├── start.ps1                   # One-command startup (Docker check + KIKI check + Chainlit)
├── requirements.txt            # All dependencies pinned to exact versions
├── .env                        # API keys and config (never commit — gitignored)
├── .env.example                # Template for new contributors
├── chainlit.md                 # Chainlit welcome screen content
│
├── tools/                      # One subfolder per tool
│   ├── kiki_recommend/         # Calls the KIKI ML API
│   ├── reconstruct_lattice/    # Runs reconstruct_lattice.syn via syneraheadless
│   ├── create_sphere/          # Runs create_sphere.syn
│   ├── create_pattern/         # Runs create_pattern.syn
│   └── rag_search/             # Semantic search over research papers
│       ├── rag_search_tool.py  # LangChain tool — called by knowledge_agent
│       ├── ingest.py           # Incremental PDF indexing into pgvector
│       ├── evaluate.py         # RAG quality evaluation (21 test cases)
│       └── documents/          # Place research PDFs here (gitignored)
│
├── config/                     # Shared configuration
│   └── presets.py              # Bone condition material presets for KIKI
│
├── scripts/                    # Utility scripts
│   ├── db_setup.py             # One-time setup: all PostgreSQL tables
│   └── visualize_graph.py      # Saves LangGraph diagrams to diagrams/
│
├── tests/                      # Test suite
│   ├── test_langsmith.py       # Verifies LangSmith connection
│   └── test_fallback.py        # Verifies OpenAI → Gemini fallback
│
└── synera_run/                 # Developer utilities for Synera
    ├── run_workflow.py         # Manually run any .syn with a JSON input
    ├── get_template.py         # Inspect inputs/outputs of any .syn file
    └── synera_headless_reference.txt
```

---

## Available tools

### `kiki_recommend`
Calls the KIKI ML API with patient data and returns the optimal lattice parameters.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `bone_preset` | str | Yes | `normal` | `osteoporotic`, `elderly`, `normal`, `athletic` |
| `body_wt` | float | Yes | 75.0 | Patient body weight in kg (55–95) |
| `max_disp` | float | No | 5.0 | Maximum allowable displacement in mm |
| `max_stress` | float | No | 50.0 | Maximum allowable stress in MPa |
| `vol_frac` | float | No | 0.25 | Volume fraction (0.20–0.48) |
| `vox_to_surf` | float | No | 100.0 | Voxel to surface distance in mm |

The ML model predicts `cell_type` and rotation angles. `vol_frac` is passed through unchanged.

### `reconstruct_lattice`
Reconstructs a 3D lattice unit cell in Synera and exports `lattice_image.png`.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `cell_type` | str | Yes | — | `GYR`, `BCC`, `FCC`, `SCH`, `DIA`, `FLU`, `OCT`, `KEV`, `DTPMS`, `SPP`, `SC`, `DDK2`, `HCG`, `PSM2`, `RDO`, `TEG3` |
| `vol_frac` | float | No | 0.25 | Volume fraction (0.0–1.0) |
| `x_rotation` | float | No | 0.0 | Rotation around X axis in degrees |
| `y_rotation` | float | No | 0.0 | Rotation around Y axis in degrees |
| `z_rotation` | float | No | 0.0 | Rotation around Z axis in degrees |

### `create_sphere`
Creates a parametric 3D sphere in Synera. Exports a `.step` file only — no image.

| Parameter | Type | Required | Default |
|---|---|---|---|
| `radius` | float | Yes | — |
| `center_x/y/z` | float | No | 0.0 |

### `create_pattern`
Creates a 2D bio-inspired surface pattern in Synera and exports `pattern_image.png`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pattern_type` | int | 1 | 0 = Curved Beam, 1 = Glass Sponge 1, 2 = Glass Sponge 2 |
| `u_spacing` | float | 0.5 | Spacing in U direction |
| `v_spacing` | float | 0.5 | Spacing in V direction |

The agent will ask which pattern type you want if not specified — there are 3 distinct varieties.

### `rag_search`
Semantic search over indexed research papers using pgvector cosine similarity.

| Parameter | Type | Description |
|---|---|---|
| `query` | str | Plain-English question or search phrase |

Returns relevant excerpts with source filename. Filters low-relevance results via cosine distance threshold to prevent hallucination.

---

## Bone condition presets

Four presets define orthotropic material constants (E, G, NU) for the KIKI model:

| Preset | Patient profile |
|---|---|
| `osteoporotic` | Severely reduced bone density — very soft, compliant lattice |
| `elderly` | Age-related reduction (65+ years) — moderately soft lattice |
| `normal` | Healthy adult — default baseline values |
| `athletic` | Dense, high-stiffness bone — stiffer lattice |

---

## RAG knowledge base

### Adding documents

1. Drop PDFs into `tools/rag_search/documents/`
2. Run `python tools/rag_search/ingest.py`

Only new or modified files are re-embedded — unchanged files are skipped.

### Testing retrieval quality

```powershell
python tools/rag_search/evaluate.py
```

Runs 21 test cases derived from real facts in the indexed papers and reports an overall quality score.

### Docker commands

```powershell
docker compose up -d        # start the database
docker compose down         # stop (data preserved)
docker compose down -v      # stop and delete ALL data
docker ps                   # check container status
docker logs kiki_postgres    # view database logs
```

### Backing up the database

```powershell
docker exec kiki_postgres pg_dump -U postgres kiki_agent > backup.sql
```

---

## Running tests

```powershell
python tests/test_langsmith.py       # verify LangSmith connection
python tests/test_fallback.py        # verify OpenAI → Gemini fallback
python tools/rag_search/evaluate.py  # verify RAG retrieval quality
```

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not set` | Missing key in `.env` | Copy `.env.example` to `.env` and fill in keys |
| `CHAINLIT_AUTH_SECRET is not set` | Missing in `.env` | Generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `syneraheadless.exe not found` | Not on PATH | Set `SYNERA_EXE` in `.env` to the full path |
| `docker: command not found` | Docker not in PATH | Open a new terminal after Docker Desktop starts |
| `POSTGRES_URI is not set` | Missing in `.env` | Add `POSTGRES_URI=postgresql://postgres:pw@localhost:5432/kiki_agent` |
| `Knowledge base unavailable` | Container not running | Run `docker compose up -d` |
| `KIKI API connection refused` | KIKI FastAPI server not running | Start the KIKI server separately before running the agent |
| Rate limit error | OpenAI quota hit | Gemini fallback fires automatically — no action needed |
