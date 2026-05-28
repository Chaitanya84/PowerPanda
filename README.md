# PowerPanda

A Graph-enhanced RAG application that answers questions about your documents by combining **knowledge graph traversal** with **vector similarity search**, powered by Claude Opus.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Documents (PDF/DOCX/XLSX/CSV/TXT)                           │
│      │                                                       │
│      ├──► Chunking ──► Embed (text-embedding-ada-002)        │
│      │                      ──► FAISS Vector Store           │
│      │                                                       │
│      └──► LLM Entity/Relation Extraction (Claude Opus)       │
│                ──► In-Memory Knowledge Graph (JSON)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Query                                                  │
│      │                                                       │
│      ├──► Embed query (text-embedding-ada-002)               │
│      │        │                                              │
│      │        ├──► Cosine similarity against node embeddings │
│      │        │        ──► Top-K relevant graph nodes        │
│      │        │              ──► N-hop traversal             │
│      │        │                    ──► Graph Context         │
│      │        │                                              │
│      │        └──► FAISS similarity search                   │
│      │                   ──► Doc Context                     │
│      │                                                       │
│      ▼                                                       │
│  Combine contexts into augmented prompt                      │
│      + soul.md personality ──► Claude Opus ──► Answer        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python) |
| Frontend | HTML/CSS/JS (dark theme, single-page) |
| LLM | Claude Opus 4.6 (via GenAI-Nexus / AWS Bedrock) |
| Embeddings | text-embedding-ada-002 (Azure OpenAI) |
| Vector Store | FAISS |
| Knowledge Graph | In-memory (Python dicts, persisted as JSON) |
| Personality | Loaded dynamically from `.github/copilot/soul.md` |
| File Support | PDF, DOCX, XLSX, CSV, TXT |

## Files

| File | Role |
|------|------|
| `main.py` | **FastAPI backend** — all API endpoints (upload, query, graph, clear) |
| `templates/index.html` | Single-page frontend UI |
| `static/style.css` | Dark theme styling |
| `static/app.js` | Frontend logic (fetch API, drag-drop, rendering) |
| `extract_graph.py` | LLM-based entity & relation extraction (Claude) |
| `build_graph.py` | In-memory graph construction, N-hop traversal, embedding similarity |
| `query_graphrag.py` | Full query pipeline (graph context + FAISS docs → Claude answer) |
| `.github/copilot/soul.md` | Personality/tone instructions sent as system prompt to Claude |
| `.github/copilot/rules.md` | Code and architecture rules for Copilot |
| `.github/copilot/skills.md` | Project-specific knowledge for Copilot |
| `req.txt` | Python dependencies |
| `.env` | Environment variables (API keys) |

## Prerequisites

### Python 3.10+

### Environment Variables

Create a `.env` file in the project root:

```env
# Login credentials for the web app
POWERPANDA_USERNAME=admin
POWERPANDA_PASSWORD=change-me

# GenAI-Nexus API Key (used for embeddings and Claude)
NEXUS_API_KEY=your-api-key-here

# GenAI-Nexus Base URL
NEXUS_BASE_URL=https://genai-nexus.api.corpinter.net
```

> **Note:** No external database or Docker is required. The knowledge graph is stored in-memory and persisted as JSON on disk.

## Installation

```bash
cd /home/prichai/QuanT/GraphRAG
pip install -r req.txt
```

## Running the Application

```bash
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

## Usage

1. **Upload Files** — Drag & drop or browse for PDF, DOCX, XLSX, CSV, or TXT files.
2. **Process** — Click "⚡ Process Files" to embed documents and build the knowledge graph.
3. **Ask Questions** — Type a query and press "Ask →" (or Ctrl+Enter).
4. **View Results** — See the answer, source files used, and graph context (entity relationships).
5. **Clear Data** — Use the sidebar "🗑 Clear All Data" button to reset everything.

## Configuration (Sidebar)

| Option | Description | Default |
|--------|-------------|---------|
| API Key | Nexus API key (optional if set in `.env`) | from `.env` |
| Graph Name | Graph identifier (used for file naming) | `powerpanda` |
| Traversal Hops | How many hops to traverse from matched nodes | 2 |
| Top-K Docs | Number of document chunks from FAISS | 4 |
| Top-K Nodes | Number of graph nodes matched by embedding | 5 |

## How It Works

### Ingestion

1. Files are parsed into text (PyPDF, python-docx, pandas, plain read).
2. Text is chunked (1000 chars, 200 overlap) and embedded into FAISS via `text-embedding-ada-002`.
3. Full text is sent to Claude Opus for entity/relation extraction (returns structured JSON).
4. Entities → graph nodes, Relations → graph edges, persisted as JSON in `powerpanda_store/`.
5. Duplicate files are detected by MD5 hash and skipped.

### Query

1. Query is embedded using `text-embedding-ada-002`.
2. **Graph path**: Cosine similarity finds top-K nodes → N-hop traversal collects graph context.
3. **Document path**: FAISS similarity search retrieves top-K text chunks.
4. Both contexts + `soul.md` personality are sent to Claude Opus.
5. Answer is returned with source files and graph context shown in the UI.

### Personality (soul.md)

The file `.github/copilot/soul.md` is loaded at query time and sent as the system prompt to Claude. Edit it to change the tone, personality, or response style — no code changes or restart needed.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve the frontend UI |
| `GET` | `/api/files` | List embedded files |
| `POST` | `/api/upload` | Upload and process files |
| `POST` | `/api/query` | Run the query pipeline |
| `GET` | `/api/graph` | Get graph stats and data |
| `DELETE` | `/api/clear` | Delete all stored data |

## Demo Files

| File | Purpose |
|------|---------|
| `SySreq.txt` | System requirements for Pet Mode (On/Off behaviour) |
| `SRS_req.md` | Software architecture with PlantUML sequence diagram |
| `PETMODE_*.csv` | Truth tables for Pet Mode conditions (battery, temp, limp home) |
| `demo_story_1.txt` | Simple demo story (BlueStar Technologies) |
| `demo_story_2.txt` | Connected demo story (SkyDrone Inc) |
| `demo_data.csv` | Structured entity relationships |

## Dependencies

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
jinja2>=3.1.0
aiofiles>=23.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
boto3>=1.28.0
botocore>=1.31.0
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-community>=0.2.0
langchain-text-splitters>=0.2.0
langchain-core>=0.2.0
faiss-cpu>=1.7.4
python-docx>=1.0.0
pandas>=2.0.0
openpyxl>=3.1.0
pypdf>=3.0.0
```
