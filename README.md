# PowerPanda

PowerPanda is a GraphRAG application built with FastAPI. It combines:
- vector retrieval from FAISS
- graph traversal over extracted entities and relations
- OpenAI models for extraction, embeddings, and answer generation

The app supports document ingestion from PDF, DOCX, XLSX, CSV, and TXT files, then answers questions using both document context and graph context.

## Recent Updates

- Login is now username/password based and reads credentials from environment variables.
- OpenAI API key is now environment-only in the web app flow.
- API key input has been removed from the UI to prevent accidental overrides.
- Template loading supports both `templates/` and `template/` directory names.
- Added test stories for graph behavior validation: `story1.txt`, `story2.txt`, `story3.txt`.

## Architecture

### Ingestion Pipeline

1. User uploads files through the web UI.
2. Files are parsed into text chunks.
3. Chunks are embedded using OpenAI embeddings and stored in FAISS.
4. Full text is sent to OpenAI for entity/relation extraction.
5. Entities and relations are persisted as an in-memory graph snapshot in `powerpanda_store/`.

### Query Pipeline

1. User asks a question.
2. Relevant graph nodes are found by embedding similarity.
3. N-hop graph traversal builds graph context.
4. FAISS retrieves top matching document chunks.
5. A combined prompt is sent to OpenAI chat completion.
6. Response is returned with sources and graph context.

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| LLM | OpenAI Chat Completions (`OPENAI_MODEL`) |
| Embeddings | OpenAI Embeddings (`OPENAI_EMBED_MODEL`) |
| Vector Store | FAISS |
| Graph Store | In-memory graph persisted to JSON |
| File Parsing | PyPDF, python-docx, pandas/openpyxl |

## Project Structure

| Path | Purpose |
|---|---|
| `main.py` | FastAPI app and API routes |
| `extract_graph.py` | Entity and relation extraction |
| `build_graph.py` | Graph build, traversal, node similarity |
| `query_graphrag.py` | Query orchestration and answer generation |
| `template/login.html` | Login page |
| `template/index.html` | Main application page |
| `template/graph_viewer.html` | Graph visualization page |
| `static/app.js` | Frontend app logic |
| `static/style.css` | Frontend styling |
| `req.txt` | Python dependencies |
| `render.yaml` | Render deployment config |
| `Procfile` | Process entrypoint |
| `powerpanda_store/` | Generated FAISS and graph artifacts |

## Requirements

- Python 3.10+
- OpenAI API key

## Environment Variables

Create a `.env` file in the project root for local development:

```env
# Required for OpenAI calls
OPENAI_API_KEY=your_openai_api_key

# Optional model settings
OPENAI_MODEL=gpt-4o
OPENAI_EMBED_MODEL=text-embedding-ada-002

# App login credentials
POWERPANDA_USERNAME=admin
POWERPANDA_PASSWORD=change-me
```

Notes:
- In the FastAPI app flow, OpenAI key is read from `OPENAI_API_KEY` in environment only.
- The UI no longer sends an API key.
- If `POWERPANDA_USERNAME` or `POWERPANDA_PASSWORD` are missing, app falls back to defaults.

## Install and Run Locally

```bash
cd /home/chaitanya/PowerPanda
pip install -r req.txt
uvicorn main:app --reload --port 8000
```

Open: http://localhost:8000

## Render Deployment

Current Render start command:

```yaml
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set these environment variables in Render service settings:

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, default `gpt-4o`)
- `OPENAI_EMBED_MODEL` (optional, default `text-embedding-ada-002`)
- `POWERPANDA_USERNAME` (recommended)
- `POWERPANDA_PASSWORD` (recommended)

## Usage

1. Open `/` and log in.
2. Upload one or more files.
3. Click Process Files.
4. Ask questions in the query panel.
5. Optionally view graph in `/graph-viewer`.
6. Use Clear All Data to reset FAISS and graph artifacts.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Login page |
| `POST` | `/api/login` | Username/password authentication |
| `GET` | `/app` | Main application page |
| `GET` | `/graph-viewer` | Graph visualization page |
| `GET` | `/api/files` | List embedded files |
| `POST` | `/api/upload` | Upload and process files |
| `POST` | `/api/query` | Ask question over GraphRAG pipeline |
| `GET` | `/api/graph` | Return mermaid graph + stats |
| `DELETE` | `/api/clear` | Clear stored FAISS and graph data |

## Demo and Test Files

- `story1.txt`
- `story2.txt`
- `story3.txt`

These stories are designed with overlapping entities so you can validate:
- cross-document entity linking
- graph traversal quality
- hybrid retrieval behavior

## Troubleshooting

### 401 invalid_api_key from OpenAI

- Verify `OPENAI_API_KEY` in Render/local environment.
- Ensure no old or incorrect secret value is set in the deployment environment.
- Redeploy after changing environment variables.

### Upload or query fails with missing key message

- Confirm `OPENAI_API_KEY` is set for the running process.
- Restart or redeploy after updating environment variables.

### Graph appears empty

- Make sure files were processed successfully.
- Check `/api/files` and `/api/graph` responses.
- Re-upload sample stories and run a query.

## Dependencies

See `req.txt` for the exact dependency list used by this project.
