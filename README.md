# CRM Order RAG Chatbot

A retrieval-augmented chatbot for answering questions about CRM orders stored as PDFs
(invoices/quotes). PDFs are chunked, embedded, and stored in Postgres with `pgvector`;
the chat API retrieves the most relevant chunks for a question and answers using only
that context.

## Architecture

- **`config.py`** — loads `.env` once and exposes every setting (models, thresholds,
  token caps, CORS origins) as module-level constants. No other module reads
  environment variables directly.
- **`schema.sql`** — canonical database schema (see [Database](#database) below).
- **`ingestion.py`** — reads PDFs from `data/orders/`, splits their text into chunks,
  embeds each chunk with `text-embedding-3-small`, and writes them into `orders` /
  `order_chunks`.
- **`database.py`** — the connection pool and the only module that runs SQL against
  `order_chunks`.
- **`rag.py`** — retrieval and generation logic: embeddings/LLM clients, prompts,
  query rewriting, relevance filtering, answer generation.
- **`main.py`** — thin FastAPI app exposing `POST /api/chat`; orchestrates the above,
  with no prompt/model/SQL knowledge of its own. For each question it:
  1. Rewrites the question into a standalone query using conversation history (if any).
  2. Embeds the query and finds the nearest chunks in `order_chunks` via cosine distance.
  3. Discards chunks below a relevance threshold — if none remain, it returns a canned
     "out of scope" response without calling the LLM.
  4. Otherwise, sends the remaining chunks + conversation history to `gpt-4o` and
     returns its answer.
- **`schemas.py`** — `ChatRequest`/`ChatResponse` Pydantic models.
- **`frontend/`** — a small React chat UI that calls the API.

## Database

Two tables, defined in `schema.sql`:

- **`orders`** — one row per order (`order_id` primary key, `source_file`,
  `ingested_at`). Holds order-level metadata that would otherwise be duplicated
  across every chunk belonging to that order.
- **`order_chunks`** — one row per embedded text chunk (`order_id` references
  `orders`, `content`, `metadata` JSONB with `page`/`chunk_index`, `embedding`).
  A single order maps to multiple chunk rows — one PDF is split into several pieces
  so retrieval can return the specific paragraph relevant to a question instead of
  the whole document.

To provision a fresh database on another server:

```bash
psql "$DATABASE_URL" -f schema.sql
```

`ingestion.py` also applies `schema.sql` automatically on startup, so running
`python ingestion.py` against an empty database is sufficient on its own — the
manual `psql` step above is for provisioning the structure ahead of time (e.g. via
an ops/deploy script) without needing Python or the OpenAI API available yet.

## Prerequisites

- Python 3.8+
- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension available
- Node.js (for the frontend)
- An OpenAI API key (used for embeddings and chat completion)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
OPENAI_API_KEY=sk-...
```

Start Postgres and make sure the target database exists, then provision the schema
(see [Database](#database)):

```bash
psql "$DATABASE_URL" -f schema.sql
```

## Ingesting order PDFs

Drop PDF files into `data/orders/` (or point `--dir` / `ORDERS_DIR` elsewhere), then run:

```bash
python ingestion.py
```

Each PDF is treated as one order; the order ID is derived from the filename. Re-running
ingestion for a file replaces any existing chunks for that order.

## Running the API

```bash
uvicorn main:app --reload --port 8000
```

`POST /api/chat`

```json
{
  "message": "What is the price for order 2301142?",
  "history": [["user", "..."], ["assistant", "..."]],
  "order_id": null,
  "customer_id": null
}
```

Returns:

```json
{
  "answer": "...",
  "source_orders": ["..."]
}
```

## Running the frontend

```bash
cd frontend
npm install
npm start
```

Expects the API to be running at `http://localhost:8000` (see `frontend/src`) and is
itself expected at `http://localhost:3000` (see CORS config in `main.py`).

## Known limitations

- **Order IDs come from filenames.** `order_id_from_filename()` only strips a leading
  `order_`/`order-` prefix; PDFs must be named after their order number (e.g.
  `2301142.pdf`) for the order ID to come out clean — a name like
  `offerte_2301142_<timestamp>.pdf` would make the *entire filename stem* the order ID.
- **Memory is session-only.** Conversation history is passed in by the client on every
  request; nothing is persisted server-side across sessions.
- **The relevance threshold is a heuristic**, calibrated against the current dataset's
  embedding distances. It may need retuning (`MAX_RELEVANT_DISTANCE` env var) if very
  different content is ingested.
