# CRM Order RAG Chatbot

A retrieval-augmented chatbot for answering questions about CRM orders stored as PDFs
(invoices/quotes). PDFs are chunked, embedded, and stored in Postgres with `pgvector`;
the chat API retrieves the most relevant chunks for a question and answers using only
that context.

## Architecture

- **`ingestion.py`** — reads PDFs from `data/orders/`, splits their text into chunks,
  embeds each chunk with `text-embedding-3-small`, and upserts them into the
  `order_chunks` table (one row per chunk, keyed by `order_id`).
- **`main.py`** — FastAPI app exposing `POST /api/chat`. For each question it:
  1. Rewrites the question into a standalone query using conversation history (if any).
  2. Embeds the query and finds the nearest chunks in `order_chunks` via cosine distance.
  3. Discards chunks below a relevance threshold — if none remain, it returns a canned
     "out of scope" response without calling the LLM.
  4. Otherwise, sends the remaining chunks + conversation history to `gpt-4o` and
     returns its answer.
- **`frontend/`** — a small React chat UI that calls the API.

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

Start Postgres and make sure the target database exists. `ingestion.py` creates the
`vector` extension and `order_chunks` table automatically on first run.

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
  `order_`/`order-` prefix; for other naming schemes (e.g. `offerte_<id>_<timestamp>.pdf`)
  the *entire filename stem* becomes the order ID, not just the numeric order number.
- **Memory is session-only.** Conversation history is passed in by the client on every
  request; nothing is persisted server-side across sessions.
- **The relevance threshold is a heuristic**, calibrated against the current dataset's
  embedding distances. It may need retuning (`MAX_RELEVANT_DISTANCE` env var) if very
  different content is ingested.
