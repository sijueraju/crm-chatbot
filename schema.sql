-- Schema for the CRM Order RAG Chatbot.
--
-- Canonical source of truth for the database structure. Run this directly
-- against a fresh database to provision a new environment:
--
--   psql "$DATABASE_URL" -f schema.sql
--
-- ingestion.py also executes this file automatically on startup (see
-- ensure_schema()), so a plain `python ingestion.py` on a fresh database
-- is sufficient too — this file just makes the structure inspectable and
-- runnable independently of the Python code.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per order. Holds order-level metadata that would otherwise be
-- duplicated across every chunk belonging to that order (e.g. source_file).
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per embedded text chunk. embedding dimension (1536) matches
-- OpenAI's text-embedding-3-small — update both here and in config.py if
-- the embedding model ever changes.
CREATE TABLE IF NOT EXISTS order_chunks (
    id SERIAL PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536)
);

CREATE INDEX IF NOT EXISTS order_chunks_order_id_idx ON order_chunks (order_id);
