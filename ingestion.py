# ingestion.py
# Reads order PDFs from a directory, chunks and embeds their text, and
# upserts the result into the order_chunks table that main.py queries.
# Each PDF is one order; the order_id is taken from its filename.
#
# Run with: python ingestion.py [--dir data/orders]

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before anything reads env vars

import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 100
DEFAULT_ORDERS_DIR = os.getenv("ORDERS_DIR", "data/orders")

ORDER_ID_PREFIX_RE = re.compile(r"^order[_-]?", re.IGNORECASE)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def order_id_from_filename(path: Path) -> str:
    return ORDER_ID_PREFIX_RE.sub("", path.stem)


def extract_pages(path: Path) -> list:
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def chunk_pdf(path: Path) -> list:
    """Returns a list of {order_id, content, metadata} dicts for one PDF."""
    order_id = order_id_from_filename(path)
    chunks = []
    for page_num, page_text in enumerate(extract_pages(path), start=1):
        if not page_text.strip():
            continue
        for i, piece in enumerate(splitter.split_text(page_text)):
            chunks.append(
                {
                    "order_id": order_id,
                    "content": piece,
                    "metadata": {
                        "source_file": path.name,
                        "page": page_num,
                        "chunk_index": i,
                    },
                }
            )
    return chunks


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS order_chunks (
                id SERIAL PRIMARY KEY,
                order_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector({EMBEDDING_DIM})
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS order_chunks_order_id_idx "
            "ON order_chunks (order_id);"
        )
    conn.commit()


def embed_texts(texts: list) -> list:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vectors.extend(embeddings.embed_documents(batch))
    return vectors


def ingest_file(conn, path: Path) -> int:
    chunks = chunk_pdf(path)
    if not chunks:
        print(f"  skip {path.name}: no extractable text")
        return 0

    order_id = chunks[0]["order_id"]
    vectors = embed_texts([c["content"] for c in chunks])

    with conn.cursor() as cur:
        cur.execute("DELETE FROM order_chunks WHERE order_id = %s;", (order_id,))
        cur.executemany(
            """
            INSERT INTO order_chunks (order_id, content, metadata, embedding)
            VALUES (%s, %s, %s, %s);
            """,
            [
                (c["order_id"], c["content"], Json(c["metadata"]), vec)
                for c, vec in zip(chunks, vectors)
            ],
        )
    conn.commit()
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest order PDFs into the order_chunks table."
    )
    parser.add_argument(
        "--dir",
        default=DEFAULT_ORDERS_DIR,
        help=f"Directory containing order PDFs (default: {DEFAULT_ORDERS_DIR})",
    )
    args = parser.parse_args()

    orders_dir = Path(args.dir)
    pdf_paths = sorted(orders_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {orders_dir}")
        return

    conn = psycopg2.connect(dsn=DATABASE_URL)
    try:
        ensure_schema(conn)
        register_vector(conn)

        total_chunks = 0
        failed = []
        for path in pdf_paths:
            print(f"Ingesting {path.name} (order_id={order_id_from_filename(path)})...")
            try:
                total_chunks += ingest_file(conn, path)
            except Exception as exc:
                conn.rollback()
                failed.append(path.name)
                print(f"  FAILED: {exc}")

        print(
            f"\nDone. {len(pdf_paths) - len(failed)}/{len(pdf_paths)} files ingested, "
            f"{total_chunks} chunks written."
        )
        if failed:
            print(f"Failed files: {', '.join(failed)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
