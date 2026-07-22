"""Postgres/pgvector connection pool and raw chunk retrieval.

This is the only module that talks to `order_chunks` directly. Both
`ingestion.py` (writes) and `rag.py` (reads, via `fetch_nearest_chunks`)
go through the same table with no ORM/vectorstore wrapper in between.
"""

from typing import Optional

import psycopg2
from psycopg2 import pool as pg_pool
from pgvector.psycopg2 import register_vector

from config import DATABASE_URL

_pool: Optional[pg_pool.SimpleConnectionPool] = None


def init_pool(minconn: int = 1, maxconn: int = 10) -> None:
    global _pool
    _pool = psycopg2.pool.SimpleConnectionPool(minconn, maxconn, dsn=DATABASE_URL)


def close_pool() -> None:
    if _pool is not None:
        _pool.closeall()


def fetch_nearest_chunks(query_vector, k: int, order_id: Optional[str] = None) -> list:
    """Runs a cosine-distance nearest-neighbor search against order_chunks.

    Always returns exactly `k` rows (or fewer if the table/order has less
    data) ordered by distance — callers are responsible for deciding which
    of those matches are actually relevant.
    """
    conn = _pool.getconn()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            if order_id:
                cur.execute(
                    """
                    SELECT order_id, content, metadata,
                           embedding <=> %s::vector AS distance
                    FROM order_chunks
                    WHERE order_id = %s
                    ORDER BY distance
                    LIMIT %s;
                    """,
                    (query_vector, order_id, k),
                )
            else:
                cur.execute(
                    """
                    SELECT order_id, content, metadata,
                           embedding <=> %s::vector AS distance
                    FROM order_chunks
                    ORDER BY distance
                    LIMIT %s;
                    """,
                    (query_vector, k),
                )
            rows = cur.fetchall()
    finally:
        _pool.putconn(conn)

    return [
        {"order_id": r[0], "content": r[1], "metadata": r[2], "distance": r[3]}
        for r in rows
    ]


def known_order_ids() -> set:
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT order_id FROM orders;")
            return {row[0] for row in cur.fetchall()}
    finally:
        _pool.putconn(conn)
