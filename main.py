# main.py
# FastAPI entrypoint for the CRM Order RAG Chatbot.
# Queries the custom `order_chunks` table directly via psycopg2 — no
# LangChain vectorstore wrapper involved, so retrieval reads from exactly
# the same table that ingestion.py writes to.
#
# Run with: uvicorn main:app --reload --port 8000

from dotenv import load_dotenv
import os

load_dotenv()  # must run before anything reads env vars

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

from typing import List, Tuple, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import psycopg2
from psycopg2 import pool
from pgvector.psycopg2 import register_vector

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5

# Cosine distance (pgvector `<=>`) above which a match is considered
# irrelevant. Calibrated empirically: on-topic order questions against this
# dataset land at ~0.51-0.70, clearly off-topic ones (code requests, trivia,
# jokes) land at ~0.86-0.90. 0.78 sits in the unused gap between them.
MAX_RELEVANT_DISTANCE = float(os.getenv("MAX_RELEVANT_DISTANCE", "0.78"))

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with questions about orders in this CRM system "
    "(e.g. price, items, delivery date, payment terms). "
    "Please ask about a specific order."
)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, max_tokens=500)

# Cheaper/smaller model for the retrieval-query rewrite step — this call
# only reformulates a question, it never generates the user-facing answer.
rewrite_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, max_tokens=100)

prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful assistant for a CRM order management system.
Answer the user's question using only the order information provided below.
If the information is not present in the context, say you don't have enough
information to answer — do not guess.

You must only answer questions about the orders in the context. Do not
write code, answer general-knowledge questions, or perform any task
unrelated to the order data below, even if directly asked — refuse those
and state that you can only help with order-related questions.

Conversation history (most recent last):
{history}

Context:
{context}

Question: {question}"""
)

rewrite_prompt = ChatPromptTemplate.from_template(
    """Given the conversation history and a follow-up question, rewrite the
follow-up question as a standalone question that includes any context
(such as order numbers) implied by the history. If the follow-up question
is already standalone, return it unchanged. Reply with only the rewritten
question — no explanation.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""
)

# --- Connection pool, created once at startup ---
db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, dsn=DATABASE_URL)
    yield
    db_pool.closeall()


app = FastAPI(title="CRM Order RAG Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: List[Tuple[str, str]] = []
    order_id: Optional[str] = None  # optional: restrict search to one order
    customer_id: Optional[str] = None  # optional: restrict to one customer's orders


def retrieve_chunks(query: str, k: int = TOP_K, order_id: Optional[str] = None):
    """Embeds the query and runs a cosine-distance nearest-neighbor search
    directly against order_chunks. Returns the top-k rows."""
    query_vector = embeddings.embed_query(query)

    conn = db_pool.getconn()
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
        db_pool.putconn(conn)

    return [
        {"order_id": r[0], "content": r[1], "metadata": r[2], "distance": r[3]}
        for r in rows
    ]


def build_context(chunks) -> str:
    return "\n\n---\n\n".join(
        f"[Order {c['order_id']}]\n{c['content']}" for c in chunks
    )


def format_history(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{role}: {content}" for role, content in history)


def contextualize_query(message: str, history: List[Tuple[str, str]]) -> str:
    """Rewrites a follow-up question into a standalone one using the
    conversation history, so retrieval isn't run on bare pronouns/ellipsis
    like "and the delivery date?". Skipped on the first turn of a
    conversation, since there's no history yet to rewrite with."""
    if not history:
        return message

    messages = rewrite_prompt.format_messages(
        history=format_history(history), question=message
    )
    response = rewrite_llm.invoke(messages)
    return response.content.strip()


@app.get("/")
async def root():
    return {"status": "ok", "service": "crm-rag-chatbot"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    search_query = contextualize_query(req.message, req.history)

    chunks = retrieve_chunks(search_query, k=TOP_K, order_id=req.order_id)
    relevant_chunks = [c for c in chunks if c["distance"] <= MAX_RELEVANT_DISTANCE]

    if not relevant_chunks:
        return {"answer": OUT_OF_SCOPE_MESSAGE, "source_orders": []}

    context = build_context(relevant_chunks)

    messages = prompt_template.format_messages(
        history=format_history(req.history), context=context, question=req.message
    )
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "source_orders": list({c["order_id"] for c in relevant_chunks}),
    }