"""FastAPI entrypoint for the CRM Order RAG Chatbot.

Wires together config, the Postgres connection pool, and the retrieval/
generation logic in rag.py. Business logic lives in rag.py and database.py;
this module only orchestrates the request lifecycle.

Run with: uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ALLOW_ORIGINS, OUT_OF_SCOPE_MESSAGE, TOP_K
from database import close_pool, init_pool
from rag import (
    build_context,
    contextualize_query,
    generate_answer,
    infer_order_id,
    retrieve_relevant_chunks,
)
from schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool()
    yield
    close_pool()


app = FastAPI(title="CRM Order RAG Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "crm-rag-chatbot"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    search_query = contextualize_query(req.message, req.history)
    order_id = req.order_id or infer_order_id(search_query)
    chunks = retrieve_relevant_chunks(search_query, k=TOP_K, order_id=order_id)

    if not chunks:
        return ChatResponse(answer=OUT_OF_SCOPE_MESSAGE, source_orders=[])

    context = build_context(chunks)
    answer = generate_answer(question=req.message, history=req.history, context=context)

    return ChatResponse(answer=answer, source_orders=list({c["order_id"] for c in chunks}))
