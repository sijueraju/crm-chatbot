"""Retrieval and generation logic: embeddings, prompts, query rewriting,
relevance filtering, and answer generation.

`main.py` only orchestrates these calls — it has no knowledge of prompts,
models, or the relevance threshold.
"""

import re
from typing import List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (
    CHAT_MAX_TOKENS,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    MAX_RELEVANT_DISTANCE,
    OPENAI_API_KEY,
    REWRITE_MAX_TOKENS,
    REWRITE_MODEL,
)
from database import fetch_nearest_chunks, known_order_ids

# Order IDs in this dataset are 5-8 digit numbers (e.g. 2301142, 250000).
ORDER_ID_CANDIDATE_RE = re.compile(r"\d{5,8}")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
chat_llm = ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, max_tokens=CHAT_MAX_TOKENS)

# Cheaper/smaller model for the retrieval-query rewrite step — this call
# only reformulates a question, it never generates the user-facing answer.
rewrite_llm = ChatOpenAI(model=REWRITE_MODEL, api_key=OPENAI_API_KEY, max_tokens=REWRITE_MAX_TOKENS)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant for a CRM order management system.
Answer the user's question using only the order information provided below.
If the information is not present in the context, say you don't have enough
information to answer — do not guess.

You must only answer questions about the orders in the context. Do not
write code, answer general-knowledge questions, or perform any task
unrelated to the order data below, even if directly asked — refuse those
and state that you can only help with order-related questions.

These documents are Swiss/German business quotes and invoices. A line like
"Offerte 2301142 Gossau, 25. Juni 2025" or "Rechnung 2500024 Gossau, 29.
Oktober 2025" states the issuing city and the document's creation/issue
date directly after the document type and number — there is no separate
"Datum:"/"Created:" label. Treat that date as the order's created/issued
date when asked, rather than saying it's missing.

Conversation history (most recent last):
{history}

Context:
{context}

Question: {question}"""
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
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

    messages = REWRITE_PROMPT.format_messages(
        history=format_history(history), question=message
    )
    response = rewrite_llm.invoke(messages)
    return response.content.strip()


def infer_order_id(message: str) -> Optional[str]:
    """Looks for a number in the message that matches a known order_id.

    Without this, retrieval falls back to an unfiltered global search, and a
    chunk from an unrelated order can outrank the correct order's own chunks
    (e.g. a payment-slip chunk containing literal "CHF"/"Betrag" tokens can
    outrank the actual price table on the order the user asked about).
    """
    candidates = ORDER_ID_CANDIDATE_RE.findall(message)
    if not candidates:
        return None

    valid_ids = known_order_ids()
    for candidate in candidates:
        if candidate in valid_ids:
            return candidate
    return None


def retrieve_relevant_chunks(query: str, k: int, order_id: Optional[str] = None) -> list:
    """Embeds the query, fetches the k nearest chunks, and discards any
    below MAX_RELEVANT_DISTANCE — the caller sees only genuinely relevant
    matches, never raw nearest-neighbor noise."""
    query_vector = embeddings.embed_query(query)
    chunks = fetch_nearest_chunks(query_vector, k=k, order_id=order_id)
    return [c for c in chunks if c["distance"] <= MAX_RELEVANT_DISTANCE]


def build_context(chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[Order {c['order_id']}]\n{c['content']}" for c in chunks
    )


def generate_answer(question: str, history: List[Tuple[str, str]], context: str) -> str:
    messages = ANSWER_PROMPT.format_messages(
        history=format_history(history), context=context, question=question
    )
    response = chat_llm.invoke(messages)
    return response.content
