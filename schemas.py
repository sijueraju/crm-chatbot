"""Pydantic request/response models for the chat API."""

from typing import List, Optional, Tuple

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    history: List[Tuple[str, str]] = []
    order_id: Optional[str] = None  # restrict retrieval to a single order
    customer_id: Optional[str] = None  # reserved for future per-customer scoping


class ChatResponse(BaseModel):
    answer: str
    source_orders: List[str]
