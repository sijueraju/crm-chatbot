"""Centralized configuration for the CRM order chatbot.

Loads environment variables once and exposes them as module-level constants,
so no other module needs to call `load_dotenv()` or `os.getenv()` directly.
Import order matters here: this module must be imported before anything else
that depends on these values.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Secrets / connection info ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
REWRITE_MODEL = "gpt-4o-mini"  # cheaper model used only to rewrite follow-up questions

# --- Retrieval ---
TOP_K = 5

# Cosine distance (pgvector `<=>`) above which a match is considered
# irrelevant. Calibrated empirically: on-topic order questions against this
# dataset land at ~0.51-0.70, clearly off-topic ones (code requests, trivia,
# jokes) land at ~0.86-0.90. 0.78 sits in the unused gap between them.
MAX_RELEVANT_DISTANCE = float(os.getenv("MAX_RELEVANT_DISTANCE", "0.78"))

# --- Generation ---
CHAT_MAX_TOKENS = 500
REWRITE_MAX_TOKENS = 100

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with questions about orders in this CRM system "
    "(e.g. price, items, delivery date, payment terms). "
    "Please ask about a specific order."
)

# --- API ---
CORS_ALLOW_ORIGINS = ["http://localhost:3000"]
