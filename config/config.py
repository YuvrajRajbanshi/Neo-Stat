"""
Configuration module for the Smart AI Chatbot.
Loads API keys and settings from environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file.
# override=True prevents stale shell-level variables from forcing old model names.
load_dotenv(override=True)

# xAI (Grok) Configuration
XAI_API_KEY = os.getenv("XAI_API_KEY", os.getenv("GROK_API_KEY", ""))
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")

# Model Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "grok-beta")
LLM_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("LLM_FALLBACK_MODELS", "grok-beta,grok-2-1212,grok-2").split(",")
    if m.strip()
]
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

# Web Search Configuration
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "3"))


def validate_config():
    """Validate that required configuration values are set."""
    errors = []

    if not XAI_API_KEY:
        errors.append("XAI_API_KEY is not set. Please set it in your .env file.")

    return errors


def get_config_summary():
    """Return a summary of current configuration (without sensitive data)."""
    return {
        "llm_model": LLM_MODEL,
        "llm_fallback_models": LLM_FALLBACK_MODELS,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_base_url": XAI_BASE_URL,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_k": RETRIEVAL_K,
        "api_key_set": bool(XAI_API_KEY)
    }
