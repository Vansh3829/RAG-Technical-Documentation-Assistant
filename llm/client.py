"""
Factory functions for the LLM and the embedding model.

Both are free-tier:
- Groq gives fast Llama-3.1 inference with a generous free tier.
- sentence-transformers embeddings run locally on CPU, no API key, no cost.

Kept behind small wrapper functions (with lru_cache) so the rest of the
codebase never has to know which concrete provider is being used.
"""
from functools import lru_cache

from config import settings


@lru_cache(maxsize=1)
def get_llm():
    """Returns a chat LLM client. Swap this function to change providers."""
    from langchain_groq import ChatGroq

    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and add it to your .env file."
        )

    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Returns a local, free embedding model (no API key required)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
