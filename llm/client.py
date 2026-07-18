"""
Factory functions for the LLM and the embedding model.

Both are free-tier:
- Groq gives fast Llama-3.1 inference with a generous free tier.
- fastembed runs embeddings locally on CPU via ONNX Runtime -- no API key, no
  cost, and (unlike sentence-transformers/PyTorch) a small enough memory
  footprint to fit in a 512MB free-tier container like Render's.

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
    """
    Returns a local, free embedding model (no API key required).

    Uses fastembed (ONNX Runtime) directly, via a small wrapper implementing
    LangChain's Embeddings interface -- instead of
    langchain_community.embeddings.FastEmbedEmbeddings, which has a known
    bug in some fastembed/pydantic version combinations where its internal
    model attribute never gets initialized (`_model` stays None). Direct and
    simple beats fighting that wrapper's private-attribute handling.

    fastembed keeps memory usage low enough to fit in a 512MB free-tier
    container (e.g. Render) -- sentence-transformers/PyTorch alone can blow
    past that limit.
    """
    from fastembed import TextEmbedding
    from langchain_core.embeddings import Embeddings

    class _FastEmbedWrapper(Embeddings):
        def __init__(self, model_name: str):
            # threads=1: keeps ONNX Runtime's memory/CPU overhead low, which
            # matters on a constrained single-core free-tier container.
            self._model = TextEmbedding(model_name=model_name, threads=1)

        def embed_documents(self, texts):
            return [vec.tolist() for vec in self._model.embed(texts)]

        def embed_query(self, text):
            return next(iter(self._model.embed([text]))).tolist()

    return _FastEmbedWrapper(settings.EMBEDDING_MODEL)