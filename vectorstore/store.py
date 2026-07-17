"""
Thin wrapper around a persistent Chroma collection.

Chroma is used because it runs embedded (no separate server process),
persists to disk, and is free and open source -- a good fit for a local,
free-tier project.
"""
from functools import lru_cache
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from llm.client import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def add_documents(documents: List[Document]) -> int:
    """Adds LangChain Document chunks to the vector store. Returns count added."""
    if not documents:
        return 0
    store = get_vectorstore()
    ids = [f"{doc.metadata.get('source', 'doc')}::chunk-{i}::{hash(doc.page_content) & 0xffffffff}"
           for i, doc in enumerate(documents)]
    store.add_documents(documents=documents, ids=ids)
    return len(documents)


def similarity_search(query: str, k: int = None) -> List[Dict[str, Any]]:
    """Returns top-k chunks with source metadata and similarity score."""
    store = get_vectorstore()
    k = k or settings.TOP_K
    results = store.similarity_search_with_relevance_scores(query, k=k)
    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "score": float(score),
        })
    return output


def count_documents() -> int:
    store = get_vectorstore()
    return store._collection.count()


def list_sources() -> List[Dict[str, Any]]:
    """Returns the distinct source files currently indexed, with chunk counts."""
    store = get_vectorstore()
    raw = store._collection.get(include=["metadatas"])
    metadatas = raw.get("metadatas", []) or []
    counts: Dict[str, int] = {}
    for meta in metadatas:
        source = (meta or {}).get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return [{"source": source, "chunk_count": count} for source, count in sorted(counts.items())]


def reset_collection() -> None:
    """Deletes and recreates the collection. Useful for a clean re-ingest."""
    store = get_vectorstore()
    store._client.delete_collection(settings.CHROMA_COLLECTION_NAME)
    get_vectorstore.cache_clear()
