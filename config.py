"""
Centralized application configuration.

All values can be overridden via environment variables or a `.env` file
in the project root. See `.env.example` for the full list of options.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM provider (Groq is free-tier friendly and fast) ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    LLM_TEMPERATURE: float = 0.0

    # --- Embeddings (fully local, no API key, no cost) ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store ---
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "tech_docs"

    # --- Retrieval / ingestion ---
    CORPUS_DIR: str = "./data/corpus"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 4

    # --- Self-corrective workflow ---
    MAX_RETRIES: int = 2
    RELEVANCE_SCORE_THRESHOLD: float = 0.0  # reserved for future numeric grading

    # --- Bonus: web search fallback (Tavily free tier) ---
    TAVILY_API_KEY: str = ""
    ENABLE_WEB_SEARCH_FALLBACK: bool = True

    # --- Bonus: conversation memory ---
    MAX_HISTORY_TURNS: int = 3  # how many past Q&A pairs to keep per session
    MAX_SESSIONS_IN_MEMORY: int = 500  # simple cap so the in-memory store can't grow unbounded

    # --- Feedback storage ---
    FEEDBACK_DB_PATH: str = "./feedback.db"

    # --- API ---
    API_TITLE: str = "RAG Technical Documentation Assistant"
    API_VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()