"""
FastAPI application exposing the RAG workflow.

Endpoints:
    POST /query       - ask a question, get an answer + sources
    POST /ingest       - ingest new documents (file uploads or URLs)
    GET  /documents    - list what's currently indexed
    POST /feedback     - submit thumbs up/down + optional comment
    GET  /health       - basic health check
"""
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from ingest import ingest_directory, ingest_paths, ingest_urls
from vectorstore.store import count_documents, list_sources
from graph.workflow import run_query


# --------------------------------------------------------------------------
# Bonus: conversation memory.
#
# Kept intentionally simple -- an in-memory dict of session_id -> list of
# (question, answer) tuples. This resets on server restart and isn't shared
# across multiple processes/workers, which is a fine tradeoff for a local /
# single-instance demo but wouldn't scale past that (see README).
#
# OrderedDict + the eviction check in `_remember_turn` keeps memory bounded:
# once MAX_SESSIONS_IN_MEMORY is hit, the oldest session is dropped.
# --------------------------------------------------------------------------
_session_history: "OrderedDict[str, list]" = OrderedDict()


def _get_history(session_id: Optional[str]) -> list:
    if not session_id:
        return []
    return _session_history.get(session_id, [])


def _remember_turn(session_id: Optional[str], question: str, answer: Optional[str]):
    if not session_id or answer is None:
        return
    history = _session_history.setdefault(session_id, [])
    history.append((question, answer))
    if len(history) > settings.MAX_HISTORY_TURNS:
        del history[: len(history) - settings.MAX_HISTORY_TURNS]
    _session_history.move_to_end(session_id)
    while len(_session_history) > settings.MAX_SESSIONS_IN_MEMORY:
        _session_history.popitem(last=False)


# --------------------------------------------------------------------------
# Feedback storage (SQLite - free, zero-setup, file-based)
# --------------------------------------------------------------------------
def _init_feedback_db():
    conn = sqlite3.connect(settings.FEEDBACK_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT,
            rating TEXT NOT NULL,
            comment TEXT,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_feedback_db()
    # Auto-ingest the bundled corpus on first run so the API is usable
    # immediately after `uvicorn main:app`, and so it also self-heals on
    # free hosting platforms whose disks reset between deploys.
    try:
        if count_documents() == 0:
            print("Vector store is empty -- auto-ingesting bundled corpus...")
            result = ingest_directory(settings.CORPUS_DIR)
            print(f"Auto-ingest complete: {result}")
    except Exception as e:
        print(f"Auto-ingest skipped due to error: {e}")
    yield


app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        default=None,
        description="Optional. Pass the session_id returned from a previous /query "
                     "call to let follow-up questions reference earlier turns. "
                     "Omit it (or send null) for a stateless, one-off question.",
    )


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str]
    sources: List[dict]
    query_type: Optional[str] = None
    retries_used: int = 0
    used_web_search: bool = False
    grounded: Optional[bool] = None
    session_id: str


class IngestUrlsRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1)


class FeedbackRequest(BaseModel):
    question: str
    answer: Optional[str] = None
    rating: str = Field(..., pattern="^(up|down)$")
    comment: Optional[str] = None


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": count_documents()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not settings.GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the server. "
                   "Add a free key from https://console.groq.com/keys to .env",
        )
    session_id = req.session_id or str(uuid.uuid4())
    try:
        history = _get_history(session_id)
        result = run_query(req.question, chat_history=history)
        _remember_turn(session_id, req.question, result.get("answer"))
        result["session_id"] = session_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.post("/ingest")
async def ingest(
    files: Optional[List[UploadFile]] = File(default=None),
):
    """
    Accepts either:
    - multipart file uploads (.md / .txt), or
    - a JSON body of URLs via a separate call to /ingest/urls (see below).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided. Use /ingest/urls for URL ingestion.")

    tmp_dir = tempfile.mkdtemp()
    saved_paths = []
    try:
        for f in files:
            if not f.filename.lower().endswith((".md", ".txt")):
                continue
            dest = os.path.join(tmp_dir, f.filename)
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved_paths.append(dest)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="Only .md and .txt files are supported.")

        result = ingest_paths(saved_paths)
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/ingest/urls")
def ingest_from_urls(req: IngestUrlsRequest):
    try:
        save_dir = os.path.join(settings.CORPUS_DIR, "from_url")
        result = ingest_urls(req.urls, save_dir=save_dir)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL ingestion failed: {e}")


@app.get("/documents")
def documents():
    return {"total_chunks": count_documents(), "sources": list_sources()}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    conn = sqlite3.connect(settings.FEEDBACK_DB_PATH)
    conn.execute(
        "INSERT INTO feedback (question, answer, rating, comment, created_at) VALUES (?, ?, ?, ?, ?)",
        (req.question, req.answer, req.rating, req.comment, time.time()),
    )
    conn.commit()
    conn.close()
    return {"status": "recorded"}


@app.get("/feedback")
def list_feedback():
    conn = sqlite3.connect(settings.FEEDBACK_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM feedback ORDER BY created_at DESC LIMIT 200").fetchall()
    conn.close()
    return {"feedback": [dict(r) for r in rows]}