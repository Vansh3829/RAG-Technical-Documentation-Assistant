"""
Node and conditional-edge (routing) functions for the self-corrective RAG
workflow. Each node function takes the current `GraphState` and returns a
dict of the fields it wants to update -- LangGraph merges this into state.
"""
import json
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from graph.state import GraphState
from llm.client import get_llm
from vectorstore.store import similarity_search

# --------------------------------------------------------------------------
# Node 1: Query Analysis
# --------------------------------------------------------------------------
_QUERY_ANALYSIS_PROMPT = """You are a query analysis assistant for a technical \
documentation search engine covering FastAPI, Pydantic, LangGraph, ChromaDB, \
and the Python Requests library.

{history_block}Given the user's raw question, do two things:
1. Rewrite it into a clear, keyword-rich search query that will work well \
against a vector similarity search over technical docs (expand abbreviations, \
add likely synonyms, resolve ambiguous pronouns -- if the question refers to \
something from the conversation history above, like "it" or "that library", \
resolve it to the actual name using the history). Keep it a single sentence.
2. Classify the question into exactly one of: "conceptual", "how-to", \
"troubleshooting", "api-reference".

Respond ONLY with compact JSON, no markdown fences, in this exact shape:
{{"rewritten_query": "...", "query_type": "..."}}

User question: {question}"""


def _format_history(chat_history: list) -> str:
    if not chat_history:
        return ""
    lines = ["Recent conversation history (oldest first):"]
    for q, a in chat_history:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines) + "\n\n"


def analyze_query(state: GraphState) -> dict:
    question = state["question"]
    history_block = _format_history(state.get("chat_history", []))
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=_QUERY_ANALYSIS_PROMPT.format(
        question=question, history_block=history_block,
    ))])
    rewritten, qtype = question, "conceptual"
    try:
        parsed = json.loads(_strip_fences(response.content))
        rewritten = parsed.get("rewritten_query", question) or question
        qtype = parsed.get("query_type", "conceptual") or "conceptual"
    except Exception:
        pass  # fall back to the original question if the LLM didn't return valid JSON

    return {
        "original_question": question,
        "question": rewritten,
        "query_type": qtype,
        "retry_count": 0,
        "max_retries": settings.MAX_RETRIES,
        "used_web_search": False,
        "regenerated": False,
    }


# --------------------------------------------------------------------------
# Node 2: Retrieval
# --------------------------------------------------------------------------
def retrieve(state: GraphState) -> dict:
    results = similarity_search(state["question"], k=settings.TOP_K)
    documents = [{**r, "relevant": None} for r in results]
    return {"documents": documents}


# --------------------------------------------------------------------------
# Node 3: Document Grading (self-corrective component)
# --------------------------------------------------------------------------
_GRADING_PROMPT = """You are grading whether a retrieved document chunk is \
relevant to a user's question. Only answer "yes" if the chunk contains \
information that would directly help answer the question. Answer "no" \
otherwise. Respond with a single word: yes or no.

Question: {question}

Document chunk:
\"\"\"
{content}
\"\"\""""


def grade_documents(state: GraphState) -> dict:
    llm = get_llm()
    question = state["original_question"]
    graded: List[dict] = []
    for doc in state.get("documents", []):
        prompt = _GRADING_PROMPT.format(question=question, content=doc["content"][:2000])
        response = llm.invoke([HumanMessage(content=prompt)])
        verdict = response.content.strip().lower()
        is_relevant = verdict.startswith("yes")
        graded.append({**doc, "relevant": is_relevant})

    relevant_docs = [d for d in graded if d["relevant"]]
    return {"documents": graded, "graded_documents": relevant_docs}


def decide_to_generate(state: GraphState) -> str:
    """Conditional edge: routes based on the document grading outcome."""
    if state.get("graded_documents"):
        return "generate"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.MAX_RETRIES)
    if retry_count < max_retries:
        return "rewrite"

    if settings.ENABLE_WEB_SEARCH_FALLBACK and settings.TAVILY_API_KEY:
        return "web_search"

    return "give_up"


# --------------------------------------------------------------------------
# Rewrite-and-retry node
# --------------------------------------------------------------------------
_REWRITE_PROMPT = """The following search query returned no relevant results \
from a technical documentation vector store: "{question}"

Original user question: "{original_question}"

Rewrite the query differently: try alternate phrasing, more general terms, or \
different keywords that might match the documentation better. Respond with \
ONLY the new search query text, nothing else."""


def rewrite_query(state: GraphState) -> dict:
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=_REWRITE_PROMPT.format(
        question=state["question"], original_question=state["original_question"],
    ))])
    new_query = response.content.strip().strip('"')
    return {
        "question": new_query or state["question"],
        "retry_count": state.get("retry_count", 0) + 1,
    }


# --------------------------------------------------------------------------
# Bonus node: Web search fallback (Tavily free tier)
# --------------------------------------------------------------------------
def web_search(state: GraphState) -> dict:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        results = client.search(state["original_question"], max_results=settings.TOP_K)
        docs = [
            {
                "content": r.get("content", ""),
                "source": r.get("url", "web-search"),
                "chunk_index": None,
                "score": r.get("score", 0.0),
                "relevant": True,
            }
            for r in results.get("results", [])
        ]
    except Exception as e:
        docs = []
        print(f"[web_search] fallback failed: {e}")

    return {"graded_documents": docs, "used_web_search": True}


# --------------------------------------------------------------------------
# Give-up node: no relevant docs, no web search available
# --------------------------------------------------------------------------
def give_up(state: GraphState) -> dict:
    return {
        "generation": (
            "I don't have enough information in the indexed documentation to "
            "answer that confidently. Try rephrasing the question, or ingest "
            "documentation that covers this topic."
        ),
        "sources": [],
        "grounded": None,
    }


# --------------------------------------------------------------------------
# Node 4: Generation
# --------------------------------------------------------------------------
_GENERATION_PROMPT = """You are a technical documentation assistant. Answer the \
user's question clearly and accurately using ONLY the provided context. If the \
context is insufficient to fully answer, say what is missing rather than \
guessing.

After the answer, add a "Sources:" line listing which source labels (e.g. \
[1], [2]) you actually relied on.

Question: {question}

Context:
{context}"""


def _format_context(docs: List[dict]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        lines.append(f"[{i}] (source: {d['source']})\n{d['content']}")
    return "\n\n".join(lines)


def generate(state: GraphState) -> dict:
    docs = state.get("graded_documents", [])
    context = _format_context(docs) if docs else "No context available."
    llm = get_llm()
    prompt = _GENERATION_PROMPT.format(question=state["original_question"], context=context)
    response = llm.invoke([HumanMessage(content=prompt)])

    sources = [{"source": d["source"], "chunk_index": d.get("chunk_index")} for d in docs]
    return {"generation": response.content, "sources": sources}


# --------------------------------------------------------------------------
# Bonus node: Hallucination / groundedness check (Self-RAG inspired)
# --------------------------------------------------------------------------
_HALLUCINATION_PROMPT = """You are fact-checking an AI-generated answer against \
its source context. Answer "yes" if the answer is fully supported by the \
context (no fabricated facts), or "no" if it contains claims not backed by the \
context. Respond with a single word: yes or no.

Context:
{context}

Answer to check:
{generation}"""


def check_hallucination(state: GraphState) -> dict:
    docs = state.get("graded_documents", [])
    if not docs:
        # Nothing to ground against (e.g. the give_up path) -- skip the check.
        return {"grounded": None}

    llm = get_llm()
    context = _format_context(docs)
    prompt = _HALLUCINATION_PROMPT.format(context=context, generation=state.get("generation", ""))
    response = llm.invoke([HumanMessage(content=prompt)])
    grounded = response.content.strip().lower().startswith("yes")
    return {"grounded": grounded}


def decide_after_hallucination_check(state: GraphState) -> str:
    """Conditional edge after the hallucination check."""
    if state.get("grounded") is not False:
        return "end"
    if not state.get("regenerated"):
        return "regenerate"
    return "end"


def mark_regenerated(state: GraphState) -> dict:
    return {"regenerated": True}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()