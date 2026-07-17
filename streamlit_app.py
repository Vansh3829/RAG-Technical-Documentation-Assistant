"""
Minimal Streamlit UI for the RAG Technical Documentation Assistant.

Run with:
    streamlit run streamlit_app.py

Expects the FastAPI backend to be running (default http://localhost:8000).
Set API_BASE_URL as an env var to point at a deployed backend instead.
"""
import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Tech Docs Assistant", page_icon="📚", layout="centered")
st.title("📚 RAG Technical Documentation Assistant")
st.caption("Self-corrective LangGraph RAG over FastAPI / Pydantic / LangGraph / ChromaDB / Requests docs")

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: question, answer, sources, feedback_given

if "session_id" not in st.session_state:
    st.session_state.session_id = None  # server assigns one on the first /query response

with st.sidebar:
    st.subheader("Conversation")
    if st.session_state.session_id:
        st.caption(f"Session: {st.session_state.session_id[:8]}...")
    if st.button("Start new conversation"):
        st.session_state.session_id = None
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("Index status")
    try:
        docs = requests.get(f"{API_BASE_URL}/documents", timeout=10).json()
        st.metric("Indexed chunks", docs.get("total_chunks", 0))
        for s in docs.get("sources", []):
            st.text(f"• {s['source']} ({s['chunk_count']} chunks)")
    except Exception as e:
        st.error(f"Backend not reachable: {e}")

    st.divider()
    st.subheader("Ingest a URL")
    url_to_ingest = st.text_input("Documentation URL")
    if st.button("Ingest URL") and url_to_ingest:
        with st.spinner("Fetching and indexing..."):
            try:
                r = requests.post(f"{API_BASE_URL}/ingest/urls", json={"urls": [url_to_ingest]}, timeout=60)
                st.success(r.json())
            except Exception as e:
                st.error(str(e))

question = st.chat_input("Ask a question about FastAPI, Pydantic, LangGraph, ChromaDB, or Requests...")

if question:
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(f"{API_BASE_URL}/query", json={
                "question": question,
                "session_id": st.session_state.session_id,
            }, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.session_id = data.get("session_id", st.session_state.session_id)
            st.session_state.history.append({
                "question": question,
                "answer": data.get("answer"),
                "sources": data.get("sources", []),
                "query_type": data.get("query_type"),
                "retries_used": data.get("retries_used", 0),
                "used_web_search": data.get("used_web_search", False),
                "grounded": data.get("grounded"),
                "feedback_given": None,  # becomes "up" or "down" once submitted
            })
        except Exception as e:
            st.session_state.history.append({
                "question": question,
                "answer": None,
                "error": str(e),
                "feedback_given": None,
            })

# Render every past turn on every rerun (including reruns triggered by a
# feedback button click) so the button's on_click handler always has a
# chance to run -- this is what makes the feedback buttons actually work,
# instead of being silently skipped because st.chat_input() only returns a
# non-None value on the single run where the user hit Enter.
for i, entry in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        if entry.get("error"):
            st.error(f"Request failed: {entry['error']}")
            continue

        st.write(entry["answer"])

        meta_bits = []
        if entry.get("query_type"):
            meta_bits.append(f"type: {entry['query_type']}")
        if entry.get("retries_used"):
            meta_bits.append(f"retries: {entry['retries_used']}")
        if entry.get("used_web_search"):
            meta_bits.append("used web search fallback")
        if entry.get("grounded") is False:
            meta_bits.append("⚠️ groundedness check flagged this answer")
        if meta_bits:
            st.caption(" · ".join(meta_bits))

        if entry.get("sources"):
            with st.expander("Sources"):
                for s in entry["sources"]:
                    st.text(f"- {s.get('source')}")

        if entry.get("feedback_given"):
            st.caption(f"Feedback recorded: {entry['feedback_given']}")
        else:
            def _submit_feedback(index: int, rating: str):
                e = st.session_state.history[index]
                try:
                    requests.post(f"{API_BASE_URL}/feedback", json={
                        "question": e["question"], "answer": e["answer"], "rating": rating,
                    }, timeout=10)
                    st.session_state.history[index]["feedback_given"] = rating
                    st.toast("Thanks for the feedback!")
                except Exception as err:
                    st.toast(f"Feedback failed to send: {err}")

            col1, col2 = st.columns(2)
            col1.button("👍 Helpful", key=f"up-{i}", on_click=_submit_feedback, args=(i, "up"))
            col2.button("👎 Not helpful", key=f"down-{i}", on_click=_submit_feedback, args=(i, "down"))