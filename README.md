# RAG Technical Documentation Assistant

Take-home assignment submission — a self-corrective RAG system for answering
questions about technical docs, built with LangGraph and served over FastAPI.

I built this to run entirely free, locally, on my own machine — no paid API
keys, no hosted vector DB, nothing that needs a credit card.

## Stack I picked and why

- **LLM: Groq (llama-3.1-8b-instant).** Free tier, fast, no card needed to sign up.
- **Embeddings: fastembed (ONNX Runtime), running locally.** No
  API calls for embeddings at all — this made ingestion fast to iterate on and
  meant I never had to worry about rate limits while testing. I originally
  used sentence-transformers/PyTorch here, but switched to fastembed after
  Render's free tier OOM'd on the PyTorch memory footprint (512MB limit) —
  more on that below.
- **Vector store: ChromaDB**, persisted to a local folder. It's embedded, so
  there's no separate DB server to run.
- **Tavily** for the optional web-search fallback (free tier, 1k searches/month).
- **SQLite** for feedback storage, because it's one file and needs zero setup.

None of this is exotic — I mostly picked whatever had the lowest setup friction
so I could spend my time on the actual graph logic instead of fighting infra.

## How it's structured

```
START -> analyze_query -> retrieve -> grade_documents
                                          |
                    (conditional: decide_to_generate)
                    /            |              \            \
              generate     rewrite_query      web_search     give_up
                 |               |                |             |
       check_hallucination   -> retrieve (loop)    -> generate      END
            /        \
  (conditional: decide_after_hallucination_check)
   end             regenerate -> mark_regenerated -> generate
```

Four required nodes, plus a few I added:

- **analyze_query** — takes the raw question, asks the LLM to rewrite it into
  something that'll match better against the vector index, and classifies it
  as conceptual / how-to / troubleshooting / api-reference. I mostly added the
  classification because the assignment mentioned it, and it turned out to be
  a nice thing to surface in the API response for debugging.
- **retrieve** — plain Chroma similarity search, top-k configurable via env var.
- **grade_documents** — this is the actual self-corrective piece. One LLM call
  per retrieved chunk asking "is this relevant, yes or no." I went back and
  forth on whether to grade all chunks in one batched call instead (would be
  faster) vs. one-at-a-time (what I ended up doing) — see the tradeoffs
  section below for why I chose the slower option.
- **rewrite_query / retrieve loop** — if grading comes back empty, an LLM
  rewrites the query and we go around again, up to `MAX_RETRIES` (default 2).
- **web_search** — optional fallback if retries run out and a Tavily key is
  configured. If not configured, falls through to...
- **give_up** — returns an honest "I don't know" instead of making something
  up. I actually think this node matters more than it sounds like it should —
  it's the difference between the system being trustworthy and not.
- **generate** — writes the final answer from whatever context survived
  grading (local docs or web search results), with a citations line.
- **check_hallucination** — bonus node, inspired by Self-RAG. A second LLM
  call checks whether the generated answer is actually backed by the context
  it was given. If not, it regenerates once (never loops forever — there's a
  `regenerated` flag in state to cap it at one retry).

### On the state schema

This was the part of the assignment that took the most actual thinking, more
than the nodes themselves. A few decisions I made and why:

- I kept `original_question` separate from `question`. `question` gets
  mutated every time the query gets rewritten (during analysis, during a
  retry), but the final answer should still be about what the user *actually*
  asked, so I didn't want to lose that.
- `documents` vs `graded_documents` — I keep both instead of just overwriting
  in place. Purely for debugging/tracing purposes: when something goes wrong,
  I want to be able to see what got retrieved vs. what actually survived
  grading, without re-running anything.
- `retry_count` / `max_retries` — this is what makes the rewrite → retrieve →
  grade loop bounded instead of infinite. Same idea with `regenerated`, a
  one-shot flag, for the hallucination-check loop. Two separate loops in this
  graph, two separate counters/flags — didn't want to conflate them.

## The corpus

I used a mix of two things:

1. Five short markdown docs I wrote myself, covering FastAPI, Pydantic,
   LangGraph, ChromaDB, and the Requests library (`data/corpus/*.md`). I wrote
   these because I wanted content that was dense with concrete, testable
   facts (specific function signatures, code snippets) rather than
   documentation-style prose, so grading and retrieval would have clear
   right/wrong answers to test against.
2. The actual GitHub READMEs for those same five projects, fetched via
   `urls.txt` and saved locally to `data/corpus/from_url/` so the repo stays
   reproducible without hitting the network again. I added these after
   realizing the assignment specifically calls out "official docs" as an
   example corpus, and I wanted at least some of what I'm indexing to be real
   source material, not just my own summaries.

Run `python ingest.py --urls urls.txt` to pull both. Every later `python
ingest.py` (no flags) will still pick up the saved copies automatically,
since it walks `data/corpus/` recursively — so once you've fetched once, it
works offline.

### Chunking

Two-pass approach:

1. Split on markdown headers first (`#`, `##`, `###`), so a chunk never
   silently straddles two unrelated sections. This mattered more than I
   expected once I saw a few chunks that mixed the end of one
   section with the start of the next — the grading LLM would sometimes call
   those "irrelevant" even when part of the chunk was actually useful.
2. Then a recursive character splitter within each section (800 chars, 120
   overlap), so nothing gets too big for the embedding model, and the overlap
   means information near a chunk boundary doesn't just disappear.

800/120 wasn't a very scientific choice — it's roughly "a few paragraphs," and
it worked well enough on both the tight, code-heavy summaries and the looser
README prose that I didn't feel a need to tune it further given the time
I had.

## Running it locally

You'll need:
- Python 3.11 or 3.12 (3.14 currently breaks `pydantic-core`'s wheel build via
  maturin — found this out the hard way, use 3.11/3.12)
- A free Groq key from console.groq.com/keys
- Optionally, a free Tavily key from tavily.com if you want the web-search
  fallback to actually do anything

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env, paste in GROQ_API_KEY (and TAVILY_API_KEY if you have one)

python ingest.py --urls urls.txt   # builds the index, fetches + saves real docs

uvicorn main:app --reload          # localhost:8000/docs for Swagger UI
```

Optional UI:
```bash
streamlit run streamlit_app.py     # localhost:8501
```

Or with Docker, if you'd rather not deal with venvs:
```bash
cp .env.example .env   # fill in GROQ_API_KEY first
docker compose up --build
```

Tests (these don't need an API key — they only cover routing logic and
chunking, not actual LLM calls):
```bash
pytest tests/
```

## API

**POST /query**
```json
{ "question": "How do I install LangGraph?" }
```
```json
{
  "answer": "To install LangGraph, you can use pip:\n\npip install -U langgraph\n\nSources: [1], [2]",
  "sources": [{"source": "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md", "chunk_index": 0}],
  "query_type": "how-to",
  "retries_used": 0,
  "used_web_search": false,
  "grounded": true
}
```

**POST /ingest** — multipart file upload (.md/.txt)
```bash
curl -X POST http://localhost:8000/ingest -F "files=@my_doc.md"
```

**POST /ingest/urls**
```bash
curl -X POST http://localhost:8000/ingest/urls \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/docs-page"]}'
```

**GET /documents** — lists indexed sources and chunk counts.

**POST /feedback**
```json
{ "question": "...", "answer": "...", "rating": "up", "comment": "optional" }
```

**GET /health** — basic liveness check.

## Conversation memory (bonus)

`/query` accepts an optional `session_id`. Leave it out (or send `null`) for a
one-off, stateless question — that's the default and how everything above
was tested. If you want follow-ups to work ("what about *that* library?"),
send back the `session_id` the server returned from the previous call:

```json
{ "question": "How do I install LangGraph?" }
```
```json
{ "answer": "...", "session_id": "3f9e2c1a-...", ... }
```
```json
{ "question": "Does it support conditional edges too?", "session_id": "3f9e2c1a-..." }
```

On the second call, `analyze_query` gets the last `MAX_HISTORY_TURNS` (3, by
default) question/answer pairs from that session and uses them to resolve
"it" → LangGraph before rewriting the query for retrieval.

I kept the implementation intentionally simple: it's an in-memory dict on the
FastAPI process, keyed by `session_id`, capped at `MAX_SESSIONS_IN_MEMORY`
sessions so it can't grow unbounded. That means history resets on server
restart and won't work correctly if you ever ran multiple API processes
behind a load balancer — both fine tradeoffs for a local/single-instance
take-home project, not something I'd ship as-is for production. Streamlit
handles this for you automatically (it remembers the session_id in
`st.session_state` and has a "Start new conversation" button to reset it).

## Things I'd do differently with more time

- Grade all retrieved chunks in a single structured-output call instead of
  one LLM call per chunk. I chose the slower, per-chunk approach because it's
  much easier to get a small free-tier model to reliably answer "yes" or "no"
  for one thing at a time than to get a clean parseable JSON array back for
  four things at once — but it does mean more API calls per query than
  necessary.
- The `RELEVANCE_SCORE_THRESHOLD` setting in `config.py` is there but unused —
  I'd like to combine the vector similarity score with the LLM's relevance
  verdict rather than relying on the LLM call alone.
- PDF ingestion. Right now it's markdown/text/HTML only.
- A basic response cache, mostly so repeated demo/test questions don't burn
  through Groq calls unnecessarily.
- The conversation memory is deliberately minimal right now (see below) —
  in-memory only, and it only feeds into query rewriting, not the final
  generation prompt. With more time I'd feed relevant history into
  `generate` too, and back the session store with something that survives a
  restart (Redis would be the obvious free-tier-friendly option).
- Worth calling out since it actually happened during deployment: my first
  attempt at deploying to Render's free tier (512MB RAM) OOM'd, because
  sentence-transformers pulls in PyTorch, and PyTorch alone is heavy enough
  to blow past that limit in a small container. Swapped to `fastembed`
  (ONNX Runtime, no PyTorch) and it fit comfortably. Same idea — local, free,
  no API key — just a lighter runtime. If you're running this only locally
  with plenty of RAM, either would've worked fine; this only bit me because
  of the free-tier hosting constraint.

## Assumptions I made

- "Free tier" was interpreted strictly — nothing in the default config
  requires a paid plan anywhere in the pipeline.
- Corpus size: went with 10 documents total (5 written + 5 real READMEs)
  rather than exactly the suggested 3–5, since combining original writeups
  with real source material felt like a better demonstration of the ingestion
  pipeline than either alone.
- Where the grading LLM and the generation LLM disagree in edge cases (e.g. a
  chunk graded relevant but the final hallucination check flags the answer
  anyway), I let the hallucination check win and trigger one regeneration,
  rather than trying to reconcile the two signals into a single score.

## Project layout

```
rag-doc-assistant/
├── main.py                  # FastAPI app + endpoints
├── config.py                # settings (env-driven)
├── ingest.py                 # loading, chunking, embedding, storing
├── streamlit_app.py          # UI
├── graph/
│   ├── state.py               # GraphState schema
│   ├── nodes.py                # node + routing functions
│   └── workflow.py              # StateGraph assembly
├── llm/client.py                # Groq + local embeddings
├── vectorstore/store.py          # Chroma wrapper
├── data/corpus/                   # written docs + from_url/ (real fetched docs)
├── urls.txt                        # real doc sources to fetch
├── tests/test_workflow.py           # routing + chunking tests
├── requirements.txt
├── Dockerfile / docker-compose.yml
└── .env.example
```