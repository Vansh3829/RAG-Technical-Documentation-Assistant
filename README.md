# RAG Technical Documentation Assistant

This is my submission for the AI/ML Engineer Intern take home  a self-corrective
RAG system for answering questions about technical docs, built with LangGraph
and served over FastAPI.

I built the whole thing to run free, locally, on my own machine. No paid API
keys, no hosted vector DB, nothing that needs a credit card to try out.

## Stack I picked and why

- **LLM: Groq (llama-3.1-8b-instant).** Free tier, fast, no card needed to sign up.
- **Embeddings: fastembed (ONNX Runtime), running locally.** No API calls for
  embeddings at all, which made ingestion quick to iterate on and meant I
  never had to think about rate limits while testing. I actually started out
  with sentence-transformers/PyTorch here, and only switched to fastembed
  later once I hit a real problem trying to deploy this (more on that below).
- **Vector store: ChromaDB**, persisted to a local folder. It's embedded, so
  there's no separate database process to run or manage.
- **Tavily** for the optional web-search fallback, free tier, 1k searches/month.
- **SQLite** for feedback storage — it's one file, needs zero setup, and is
  plenty for what this needs to do.

None of this is exotic. I mostly picked whatever had the least setup friction
so I could spend my actual time on the graph logic instead of fighting infra.

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

The four required nodes are in there, plus a few of my own:

- **analyze_query** takes the raw question, has the LLM rewrite it into
  something that'll match better against the vector index, and classifies it
  as conceptual / how-to / troubleshooting / api-reference. I added the
  classification mostly because the assignment mentioned it as optional, and
  it ended up being genuinely useful to see in the API response while
  debugging.
- **retrieve** is a plain Chroma similarity search, top-k configurable through
  an env var.
- **grade_documents** is the actual self-corrective piece — one LLM call per
  retrieved chunk, asking a plain yes/no "is this relevant." I went back and
  forth on grading everything in a single batched call instead (would be
  faster) versus one at a time (what I went with). More on why below.
- if grading comes back with nothing relevant, **rewrite_query** has the LLM
  try a different phrasing and loops back to retrieve, up to `MAX_RETRIES`
  (2 by default).
- **web_search** is an optional fallback if retries run out and a Tavily key
  is configured. If there's no key, it falls through to...
- **give_up**, which just returns an honest "I don't know" instead of making
  something up. This node matters more than it might sound like — it's the
  difference between the system being trustworthy and not.
- **generate** writes the final answer from whatever context survived
  grading, with a citations line at the end.
- **check_hallucination** is a bonus node, inspired by Self-RAG. A second LLM
  call checks whether the generated answer is actually backed by the context
  it was given. If it isn't, the graph regenerates once — never more than
  once, there's a `regenerated` flag in state that caps it.

### On the state schema

Honestly, this took more actual thought than the nodes themselves. A few
decisions worth explaining:

- `original_question` is kept separate from `question`. `question` gets
  mutated every time it's rewritten (during analysis, during a retry), but
  the final answer should still be about what the user actually asked, so I
  didn't want that to get lost along the way.
- I keep both `documents` and `graded_documents` rather than overwriting one
  with the other. Mostly for debugging — when something looks off, I want to
  see what got retrieved versus what actually survived grading, without
  having to re-run anything.
- `retry_count` / `max_retries` is what keeps the rewrite → retrieve → grade
  loop bounded instead of infinite. `regenerated` does the same job for the
  hallucination-check loop. Two separate loops, two separate counters — I
  didn't want to conflate them into one and lose track of which loop was
  which.

## The corpus

I used a mix of two things:

1. Five short markdown docs I wrote myself, covering FastAPI, Pydantic,
   LangGraph, ChromaDB, and the Requests library (`data/corpus/*.md`). I
   wrote these because I wanted content dense with concrete, checkable facts
   — specific function signatures, small code snippets — so grading and
   retrieval would have clear right/wrong answers to test against, rather
   than vague prose.
2. The actual GitHub READMEs for those same five projects, fetched via
   `urls.txt` and saved locally to `data/corpus/from_url/`, so the repo
   stays reproducible without hitting the network again. I added these after
   noticing the assignment specifically calls out official docs as an
   example corpus, and wanted at least part of what's indexed to be real
   source material, not just my own summaries of it.

Run `python ingest.py --urls urls.txt` to pull in both. Every later
`python ingest.py` (no flags) will still pick up the saved copies
automatically, since it walks `data/corpus/` recursively — so once you've
fetched once, the whole thing works offline from then on.

### Chunking

Two passes:

1. Split on markdown headers first (`#`, `##`, `###`), so a chunk never
   silently straddles two unrelated sections. This mattered more than I
   expected — early on I saw chunks that mixed the tail of one section with
   the start of the next, and the grading LLM would sometimes call the whole
   thing "irrelevant" even though part of it genuinely wasn't.
2. Then a recursive character splitter within each section (800 characters,
   120 overlap), so nothing gets too large for the embedding model, and the
   overlap means information sitting near a chunk boundary doesn't just
   vanish.

800/120 wasn't a scientific choice, more like "roughly a few paragraphs." It
held up fine on both the tight, code-heavy docs I wrote and the looser README
prose, so I didn't feel the need to tune it further given the time I had.

## Running it locally

You'll need:
- Python 3.11 or 3.12 (3.14 currently breaks `pydantic-core`'s wheel build via
  maturin — I found this out the hard way, so save yourself the trouble and
  use 3.11/3.12)
- A free Groq key from console.groq.com/keys
- Optionally, a free Tavily key from tavily.com if you want the web-search
  fallback to actually do something

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env, paste in GROQ_API_KEY (and TAVILY_API_KEY if you have one)

python ingest.py --urls urls.txt   # builds the index, fetches + saves the real docs

uvicorn main:app --reload          # localhost:8000/docs for Swagger UI
```

Optional UI:
```bash
streamlit run streamlit_app.py     # localhost:8501
```

Or with Docker, if you'd rather skip the venv dance:
```bash
cp .env.example .env   # fill in GROQ_API_KEY first
docker compose up --build
```

Tests (these don't need an API key — they only exercise routing logic and
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

**GET /documents** — lists indexed sources and their chunk counts.

**POST /feedback**
```json
{ "question": "...", "answer": "...", "rating": "up", "comment": "optional" }
```

**GET /health** — basic liveness check.

## Conversation memory (bonus)

`/query` takes an optional `session_id`. Leave it out (or send `null`) for a
one-off, stateless question — that's the default, and how everything above
was tested. If you want follow-ups to work ("does it support X too?"), pass
back the `session_id` the server handed you from the previous response:

```json
{ "question": "How do I install LangGraph?" }
```
```json
{ "answer": "...", "session_id": "3f9e2c1a-...", ... }
```
```json
{ "question": "Does it support conditional edges too?", "session_id": "3f9e2c1a-..." }
```

On that second call, `analyze_query` pulls in the last `MAX_HISTORY_TURNS`
(3 by default) question/answer pairs from the session and uses them to
resolve "it" → LangGraph before rewriting the query.

I kept this deliberately simple — an in-memory dict on the FastAPI process,
keyed by `session_id`, capped at `MAX_SESSIONS_IN_MEMORY` so it can't grow
without bound. That means history resets on a server restart and wouldn't
behave correctly across multiple processes behind a load balancer. Both are
fine tradeoffs for a local, single-instance take-home project, just not
something I'd ship as-is for production. The Streamlit app handles the
session_id bookkeeping for you automatically, and has a "Start new
conversation" button to reset it.

## On deployment

I did try to get this live on a free host, and I want to be upfront about how
that went rather than pretend it's a solved problem.

I got it running on Render's free tier, but repeatedly hit its 512MB memory
ceiling. The first cause was genuinely my fault — sentence-transformers pulls
in PyTorch, and PyTorch alone is heavy enough to blow past 512MB in a small
container, so I swapped to fastembed (ONNX Runtime, no PyTorch) for a much
lighter footprint. After fixing that, I also found and fixed a second real
bug: fastembed's default cache directory lives under `/tmp`, and Render (like
most container platforms) mounts a fresh, empty `/tmp` at runtime, separate
from whatever got written there during the Docker build — so my "pre-download
the model at build time" step was silently getting thrown away every single
deploy. I fixed that too, pointing the cache at a path inside the app
directory instead.

Even after both of those fixes, the container was still tight enough against
512MB that it kept failing intermittently. At that point I looked into
switching to Hugging Face Spaces, but they recently moved Docker-based Spaces
behind a paid plan, so that option closed too. The remaining genuinely free,
no-card options I found all cap out at the same 512MB, which suggests the
ceiling here isn't really about which host — it's that this stack (LangGraph
+ LangChain + ChromaDB + ONNX Runtime, all required by the assignment) has a
baseline footprint that's simply tight against a 512MB limit, regardless of
which free platform it's running on.

Rather than keep chasing a free host with more headroom, I decided to stop
here: the assignment's actual requirement is "a working FastAPI application
that can be run locally," which this is, with clear setup steps above that
get it running in a few minutes. If it's useful, I'm also happy to redeploy
to a host with more free memory headroom (e.g. Google Cloud Run's free tier)
on request — I just didn't want to add a credit card to any service by
default without checking first.

## Things I'd do differently with more time

- Grade all retrieved chunks in a single structured-output call instead of
  one LLM call per chunk. I went with the slower, per-chunk approach because
  it's a lot easier to get a small free-tier model to reliably answer yes or
  no for one thing at a time than to get a clean, parseable JSON array back
  for four things at once — but it does mean more API calls per query than
  strictly necessary.
- The `RELEVANCE_SCORE_THRESHOLD` setting in `config.py` is there but unused.
  I'd like to combine the vector similarity score with the LLM's relevance
  verdict instead of relying on the LLM call alone.
- PDF ingestion — right now it's markdown/text/HTML only.
- A basic response cache, mostly so repeated demo or test questions don't
  burn through Groq calls unnecessarily.
- The conversation memory is intentionally minimal — in-memory only, and it
  only feeds into query rewriting, not the final generation prompt. With more
  time I'd feed relevant history into `generate` too, and back the session
  store with something that survives a restart (Redis would be the obvious
  free-tier-friendly choice).

## Assumptions I made

- "Free tier" was interpreted strictly: nothing in the default config
  requires a paid plan anywhere in the pipeline.
- Corpus size: I went with 10 documents total (5 written + 5 real READMEs)
  rather than exactly the suggested 3–5, since combining my own writeups with
  real source material felt like a better demonstration of the ingestion
  pipeline than either alone would have been.
- Where the grading LLM and the generation LLM disagree in edge cases — say,
  a chunk graded relevant but the final hallucination check flags the answer
  anyway — I let the hallucination check win and trigger one regeneration,
  rather than trying to reconcile both signals into a single score.

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