FROM python:3.11-slim

WORKDIR /app

# System deps for chromadb (hnswlib build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedding model at build time so it's cached in the image
# and the container doesn't have to fetch it over the network on every cold
# start. Calls fastembed's TextEmbedding directly (matching llm/client.py) --
# NOT langchain_community.embeddings.FastEmbedEmbeddings, which has a bug
# that silently no-ops instead of actually downloading/initializing the
# model, which is why this step was previously not actually caching anything.
#
# cache_dir is explicit and matches EMBEDDING_CACHE_DIR in config.py exactly
# (./.fastembed_cache, resolved from WORKDIR /app). This matters: fastembed's
# *default* cache dir lives under /tmp, and most hosting platforms (Render
# included) mount /tmp fresh at container runtime -- separate from whatever
# was written there during this build step -- so without an explicit,
# non-/tmp path here, this "pre-download" would silently do nothing useful.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', threads=1, cache_dir='./.fastembed_cache')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]