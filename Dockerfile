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
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', threads=1)"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]