FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the embedding model at build time so it's cached in the image
# and the first request doesn't pay the download cost. Using fastembed
# (ONNX Runtime) instead of sentence-transformers/PyTorch keeps this small
# enough to run comfortably in a 512MB free-tier container.
RUN python -c "from langchain_community.embeddings import FastEmbedEmbeddings; FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]