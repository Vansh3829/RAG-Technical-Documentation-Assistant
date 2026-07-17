# ChromaDB: Core Concepts

## What is ChromaDB

Chroma is an open-source embedding database (a "vector store") designed to
make it easy to store text chunks alongside their vector embeddings and later
retrieve the chunks whose embeddings are most similar to a query embedding.
It can run fully embedded inside a Python process with on-disk persistence,
which makes it convenient for local development and small-to-medium
production workloads without needing a separate database server.

## Persistent Client

Chroma can be used with an in-memory client (data is lost when the process
exits) or a persistent client that writes data to a directory on disk.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="docs")
```

## Collections

A "collection" in Chroma is roughly analogous to a table in a relational
database: it is a named group of records, where each record consists of an
id, the original document text, an embedding vector, and optional metadata
(a dictionary of key-value pairs such as source filename or page number).

```python
collection.add(
    ids=["doc1_chunk0", "doc1_chunk1"],
    documents=["first chunk text", "second chunk text"],
    metadatas=[{"source": "doc1.md"}, {"source": "doc1.md"}],
)
```

If you do not supply embeddings directly, Chroma can compute them for you
using a configurable embedding function, or you can pass in embeddings that
were computed by an external model (for example, a sentence-transformers
model or a hosted embeddings API) and store them alongside the documents.

## Querying

To retrieve similar chunks, call `collection.query()` with either the raw
query text (if an embedding function is configured) or a precomputed query
embedding, plus `n_results` for how many matches to return.

```python
results = collection.query(query_texts=["How do I install FastAPI?"], n_results=4)
```

The results include the matched documents, their metadata, and a distance
score indicating how close each match is to the query in embedding space
(lower distance generally means higher similarity, depending on the distance
metric configured for the collection).

## Filtering with Metadata

Chroma supports filtering results by metadata using a `where` clause, which is
useful when you want to restrict a search to a specific document, source, or
category rather than searching across the entire collection.

```python
collection.query(
    query_texts=["installation steps"],
    n_results=4,
    where={"source": "fastapi_basics.md"},
)
```

## Choosing Chunk Size

Because similarity search compares whole-chunk embeddings, the size of each
chunk affects retrieval quality. Chunks that are too large tend to dilute the
embedding with unrelated content, lowering precision, while chunks that are
too small may lack enough context to be useful once retrieved. A common
starting point for technical prose is a few hundred tokens per chunk with a
small overlap between consecutive chunks, so that information near chunk
boundaries is not split awkwardly across two chunks that never get retrieved
together.
