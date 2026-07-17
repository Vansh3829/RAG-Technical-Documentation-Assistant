"""
Document ingestion pipeline.

Loads Markdown/text documents (from a local directory and/or a list of URLs),
splits them into overlapping chunks using a markdown-aware splitter, and
stores the chunks + embeddings in the persistent Chroma vector store.

Run standalone:
    python ingest.py                      # ingest ./data/corpus
    python ingest.py --urls urls.txt       # also fetch and ingest URLs
    python ingest.py --reset               # wipe the collection first

Or call `ingest_paths()` / `ingest_urls()` programmatically (used by the
`/ingest` API endpoint and by the FastAPI startup event).
"""
import argparse
import os
import re
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import settings
from vectorstore.store import add_documents, count_documents, reset_collection

MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _chunk_text(text: str, source: str) -> List[Document]:
    """
    Chunking strategy (see README for the reasoning):
    1. First split on Markdown headers, so a chunk never silently crosses a
       section boundary (keeps a chunk topically coherent).
    2. Then run a recursive character splitter within each section so no
       single chunk becomes too large for the embedding model / LLM context.
    3. Keep a small overlap so information near a chunk boundary is not lost.
    """
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS)
    try:
        header_chunks = header_splitter.split_text(text)
    except Exception:
        header_chunks = [Document(page_content=text, metadata={})]

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks: List[Document] = []
    for i, hc in enumerate(char_splitter.split_documents(header_chunks)):
        hc.metadata["source"] = source
        hc.metadata["chunk_index"] = len(final_chunks)
        final_chunks.append(hc)
    return final_chunks


def ingest_paths(paths: List[str]) -> dict:
    """Ingests a list of local file paths (.md or .txt)."""
    all_chunks: List[Document] = []
    ingested_files = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        chunks = _chunk_text(text, source)
        all_chunks.extend(chunks)
        ingested_files.append({"source": source, "chunks": len(chunks)})

    added = add_documents(all_chunks)
    return {"files": ingested_files, "chunks_added": added, "total_chunks_in_store": count_documents()}


def ingest_directory(directory: str) -> dict:
    """Ingests every .md / .txt file found in `directory`, recursively (so
    e.g. data/corpus/from_url/*.md -- saved copies of fetched real docs --
    are picked up automatically on every run, including the FastAPI startup
    auto-ingest)."""
    paths = []
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if fname.lower().endswith((".md", ".txt")):
                paths.append(os.path.join(root, fname))
    return ingest_paths(sorted(paths))


def _slugify(url: str) -> str:
    slug = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", slug).strip("_")
    return slug[:120] or "url_doc"


def _save_local_copy(url: str, text: str, save_dir: str) -> str:
    """
    Persists the fetched/cleaned text to disk under `save_dir`, so the real
    documentation content that was ingested becomes part of the committed,
    offline-reproducible corpus (per the assignment: "provide the documents
    you used, or a script to fetch them" -- this gives both).
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{_slugify(url)}.md"
    path = os.path.join(save_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<!-- source: {url} -->\n\n{text}")
    return path


def ingest_urls(urls: List[str], save_dir: Optional[str] = None) -> dict:
    """
    Fetches each URL, extracts clean text (markdown pass-through for raw
    .md/.txt URLs, HTML-stripped text otherwise), ingests it, and -- unless
    `save_dir` is None -- writes a local copy so the real content is
    committed to the repo and re-ingestible without network access.
    """
    all_chunks: List[Document] = []
    ingested = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "rag-doc-assistant/1.0"})
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "").lower()
            is_plain_markdown = (
                url.lower().endswith((".md", ".txt"))
                or "text/plain" in content_type
                or "text/markdown" in content_type
            )

            if is_plain_markdown:
                # Raw markdown / plain text (e.g. a GitHub raw README URL) --
                # use as-is, do NOT run it through an HTML parser, since that
                # would mangle code blocks containing "<" / ">".
                text = resp.text
            else:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n")
                text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

            saved_path = None
            if save_dir:
                saved_path = _save_local_copy(url, text, save_dir)

            chunks = _chunk_text(text, source=url)
            all_chunks.extend(chunks)
            entry = {"source": url, "chunks": len(chunks)}
            if saved_path:
                entry["saved_to"] = saved_path
            ingested.append(entry)
        except Exception as e:
            ingested.append({"source": url, "error": str(e)})

    added = add_documents(all_chunks)
    return {"urls": ingested, "chunks_added": added, "total_chunks_in_store": count_documents()}


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument("--dir", default=settings.CORPUS_DIR, help="Directory of .md/.txt files")
    parser.add_argument("--urls", default=None, help="Path to a text file with one URL per line")
    parser.add_argument("--reset", action="store_true", help="Wipe the collection before ingesting")
    parser.add_argument(
        "--save-dir",
        default=os.path.join(settings.CORPUS_DIR, "from_url"),
        help="Where to save local copies of fetched URL content (offline reproducibility). "
             "Pass --no-save to skip saving.",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save local copies of fetched URLs")
    args = parser.parse_args()

    if args.reset:
        try:
            reset_collection()
            print("Collection reset.")
        except Exception as e:
            print(f"Nothing to reset ({e}).")

    result = ingest_directory(args.dir)
    print(f"Ingested directory '{args.dir}': {result}")

    if args.urls and os.path.isfile(args.urls):
        with open(args.urls, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        if urls:
            save_dir = None if args.no_save else args.save_dir
            result = ingest_urls(urls, save_dir=save_dir)
            print(f"Ingested URLs: {result}")


if __name__ == "__main__":
    main()