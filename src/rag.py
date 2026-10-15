"""Small local RAG index and search tool for Infosys annual reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = ROOT / "data" / "processed"
VECTOR_DIR = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "infosys_annual_reports"
OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
PAGE_HEADING = re.compile(r"(?m)^--- Page (\d+) ---\s*$")


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Create embeddings using the local Ollama HTTP API (no extra SDK needed)."""

    def __init__(self, url: str = OLLAMA_URL, model: str = EMBEDDING_MODEL) -> None:
        self.url = url.rstrip("/")
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        request = Request(
            f"{self.url}/api/embed",
            data=json.dumps({"model": self.model, "input": input}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=180) as response:
                payload = json.loads(response.read())
        except Exception as error:
            raise RuntimeError(
                f"Could not get embeddings from Ollama at {self.url}: {error}"
            ) from error
        return payload["embeddings"]

    @staticmethod
    def name() -> str:
        return "infosys_local_ollama_http"

    def get_config(self) -> dict[str, str]:
        return {"url": self.url, "model": self.model}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> OllamaEmbeddingFunction:
        return OllamaEmbeddingFunction(
            url=str(config.get("url", OLLAMA_URL)),
            model=str(config.get("model", EMBEDDING_MODEL)),
        )


def split_page(text: str, max_chars: int = 2200, overlap: int = 220) -> list[str]:
    """Split one page into overlapping chunks without crossing page boundaries."""
    clean = text.strip()
    if not clean:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        if end < len(clean):
            boundary = max(clean.rfind("\n", start, end), clean.rfind(" ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def report_pages(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    headings = list(PAGE_HEADING.finditer(text))
    pages: list[tuple[int, str]] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        pages.append((int(heading.group(1)), text[heading.end() : end].strip()))
    return pages


def make_chunks() -> tuple[list[str], list[dict[str, Any]], list[str]]:
    text_files = sorted(TEXT_DIR.glob("FY*.txt"))
    if len(text_files) != 5:
        raise FileNotFoundError(
            f"Expected five FY text files in {TEXT_DIR}; found {len(text_files)}."
        )

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    for text_path in text_files:
        year_match = re.fullmatch(r"FY(\d{4})", text_path.stem)
        if not year_match:
            raise ValueError(f"Unexpected report filename: {text_path.name}")
        year = int(year_match.group(1))
        for page_number, page_text in report_pages(text_path):
            for chunk_number, chunk in enumerate(split_page(page_text)):
                documents.append(chunk)
                metadatas.append(
                    {
                        "financial_year": year,
                        "source_filename": f"{text_path.stem}.pdf",
                        "text_filename": text_path.name,
                        "page_number": page_number,
                        "chunk_number": chunk_number,
                    }
                )
                ids.append(f"{text_path.stem}-p{page_number:04d}-c{chunk_number:03d}")

    if not documents:
        raise ValueError("No non-empty report pages were found to index.")
    return documents, metadatas, ids


def index_reports() -> None:
    if VECTOR_DIR.exists():
        raise FileExistsError(
            f"Will not overwrite existing vector store: {VECTOR_DIR}. "
            "Use the existing index or choose a new path after reviewing it."
        )

    documents, metadatas, ids = make_chunks()
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    batch_size = 32
    for start in range(0, len(documents), batch_size):
        stop = start + batch_size
        collection.add(
            ids=ids[start:stop],
            documents=documents[start:stop],
            metadatas=metadatas[start:stop],
        )
        print(f"Indexed {min(stop, len(documents))}/{len(documents)} chunks", end="\r")
    print(f"\nIndexed {len(documents)} page-aware chunks from five reports.")
    print(f"Local ChromaDB store: {VECTOR_DIR.relative_to(ROOT)}")


def get_collection() -> Any:
    if not VECTOR_DIR.exists():
        raise FileNotFoundError("The local index is missing. Run: python src/rag.py index")
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddingFunction(),
    )


def search_chunks(
    question: str,
    top_k: int = 5,
    financial_year: int | None = None,
    contains_text: str | None = None,
) -> list[dict[str, Any]]:
    collection = get_collection()
    filters = {"financial_year": financial_year} if financial_year is not None else None
    document_filter = {"$contains": contains_text} if contains_text else None
    result = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        where=filters,
        where_document=document_filter,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict[str, Any]] = []
    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": float(distance),
            }
        )
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index", help="Create a new local ChromaDB index")
    search_parser = subparsers.add_parser("search", help="Search indexed report passages")
    search_parser.add_argument("question", help="Question about the annual reports")
    search_parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        if args.command == "index":
            index_reports()
            return 0
        for hit in search_chunks(args.question, args.top_k):
            meta = hit["metadata"]
            print(
                f"[{meta['source_filename']}, p. {meta['page_number']}] "
                f"(distance {hit['distance']:.3f})"
            )
            print(hit["text"][:900].strip())
            print()
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
