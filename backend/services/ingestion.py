"""
Document ingestion service.
Handles PDF/DOCX parsing, chunking, embedding, and storing in ChromaDB.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_embeddings():
    """Initialize OpenAI embeddings."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store(collection_name: str = "finlex") -> Chroma:
    """Get or create ChromaDB vector store."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def load_document(file_path: str) -> List[Document]:
    """Load a PDF or DOCX document and return LangChain Document objects."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()

    # Attach source metadata
    for doc in documents:
        doc.metadata["source_file"] = Path(file_path).name
        doc.metadata["file_hash"] = _hash_file(file_path)

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_document(file_path: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
    1. Load document
    2. Chunk
    3. Embed and store in ChromaDB
    Returns metadata about the ingestion.
    """
    # Load
    documents = load_document(file_path)
    total_pages = len(documents)

    # Chunk
    chunks = chunk_documents(documents)
    total_chunks = len(chunks)

    # Store in vector DB
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return {
        "file": Path(file_path).name,
        "pages": total_pages,
        "chunks": total_chunks,
        "status": "success",
    }


def list_ingested_documents() -> List[str]:
    """Return list of unique source files in the vector store."""
    vector_store = get_vector_store()
    collection = vector_store._collection
    results = collection.get(include=["metadatas"])
    sources = set()
    for meta in results["metadatas"]:
        if meta and "source_file" in meta:
            sources.add(meta["source_file"])
    return sorted(list(sources))


def delete_document(filename: str) -> Dict[str, Any]:
    """Delete all chunks for a given source file from the vector store."""
    vector_store = get_vector_store()
    collection = vector_store._collection
    results = collection.get(include=["metadatas"])
    ids_to_delete = [
        results["ids"][i]
        for i, meta in enumerate(results["metadatas"])
        if meta and meta.get("source_file") == filename
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return {"deleted_chunks": len(ids_to_delete), "file": filename}


def _hash_file(file_path: str) -> str:
    """MD5 hash of file for deduplication."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
