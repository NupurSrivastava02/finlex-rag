"""
FastAPI routers for document management and RAG query endpoints.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend.services.ingestion import ingest_document, list_ingested_documents, delete_document
from backend.services.rag import query
from backend.services.evaluation import run_evaluation, DEFAULT_TEST_CASES

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ── Document Router ──────────────────────────────────────────────────────────
doc_router = APIRouter(prefix="/documents", tags=["documents"])


@doc_router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF or DOCX document."""
    allowed = {".pdf", ".docx", ".doc"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported. Use PDF or DOCX.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_document(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


@doc_router.get("/list")
async def list_documents():
    """List all ingested documents."""
    docs = list_ingested_documents()
    return {"documents": docs, "count": len(docs)}


@doc_router.delete("/{filename}")
async def remove_document(filename: str):
    """Delete a document and its chunks from the vector store."""
    result = delete_document(filename)
    # Also remove the uploaded file
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return result


# ── Query Router ─────────────────────────────────────────────────────────────
query_router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str


class EvalRequest(BaseModel):
    test_cases: Optional[List[dict]] = None


@query_router.post("")
async def ask_question(req: QueryRequest):
    """Ask a question against the ingested documents."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = query(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@query_router.post("/evaluate")
async def evaluate_pipeline(req: EvalRequest):
    """Run RAGAS evaluation on the pipeline."""
    test_cases = req.test_cases or DEFAULT_TEST_CASES
    try:
        scores = run_evaluation(test_cases)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return scores
