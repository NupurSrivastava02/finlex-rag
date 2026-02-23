"""
FinLex RAG — Financial & Legal Document Intelligence System
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from backend.routers.api import doc_router, query_router

app = FastAPI(
    title="FinLex RAG",
    description="Financial & Legal Document Intelligence powered by RAG",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(doc_router, prefix="/api")
app.include_router(query_router, prefix="/api")

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend", "static")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the frontend UI."""
    index_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    return FileResponse(index_path)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "FinLex RAG"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
