import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag_system import RAGSystem


app = FastAPI(
    title="Company RAG Assistant API",
    version="1.0.0",
    description="HTTP API for querying company policy documents.",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask the RAG system.")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of relevant chunks to retrieve.")


def get_rag() -> RAGSystem:
    docs_folder = os.getenv("DOCS_FOLDER", "data/documents")
    return RAGSystem(docs_folder=docs_folder)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "Company RAG Assistant API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        rag = get_rag()
        return {
            "status": "ok",
            "documents_loaded": len(rag.chunks),
            "top_k": rag.top_k,
        }
    except Exception as exc:  # pragma: no cover - defensive health check
        return {"status": "error", "detail": str(exc)}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    rag = get_rag()
    answer, sources = rag.answer(request.question, top_k=request.top_k, return_sources=True)
    return {
        "answer": answer,
        "sources": [
            {
                "source": source.get("source", "unknown"),
                "page": source.get("page"),
                "text": source.get("text", ""),
                "score": source.get("score"),
            }
            for source in sources
        ],
    }
