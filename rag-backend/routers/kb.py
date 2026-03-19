# routers/kb.py
#
# CHANGES:
#   - JWT protection on wipe endpoint (stats/health remain public for UI polling)
#   - parent_count removed from stats (ParentStore gone — inline now)
#   - parent_store.reset() call removed from wipe

from fastapi import APIRouter, Depends
from schemas import (
    DocumentsResponse, HealthResponse,
    StatsResponse, WipeResponse,
)
from services import rag_service
from routers.ingest import _wipe_hashes
from auth.dependencies import get_current_user
from config import settings

router = APIRouter(tags=["kb"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """Public — used by frontend to check if backend is up."""
    return HealthResponse(
        status          = "ok",
        groq_configured = bool(settings.groq_api_key),
    )


@router.get("/stats", response_model=StatsResponse)
async def stats():
    """
    Public — frontend polls this on load to show KB status.
    No auth required so the sidebar can display stats before login too.
    """
    vs   = rag_service.get_vector_store()
    bm25 = rag_service.get_bm25_store()
    return StatsResponse(
        total_vectors   = vs.count(),
        bm25_docs       = len(bm25),
        parent_count    = 0,           # ParentStore removed — inline now
        indexed_files   = vs.list_sources(),
        embedding_model = settings.embedding_model,
        llm_model       = settings.groq_model,
        collection      = settings.qdrant_collection,
    )


@router.get("/documents", response_model=DocumentsResponse)
async def documents():
    files = rag_service.get_vector_store().list_sources()
    return DocumentsResponse(files=files, total_files=len(files))


@router.delete("/collection", response_model=WipeResponse)
async def wipe(current_user: dict = Depends(get_current_user)):
    """Wipe the entire knowledge base. Requires auth."""
    rag_service.get_vector_store().reset_collection()
    rag_service.get_bm25_store().reset()
    _wipe_hashes()
    return WipeResponse(status="ok", message="Knowledge base wiped.")