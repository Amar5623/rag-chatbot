# routers/chat.py
#
# CHANGES:
#   - JWT protection on all endpoints
#   - session_id derived from JWT user_id
#   - POST /session/pin   — pin a source file for focused retrieval
#   - DELETE /session/pin — unpin, return to full KB search
#   - GET /session/pin    — return current pin status

import json
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from schemas import ChatRequest, ClearRequest
from services import rag_service
from auth.dependencies import get_current_user

router = APIRouter(tags=["chat"])


# ── Pin request model ─────────────────────────────────────────

class PinRequest(BaseModel):
    filename: str


# ── Chat stream ───────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(
    req         : ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    session_id   = current_user["user_id"]
    vector_store = rag_service.get_vector_store()
    has_kb       = vector_store.count() > 0
    chain        = rag_service.get_or_create_session(session_id)

    async def event_generator():
        try:
            for chunk in chain.stream(req.question, has_kb=has_kb):
                if isinstance(chunk, str):
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                else:
                    is_document = chunk.query_type == "document"
                    citations   = []
                    image_urls  = []

                    if is_document:
                        citations = [
                            {
                                "source"      : c.get("source", ""),
                                "page"        : c.get("page"),
                                "heading"     : c.get("heading", ""),
                                "section_path": c.get("section_path", ""),
                                "chunk_type"  : c.get("type", "text"),
                            }
                            for c in chunk.get_citations()
                        ]
                        image_urls = [
                            f"/images/{Path(p).name}"
                            for p in chunk.get_images()
                        ]

                    yield f"data: {json.dumps({'done': True, 'citations': citations, 'image_urls': image_urls, 'query_type': chunk.query_type, 'usage': chunk.usage})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Session management ────────────────────────────────────────

@router.post("/session/clear")
async def clear_session(
    req         : ClearRequest,
    current_user: dict = Depends(get_current_user),
):
    rag_service.clear_session(current_user["user_id"])
    return {"status": "ok"}


# ── Pin / unpin ───────────────────────────────────────────────

@router.post("/session/pin")
async def pin_source(
    req         : PinRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Pin the session to a single source file.
    All subsequent chat queries will only retrieve from this file.
    """
    session_id = current_user["user_id"]
    chain      = rag_service.get_or_create_session(session_id)

    # Verify the file actually exists in the KB
    sources = rag_service.get_vector_store().list_sources()
    if req.filename not in sources:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{req.filename}' not found in the knowledge base.",
        )

    chain.set_source_filter(req.filename)
    return {"status": "ok", "pinned": req.filename}


@router.delete("/session/pin")
async def unpin_source(
    current_user: dict = Depends(get_current_user),
):
    """Remove the source pin — return to full KB search."""
    session_id = current_user["user_id"]
    chain      = rag_service.get_or_create_session(session_id)
    chain.clear_source_filter()
    return {"status": "ok", "pinned": None}


@router.get("/session/pin")
async def get_pin(
    current_user: dict = Depends(get_current_user),
):
    """Return the currently pinned filename for this session, or null."""
    session_id = current_user["user_id"]
    chain      = rag_service.get_or_create_session(session_id)
    return {"pinned": chain.get_source_filter()}