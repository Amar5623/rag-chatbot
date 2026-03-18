# routers/chat.py

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas import ChatRequest, ClearRequest
from services import rag_service

router = APIRouter(tags=["chat"])


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    vector_store = rag_service.get_vector_store()
    has_kb       = vector_store.count() > 0
    chain        = rag_service.get_or_create_session(req.session_id)

    async def event_generator():
        try:
            for chunk in chain.stream(req.question, has_kb=has_kb):
                if isinstance(chunk, str):
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                else:
                    # Only send citations and images for full document answers.
                    # For chitchat and general-knowledge fallback responses,
                    # retrieval is empty so citations and images are always [].
                    # We gate on query_type to be explicit and future-proof.
                    is_document = chunk.query_type == "document"

                    citations  = []
                    image_urls = []

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


@router.post("/session/clear")
async def clear_session(req: ClearRequest):
    rag_service.clear_session(req.session_id)
    return {"status": "ok"}