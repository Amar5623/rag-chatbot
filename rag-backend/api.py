# api.py
# Routes only. Models → schemas.py  |  State + helpers → services.py
#
# Run:
#   uvicorn api:app --host 0.0.0.0 --port 8000 --reload

import os
import sys
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schemas import (
    ChatRequest, ClearRequest,
    Citation, ChatResponse, IngestResponse,
    StatsResponse, HealthResponse, DocumentsResponse, WipeResponse,
)
from services import (
    state, initialise,
    save_parent_store, rebuild_chain,
    get_or_create_session,
    ingest_file, SUPPORTED_EXTENSIONS,
    has_kb, is_groq_configured,
)
from config import QDRANT_COLLECTION, PARENT_STORE_PATH


# ─────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialise()
    yield
    print("\n=== Shutting down ===")


# ─────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────

app = FastAPI(
    title    = "RAG Chatbot API",
    version  = "2.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

_IMAGES_ABS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted_images")
os.makedirs(_IMAGES_ABS, exist_ok=True)
app.mount("/images", StaticFiles(directory=_IMAGES_ABS), name="images")


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status          = "ok",
        groq_configured = is_groq_configured(),
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    return StatsResponse(
        total_vectors   = state.store.count(),
        bm25_docs       = len(state.bm25),
        parent_count    = len(state.parent_store),
        indexed_files   = state.store.list_sources(),
        embedding_model = state.embedder.model_name,
        llm_model       = state.chain.llm.model_name,
        collection      = QDRANT_COLLECTION,
    )


@app.get("/documents", response_model=DocumentsResponse)
def documents():
    files = state.store.list_sources()
    return DocumentsResponse(files=files, total_files=len(files))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    all_children : list[dict] = []
    all_parents  : dict       = {}
    indexed      : list[str]  = []

    for upload in files:
        ext = os.path.splitext(upload.filename)[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {upload.filename}",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name

        try:
            children, parents = ingest_file(tmp_path, upload.filename)
            all_children.extend(children)
            all_parents.update(parents)
            indexed.append(upload.filename)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process {upload.filename}: {e}",
            )
        finally:
            os.unlink(tmp_path)

    state.store.add_documents(all_children)
    state.bm25.add(all_children)
    state.parent_store.update(all_parents)
    save_parent_store()
    rebuild_chain()

    return IngestResponse(
        status        = "success",
        files_indexed = indexed,
        total_chunks  = len(all_children),
        total_parents = len(all_parents),
        message       = (
            f"Indexed {len(all_children)} chunks across {len(indexed)} file(s). "
            f"Total KB: {state.store.count()} vectors."
        ),
    )


@app.post("/chat", response_model=ChatResponse)
def chat_blocking(req: ChatRequest):
    get_or_create_session(req.session_id)
    result = state.chain.ask(question=req.question, has_kb=has_kb())
    state.sessions[req.session_id] = state.chain.history

    citations = [
        Citation(
            source       = c.get("source", ""),
            page         = c.get("page"),
            heading      = c.get("heading", ""),
            section_path = c.get("section_path", ""),
            chunk_type   = c.get("type", "text"),
        )
        for c in result.get_citations()
    ]

    return ChatResponse(
        answer     = result.get_answer(),
        query_type = result.query_type,
        citations  = citations,
        usage      = result.usage,
        session_id = req.session_id,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            for item in state.chain.stream(
                question = req.question,
                has_kb   = has_kb(),
            ):
                if isinstance(item, str):
                    payload = json.dumps({"token": item})
                else:
                    citations = [
                        {
                            "source"      : c.get("source", ""),
                            "page"        : c.get("page"),
                            "heading"     : c.get("heading", ""),
                            "section_path": c.get("section_path", ""),
                            "chunk_type"  : c.get("type", "text"),
                        }
                        for c in item.get_citations()
                    ]
                    image_urls = [
                        "/images/" + os.path.basename(p)
                        for p in item.get_images()
                        if p and os.path.exists(p)
                    ]
                    payload = json.dumps({
                        "done"       : True,
                        "query_type" : item.query_type,
                        "citations"  : citations,
                        "image_urls" : image_urls,
                        "usage"      : item.usage,
                    })
                yield f"data: {payload}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/session/clear")
def clear_session(req: ClearRequest):
    if req.session_id in state.sessions:
        state.sessions[req.session_id].clear()
    return {"status": "cleared", "session_id": req.session_id}


@app.delete("/collection", response_model=WipeResponse)
def wipe_collection():
    state.store.reset_collection()
    state.bm25.reset()
    state.parent_store.clear()
    state.sessions.clear()

    if Path(PARENT_STORE_PATH).exists():
        Path(PARENT_STORE_PATH).unlink()

    rebuild_chain()

    return WipeResponse(
        status  = "wiped",
        message = "Knowledge base cleared. Upload new documents to start fresh.",
    )


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
        workers = 1,
    )