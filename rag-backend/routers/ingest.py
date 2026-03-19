# routers/ingest.py
#
# CHANGES:
#   - JWT protection on all endpoints via Depends(get_current_user)
#   - DELETE /ingest/{filename} added — deletes vectors + BM25 + hash entry
#   - Loaders now instantiated with file_path (matching original BaseLoader signature)
#   - chunk_hierarchical() returns only children (parent inline)
#   - Hash registry kept for duplicate prevention
#   - PIN FOCUS: optional source filter passed to retriever via session

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from auth.dependencies import get_current_user
from config import settings
from services import rag_service as _svc
from ingestion.csv_loader  import CSVLoader
from ingestion.pdf_loader  import PDFLoader
from ingestion.text_loader import TextLoader
from ingestion.xlsx_loader import XLSXLoader
from schemas import DeleteFileResponse, IngestResponse, IngestStatusResponse
from services import rag_service

router = APIRouter(tags=["ingest"])

# ── Hash registry ─────────────────────────────────────────────
_HASH_FILE = Path(settings.qdrant_path).parent / "file_hashes.json"


def _load_hashes() -> dict:
    if _HASH_FILE.exists():
        try:
            return json.loads(_HASH_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_hashes(hashes: dict) -> None:
    _HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HASH_FILE.write_text(json.dumps(hashes, indent=2))


def _wipe_hashes() -> None:
    if _HASH_FILE.exists():
        _HASH_FILE.unlink()


def _remove_hash_for_file(filename: str) -> None:
    hashes  = _load_hashes()
    updated = {h: f for h, f in hashes.items() if f != filename}
    _save_hashes(updated)


# ── Loader dispatch ───────────────────────────────────────────
# Each loader is instantiated with file_path (BaseLoader.__init__ requires it)

def _get_loader(tmp_path: str, filename: str):
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return PDFLoader(tmp_path)
    if ext == ".csv":
        return CSVLoader(tmp_path)
    if ext == ".xlsx":
        return XLSXLoader(tmp_path)
    if ext == ".txt":
        return TextLoader(tmp_path)
    return None


# ── Core ingest logic (runs in threadpool) ────────────────────

def _ingest_files_sync(file_paths: list[tuple[str, str]]) -> dict:
    """
    file_paths: list of (tmp_path, original_filename)
    """
    hashes       = _load_hashes()
    chunker      = _svc.get_chunker()
    vector_store = rag_service.get_vector_store()
    bm25_store   = rag_service.get_bm25_store()

    files_indexed: list[str] = []
    skipped      : list[str] = []
    all_children : list[dict] = []

    for tmp_path, filename in file_paths:
        # ── 1. Duplicate check ────────────────────────────
        raw   = Path(tmp_path).read_bytes()
        fhash = hashlib.sha256(raw).hexdigest()

        if fhash in hashes:
            print(f"  [INGEST] Skipping duplicate: {filename}")
            skipped.append(filename)
            continue

        # ── 2. Load ───────────────────────────────────────
        loader = _get_loader(tmp_path, filename)
        if not loader:
            print(f"  [INGEST] Unsupported file type: {filename}")
            skipped.append(filename)
            continue

        try:
            blocks = loader.load()
        except Exception as e:
            print(f"  [INGEST] Load failed for {filename}: {e}")
            skipped.append(filename)
            continue

        if not blocks:
            skipped.append(filename)
            continue

        # Ensure source is stamped with the original filename
        for b in blocks:
            b["source"] = filename

        # ── 3. Chunk — strategy determined by CHUNKER setting ────
        # HierarchicalChunker  → chunk_hierarchical() returns flat list with parent_content inline
        # RecursiveChunker / FixedSizeChunker → chunk_documents() returns flat list (no parent)
        from ingestion.chunker import HierarchicalChunker
        if isinstance(chunker, HierarchicalChunker):
            children = chunker.chunk_hierarchical(blocks)
        else:
            children = chunker.chunk_documents(blocks)

        # ── 4. Collect ────────────────────────────────────
        all_children.extend(children)
        files_indexed.append(filename)
        hashes[fhash] = filename

    # ── 5. Index ──────────────────────────────────────────
    if all_children:
        vector_store.add_documents(all_children)
        bm25_store.add(all_children)

    _save_hashes(hashes)

    return {
        "files_indexed": files_indexed,
        "skipped"      : skipped,
        "total_chunks" : len(all_children),
        "total_parents": len(all_children),
    }


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    files       : list[UploadFile] = File(...),
    current_user: dict             = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    tmp_dir    = Path("/tmp") / f"rag_ingest_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_paths : list[tuple[str, str]] = []

    try:
        for upload in files:
            tmp_path = tmp_dir / upload.filename
            content  = await upload.read()
            tmp_path.write_bytes(content)
            file_paths.append((str(tmp_path), upload.filename))

        result = await run_in_threadpool(_ingest_files_sync, file_paths)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return IngestResponse(
        status        = "ok",
        files_indexed = result["files_indexed"],
        total_chunks  = result["total_chunks"],
        total_parents = result["total_parents"],
        message       = (
            f"Indexed {len(result['files_indexed'])} file(s). "
            f"Skipped {len(result['skipped'])} duplicate(s)."
        ),
    )


@router.delete("/ingest/{filename}", response_model=DeleteFileResponse)
async def delete_file(
    filename    : str,
    current_user: dict = Depends(get_current_user),
):
    vector_store = rag_service.get_vector_store()
    sources = vector_store.list_sources()
    if filename not in sources:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found in the knowledge base.",
        )

    result = await run_in_threadpool(rag_service.delete_file_from_stores, filename)
    _remove_hash_for_file(filename)

    return DeleteFileResponse(
        status          = "ok",
        filename        = filename,
        vectors_deleted = result["vectors_deleted"],
        message         = (
            f"Deleted '{filename}': "
            f"{result['vectors_deleted']} vectors removed."
        ),
    )


@router.get("/ingest/status/{task_id}", response_model=IngestStatusResponse)
async def ingest_status(
    task_id     : str,
    current_user: dict = Depends(get_current_user),
):
    task = rag_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return IngestStatusResponse(**task)