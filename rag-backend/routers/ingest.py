# routers/ingest.py
import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ingestion.chunker    import ChunkerFactory, HierarchicalChunker
from ingestion.csv_loader  import CSVLoader
from ingestion.pdf_loader  import PDFLoader
from ingestion.text_loader import TextLoader
from ingestion.xlsx_loader import XLSXLoader
from services import rag_service
from config import settings

router = APIRouter(prefix="/ingest", tags=["ingest"])

_LOADERS = {
    ".pdf" : PDFLoader,
    ".csv" : CSVLoader,
    ".xlsx": XLSXLoader,
    ".txt" : TextLoader,
}

_chunker: HierarchicalChunker = None


def _get_chunker() -> HierarchicalChunker:
    global _chunker
    if _chunker is None:
        _chunker = ChunkerFactory.get("hierarchical")
    return _chunker


# ── Deduplication helpers ─────────────────────────────────────

def _hash_db_path() -> str:
    return str(Path(settings.qdrant_path).parent / "hashes.db")


def _init_hash_db() -> None:
    with sqlite3.connect(_hash_db_path()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hashes (
                hash     TEXT PRIMARY KEY,
                filename TEXT NOT NULL
            )
        """)
        conn.commit()


def _is_duplicate(file_hash: str) -> bool:
    _init_hash_db()
    with sqlite3.connect(_hash_db_path()) as conn:
        return conn.execute(
            "SELECT 1 FROM hashes WHERE hash = ?", (file_hash,)
        ).fetchone() is not None


def _register_hash(file_hash: str, filename: str) -> None:
    with sqlite3.connect(_hash_db_path()) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO hashes (hash, filename) VALUES (?, ?)",
            (file_hash, filename),
        )
        conn.commit()


def _wipe_hashes() -> None:
    _init_hash_db()
    with sqlite3.connect(_hash_db_path()) as conn:
        conn.execute("DELETE FROM hashes")
        conn.commit()


# ── Background ingestion ──────────────────────────────────────

async def _run_ingestion(
    files_data: list[tuple[str, str, bytes]],
    task_id: str,
) -> None:
    rag_service.set_task(task_id, "running", 0, "Starting ingestion…")

    vector_store = rag_service.get_vector_store()
    bm25_store   = rag_service.get_bm25_store()
    parent_store = rag_service.get_parent_store()
    chunker      = _get_chunker()

    all_children: list[dict] = []
    all_parents:  dict       = {}
    indexed:  list[str]      = []
    skipped:  list[str]      = []
    total = len(files_data)

    for i, (filename, ext, data) in enumerate(files_data):
        progress = int((i / total) * 80)
        rag_service.set_task(task_id, "running", progress, f"Processing {filename}…")

        # ── Dedup check ───────────────────────────────────────
        file_hash = hashlib.sha256(data).hexdigest()
        if _is_duplicate(file_hash):
            skipped.append(filename)
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            docs = _LOADERS[ext](tmp_path).load()
            for d in docs:
                d["source"] = filename

            children, parents = chunker.chunk_hierarchical(docs)
            for c in children:
                c["source"] = filename

            all_children.extend(children)
            all_parents.update(parents)
            indexed.append(filename)
            _register_hash(file_hash, filename)

        except Exception as e:
            rag_service.set_task(task_id, "running", progress, f"Error on {filename}: {e}")
        finally:
            os.unlink(tmp_path)

    # ── Persist ───────────────────────────────────────────────
    if all_children:
        rag_service.set_task(task_id, "running", 85, "Indexing vectors…")
        vector_store.add_documents(all_children)

        rag_service.set_task(task_id, "running", 90, "Updating BM25…")
        bm25_store.add(all_children)

        rag_service.set_task(task_id, "running", 95, "Saving parents…")
        parent_store.add(all_parents)

    rag_service.set_task(
        task_id, "done", 100,
        f"Indexed {len(indexed)} file(s). Skipped {len(skipped)} duplicate(s).",
        result={
            "files_indexed" : indexed,
            "files_skipped" : skipped,
            "total_chunks"  : len(all_children),
            "total_parents" : len(all_parents),
        },
    )


# ── Endpoints ─────────────────────────────────────────────────

@router.post("")
async def ingest_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(400, "No files provided")

    files_data: list[tuple[str, str, bytes]] = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in _LOADERS:
            raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {list(_LOADERS)}")
        data = await f.read()
        files_data.append((f.filename, ext, data))

    task_id = uuid4().hex
    rag_service.set_task(task_id, "queued", 0, "Queued for processing")
    background_tasks.add_task(_run_ingestion, files_data, task_id)

    return {"task_id": task_id, "status": "queued"}


@router.get("/status/{task_id}")
async def ingest_status(task_id: str):
    task = rag_service.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return task