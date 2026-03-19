# schemas.py
# All Pydantic request and response models for the RAG API.

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    question   : str = Field(..., min_length=1, max_length=2000)
    session_id : str = Field(default="default", max_length=64)


class ClearRequest(BaseModel):
    session_id : str = Field(default="default")


# ── Responses ─────────────────────────────────────────────

class Citation(BaseModel):
    source      : str
    page        : int | None = None
    heading     : str        = ""
    section_path: str        = ""
    chunk_type  : str        = "text"


class ChatResponse(BaseModel):
    answer     : str
    query_type : str
    citations  : list[Citation] = []
    usage      : dict           = {}
    session_id : str


class IngestResponse(BaseModel):
    status        : str
    files_indexed : list[str]
    total_chunks  : int
    total_parents : int
    message       : str


class StatsResponse(BaseModel):
    total_vectors  : int
    bm25_docs      : int
    parent_count   : int
    indexed_files  : list[str]
    embedding_model: str
    llm_model      : str
    collection     : str


class HealthResponse(BaseModel):
    status          : str
    version         : str = "2.2.0"
    groq_configured : bool


class DocumentsResponse(BaseModel):
    files       : list[str]
    total_files : int


class WipeResponse(BaseModel):
    status  : str
    message : str


class DeleteFileResponse(BaseModel):
    status        : str
    filename      : str
    vectors_deleted: int
    message       : str


class IngestStatusResponse(BaseModel):
    status  : str
    progress: int = 0
    message : str = ""
    result  : dict = {}


__all__ = [
    "ChatRequest", "ClearRequest",
    "Citation", "ChatResponse", "IngestResponse",
    "StatsResponse", "HealthResponse", "DocumentsResponse",
    "WipeResponse", "DeleteFileResponse", "IngestStatusResponse",
]