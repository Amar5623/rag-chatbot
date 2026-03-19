# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from services.rag_service import startup
from routers import chat, ingest, kb
from auth.router import router as auth_router, init_user_store
from auth.user_store import UserStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise RAG singletons
    await startup()

    # Initialise user store
    data_dir = Path(settings.qdrant_path).parent
    user_store = UserStore(path=str(data_dir / "users.db"))
    init_user_store(user_store)

    yield


app = FastAPI(
    title="RAG Chatbot API",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,          # needed for Authorization header
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted images as static files
images_dir = Path(__file__).parent / "data" / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

app.include_router(auth_router)
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(kb.router)