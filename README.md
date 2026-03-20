# ✦ RAG Chatbot

> A production-ready, fully containerized **Retrieval-Augmented Generation (RAG)** chatbot with a sleek React frontend, FastAPI backend, hybrid search, JWT authentication, real-time streaming, and support for multiple LLM and vector store providers.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Groq%20%7C%20Ollama-orange" />
  <img src="https://img.shields.io/badge/Vector%20DB-Qdrant%20%7C%20Pinecone-purple" />
  <img src="https://img.shields.io/badge/Version-2.2.0-brightgreen" />
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [RAG Pipeline Deep Dive](#-rag-pipeline-deep-dive)
- [Configuration & Providers](#-configuration--providers)
- [API Reference](#-api-reference)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Frontend UI](#-frontend-ui)

---

## 🌟 Overview

RAG Chatbot is a **full-stack, production-grade AI application** that lets users upload documents (PDF, TXT, CSV, XLSX) and have intelligent, cited conversations with their content. The system goes far beyond basic retrieval — it uses a **multi-stage pipeline** with hybrid BM25 + vector search, cross-encoder reranking, hierarchical chunking, query routing, and rolling conversation memory to deliver accurate, grounded answers.

The entire stack runs in Docker with a single `docker-compose up`, while remaining fully configurable for different cloud and local setups.

---

## ✨ Key Features

### 🔍 Advanced Retrieval
- **Hybrid Search** — BM25 sparse retrieval fused with dense vector search via Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores top candidates for precision
- **Hierarchical Chunking** — parent-child chunks preserve context while keeping retrieval granular
- **Query Expansion** — LLM-generated sub-questions to broaden recall before retrieval
- **Source Pinning** — users can pin retrieval to a single indexed file directly from the UI

### 📄 Rich Document Ingestion
- **PDF** — text blocks, heading detection (font-size heuristics), bullet detection, embedded table extraction (pdfplumber → Markdown), image extraction with page-context semantic descriptions
- **TXT** — paragraph-level splitting on double newlines
- **CSV** — row-batch chunking with full schema metadata
- **XLSX** — multi-sheet support with column type annotations and row-range metadata

### 🧠 Intelligent Query Routing
- **Chit-chat detection** — greetings, casual questions bypass the RAG pipeline entirely for faster, more natural responses
- **No-KB fallback** — responds from general LLM knowledge when no documents are indexed
- **Weak-context fallback** — gracefully falls back to general knowledge when reranker confidence is below threshold
- **Full RAG** — document-grounded answers with source citations when context is strong

### 💬 Conversation Memory
- **Sliding window** — maintains the last N turns as direct chat context
- **Rolling summary** — every 5 turns, the LLM produces a 2-sentence summary injected into the system prompt, replacing noisy regex-based Entity Memory

### 🔐 Authentication
- JWT-based signup/login with bcrypt password hashing
- SQLite user store (`users.db`) persisted via Docker volume
- Per-user isolated chat sessions — session ID derived from JWT `user_id`

### 🔄 Real-Time Streaming
- Server-Sent Events (SSE) stream tokens to the browser as they are generated
- Nginx configured with `proxy_buffering off` and `chunked_transfer_encoding on` to ensure zero-latency token delivery

### 🐳 Docker-First
- Multi-stage Dockerfiles for lean final images (builder stage installs deps, runtime stage is clean)
- `docker-compose.yml` orchestrates frontend (Nginx) + backend (Uvicorn) with shared data volumes
- HuggingFace model cache mounted at `data/hf_cache` and persisted across rebuilds

---

## 🏛 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser / Client                      │
│                  React 18 + Vite (port 5173)                 │
└────────────────────────┬────────────────────────────────────┘
                         │  HTTP / SSE
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Nginx Reverse Proxy (port 80)                   │
│  /api/*  →  FastAPI backend   (strips /api prefix)           │
│  /images → FastAPI static     (extracted PDF images)         │
│  /*      → React SPA build                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (port 8000)                     │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │  /auth   │  │  /chat   │  │  /ingest │  │    /kb     │  │
│  │ signup   │  │ /stream  │  │  POST    │  │  /stats    │  │
│  │ login    │  │ /clear   │  │  DELETE  │  │  /docs     │  │
│  └──────────┘  └────┬─────┘  └────┬─────┘  └────────────┘  │
│                     │             │                          │
│              ┌──────▼─────────────▼──────┐                  │
│              │       RAG Service          │                  │
│              │  Session Manager (TTL)     │                  │
│              └──────────────┬────────────┘                  │
│                             │                                │
│              ┌──────────────▼────────────────────────┐      │
│              │            RAG Chain                   │      │
│              │                                        │      │
│              │  1. QueryRouter (chit-chat / general / │      │
│              │                  document)             │      │
│              │  2. Query Expansion (sub-questions)    │      │
│              │  3. HybridRetriever                    │      │
│              │     ├─ BM25Store (sparse / keyword)   │      │
│              │     └─ VectorStore (dense / semantic)  │      │
│              │  4. RRF Fusion (k=60)                  │      │
│              │  5. CrossEncoder Reranker              │      │
│              │  6. Prompt Builder + LLM Stream        │      │
│              │  7. ChatHistory + RollingSummary       │      │
│              └────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────┐   ┌──────────────────────────┐    │
│  │     Embedder          │   │     Vector Store          │    │
│  │  HuggingFace BGE      │   │  Qdrant (local)           │    │
│  │  Ollama (optional)    │   │  Pinecone (cloud)         │    │
│  └──────────────────────┘   └──────────────────────────┘    │
│                                                              │
│  ┌──────────────────────┐   ┌──────────────────────────┐    │
│  │       LLM             │   │      Ingestion            │    │
│  │  Groq (cloud, fast)   │   │  PDF / TXT / CSV / XLSX  │    │
│  │  Ollama (local, free) │   │  Chunker (hierarchical   │    │
│  └──────────────────────┘   │   / recursive / fixed)    │    │
│                              └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     data/qdrant    data/bm25.pkl   data/users.db
    (vector index)  (sparse index)  (user accounts)
```

---

## 📁 Project Structure

```
amar5623-rag-chatbot/
├── docker-compose.yml
│
├── rag-backend/
│   ├── main.py               # FastAPI app, lifespan, routers, CORS, static files
│   ├── config.py             # Pydantic settings (env → typed config + legacy constants)
│   ├── schemas.py            # All Pydantic request/response models
│   ├── requirements.txt
│   ├── Dockerfile            # Multi-stage: builder + slim runtime
│   │
│   ├── auth/
│   │   ├── router.py         # POST /auth/signup, /auth/login
│   │   ├── jwt_handler.py    # HS256 JWT encode/decode (python-jose)
│   │   ├── dependencies.py   # get_current_user FastAPI dependency
│   │   └── user_store.py     # SQLite user persistence (bcrypt hashed passwords)
│   │
│   ├── chains/
│   │   └── rag_chain.py      # RAGChain orchestrator, QueryRouter, ChainResponse
│   │
│   ├── embeddings/
│   │   └── embedder.py       # BGE/HuggingFace + Ollama embedders, EmbedderFactory
│   │
│   ├── generation/
│   │   ├── groq_llm.py       # GroqLLM, ChatHistory, RollingSummary, LLMFactory
│   │   └── ollama_llm.py     # OllamaLLM (local inference)
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py     # PDF → text/headings/bullets, tables (pdfplumber), images (fitz)
│   │   ├── text_loader.py    # TXT → paragraph chunks
│   │   ├── csv_loader.py     # CSV → row-batch chunks with schema
│   │   ├── xlsx_loader.py    # XLSX → multi-sheet chunks
│   │   └── chunker.py        # Hierarchical / recursive / fixed chunking strategies
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py   # BM25 + vector fusion via RRF
│   │   ├── naive_retriever.py    # Vector-only retrieval (simpler fallback)
│   │   ├── bm25_store.py         # Persistent BM25 index (pickle), add/delete/rebuild
│   │   └── reranker.py           # CrossEncoder ms-marco reranker
│   │
│   ├── routers/
│   │   ├── chat.py           # POST /chat/stream (SSE), POST /session/clear
│   │   ├── ingest.py         # POST /ingest, DELETE /ingest/{file}, GET /ingest/status
│   │   └── kb.py             # GET /stats, /documents, DELETE /collection, GET /health
│   │
│   ├── services/
│   │   └── rag_service.py    # Singleton manager: embedder, vector store, BM25, sessions, tasks
│   │
│   └── utils/
│       ├── image_captioner.py  # Tesseract OCR wrapper for image-to-chunk conversion
│       └── table_parser.py     # ParsedTable (Markdown/JSON/stats), TableParser (CSV/XLSX/HTML)
│
└── rag-frontend/
    ├── Dockerfile            # Vite build → Nginx serve
    ├── nginx.conf            # SPA routing + /api proxy + SSE buffering headers
    ├── package.json          # React 18, react-markdown, remark-gfm, Vite
    ├── vite.config.js        # Dev proxy: /api → :8000, /images → :8000
    │
    └── src/
        ├── App.jsx           # Root: auth gate, layout, sidebar/chat state coordination
        ├── api.js            # All fetch calls, JWT helpers, SSE async generator
        ├── index.css         # CSS variables, dark theme, animations (pulse, glow)
        │
        ├── components/
        │   ├── AuthPage.jsx      # Login / Signup toggle form
        │   ├── ChatWindow.jsx    # Message list, textarea input, streaming status
        │   ├── MessageBubble.jsx # Markdown render, citations, image thumbnails, usage stats
        │   └── Sidebar.jsx       # File upload, KB stats, pin/delete file actions, sign-out
        │
        └── hooks/
            └── useChat.js        # SSE stream consumer, message state machine (token → done → error)
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | React 18, Vite 5 | SPA framework and fast dev server |
| **Markdown rendering** | react-markdown + remark-gfm | Rich LLM response rendering (tables, code, lists) |
| **API layer** | FastAPI 0.111 | Async REST API with automatic OpenAPI docs |
| **Auth** | python-jose, passlib[bcrypt] | JWT token creation/validation + password hashing |
| **User store** | SQLite (stdlib) | Lightweight persistent user accounts, no extra service |
| **LLM (cloud)** | Groq API (`llama-3.1-8b-instant`) | Extremely fast cloud inference |
| **LLM (local)** | Ollama (`llama3.2`) | Fully offline inference, no API key needed |
| **Embeddings (default)** | BAAI/bge-base-en-v1.5 via sentence-transformers | Asymmetric BGE retrieval, 768-dim, offline |
| **Embeddings (alt)** | Ollama `nomic-embed-text` | Local embedding alternative |
| **Vector store (default)** | Qdrant (local file mode) | Dense vector search, zero-config, persists to disk |
| **Vector store (alt)** | Pinecone | Cloud-managed vector index |
| **Sparse retrieval** | BM25 via rank-bm25 (pickled to disk) | Keyword-level matching, persistent across restarts |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Semantic re-scoring of retrieved candidates |
| **PDF parsing** | PyMuPDF (fitz), pdfplumber | Text/heading/image extraction + table → Markdown |
| **Spreadsheets** | pandas, openpyxl | CSV and XLSX multi-sheet ingestion |
| **Web server** | Nginx | Reverse proxy, SPA serving, SSE pass-through |
| **Containers** | Docker + Docker Compose | Single-command deployment, isolated environments |

---

## 🔬 RAG Pipeline Deep Dive

### Stage 1 — Document Ingestion

```
File Upload (multipart/form-data)
    │
    ▼
Loader (auto-selected by file extension)
    │  PDF  → text blocks, heading detection (font-size ratio heuristic),
    │          bullet detection, tables (pdfplumber → Markdown),
    │          images (fitz → saved to data/images/ with page-context descriptions)
    │  TXT  → paragraph split on double newlines
    │  CSV  → row batches with schema header
    │  XLSX → per-sheet row batches with column type annotations
    │
    ▼
Chunker (CHUNKER env var)
    │
    ├─ hierarchical (default)
    │     Parent chunks (1200 tokens) → stored in in-memory parent_store dict
    │     Child chunks  (300 tokens)  → stored in vector store + BM25
    │     Each child carries parent_id for context expansion at retrieval time
    │
    ├─ recursive  → LangChain RecursiveCharacterTextSplitter (flat)
    └─ fixed      → LangChain CharacterTextSplitter (flat, fastest)
    │
    ▼
Embedder → BGE embed_documents() [no query prefix — asymmetric by design]
    │
    ▼
VectorStore.upsert()     ←  Qdrant or Pinecone
BM25Store.add()          ←  Persistent BM25, rebuilt + saved to bm25.pkl
```

### Stage 2 — Chat & Retrieval (7-step pipeline)

```
User Question
    │
    ▼
① QueryRouter
   ├─ CHITCHAT  → direct LLM reply with casual system prompt, skip retrieval
   ├─ GENERAL   → LLM with general-knowledge fallback prompt (no KB or weak context)
   └─ DOCUMENT  → continue to full RAG pipeline ↓

② Query Expansion
   LLM generates N sub-questions from the original question to broaden recall

③ HybridRetriever (per expanded query)
   ├─ BM25Search   → top_k keyword matches from persistent BM25 index
   └─ VectorSearch → top_k semantic matches from Qdrant / Pinecone
   → RRF Fusion: score = Σ 1 / (k + rank_i)  where k = 60

④ Source Filter (optional)
   If the user pinned a file in the UI, results are filtered to that source only

⑤ CrossEncoder Reranker
   Re-scores top candidates with cross-encoder/ms-marco-MiniLM-L-6-v2
   Returns top 5 with float confidence scores
   If best_score < MIN_RERANK_SCORE → falls back to general knowledge

⑥ Prompt Builder
   Assembles: system_prompt + rolling_summary + sliding_window_history + context + question

⑦ LLM Stream
   Groq or Ollama streams tokens → yielded as SSE events → browser renders in real time
   Final ChainResponse carries: citations (source, page, heading, section_path), query_type, token usage
```

### BGE Asymmetric Retrieval

One of the more impactful design choices: the BGE embedder applies **different treatment for queries vs documents**:

```python
# Query embedding — adds retrieval prefix
embed_text("What is the revenue?")
# Encodes: "Represent this sentence for searching relevant passages: What is the revenue?"

# Document embedding — no prefix
embed_documents(["Q3 revenue was $1.2M..."])
# Encodes as-is
```

BGE models are explicitly trained this way and yield ~10% better MTEB retrieval scores vs symmetric MiniLM. The prefix is only added for queries; documents are indexed without it.

### Rolling Summary vs Entity Memory

The original entity memory approach extracted "named entities" with regex — in practice capturing stop words, dates, and generic phrases as "facts", polluting every system prompt with noise. The replacement:

```
Every 5 turns → LLM produces a 2-sentence summary of recent conversation
→ Injected as [Conversation summary: ...] block into the system prompt
→ Replaced with a fresh summary after the next 5 turns
```

This is opt-in (disabled by default for latency) and gives clean, meaningful context injection.

---

## ⚙️ Configuration & Providers

All configuration is driven by environment variables. Every core component is independently swappable:

### LLM Provider
```bash
LLM_PROVIDER=groq      # default — Groq cloud, fast, requires GROQ_API_KEY
LLM_PROVIDER=ollama    # local Ollama, requires `ollama serve` running
```

### Embedding Provider
```bash
EMBEDDER=huggingface   # default — BAAI/bge-base-en-v1.5, local, fully offline
EMBEDDER=ollama        # Ollama nomic-embed-text
```

### Vector Store
```bash
VECTOR_STORE=qdrant    # default — local file-based, no API key, persists to disk
VECTOR_STORE=pinecone  # cloud-managed, requires PINECONE_API_KEY
```

### Chunking Strategy
```bash
CHUNKER=hierarchical   # default — parent-child, best retrieval quality
CHUNKER=recursive      # flat recursive, good general-purpose
CHUNKER=fixed          # flat fixed-size, fastest ingestion
```

### Chunking Parameters
```bash
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHILD_CHUNK_SIZE=300
CHILD_CHUNK_OVERLAP=30
PARENT_CHUNK_SIZE=1200
PARENT_CHUNK_OVERLAP=100
TOP_K=20               # retrieval candidates before reranking
MIN_RERANK_SCORE=0.1   # fallback threshold
```

---

## 🌐 API Reference

### Auth
| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/signup` | `{email, password}` | Register a new user |
| `POST` | `/auth/login` | `{email, password}` | Login, returns `{access_token, user_id, email}` |

### Chat
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/chat/stream` | ✅ JWT | SSE streaming chat — body `{question}` |
| `POST` | `/session/clear` | ✅ JWT | Clear conversation history for current user |

### Ingestion
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/ingest` | ✅ JWT | Upload files (`multipart/form-data`), returns `{task_id}` |
| `GET` | `/ingest/status/{task_id}` | ✅ JWT | Poll async ingest progress `{status, progress, message, result}` |
| `DELETE` | `/ingest/{filename}` | ✅ JWT | Delete a specific file from all indexes |

### Knowledge Base
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/stats` | ❌ public | Vector count, BM25 docs, models, indexed files |
| `GET` | `/documents` | ✅ JWT | List indexed filenames |
| `DELETE` | `/collection` | ✅ JWT | Wipe entire knowledge base |
| `GET` | `/health` | ❌ public | Service health check + version |

### SSE Stream Event Format
```
data: {"token": "The "}
data: {"token": "answer "}
data: {"token": "is..."}
data: {
  "done": true,
  "citations": [{"source": "doc.pdf", "page": 3, "heading": "Revenue", "section_path": "Results > Revenue"}],
  "query_type": "document",
  "usage": {"prompt_tokens": 820, "completion_tokens": 145}
}
```

---

## 🚀 Getting Started

### Prerequisites
- Docker + Docker Compose installed
- A [Groq API key](https://console.groq.com) (free tier available) **or** Ollama running locally

### 1. Clone & Configure

```bash
git clone https://github.com/amar5623/rag-chatbot.git
cd rag-chatbot

cp rag-backend/.env.example rag-backend/.env
```

Open `rag-backend/.env` and fill in your keys:

```env
GROQ_API_KEY=gsk_your_key_here
JWT_SECRET_KEY=a-long-random-string-change-this-in-production
```

### 2. Launch with Docker Compose

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Frontend (Nginx) | http://localhost |
| Backend API | http://localhost/api |
| Interactive API Docs | http://localhost/api/docs |

> On first run the BGE embedding model (~270 MB) is downloaded from HuggingFace and cached in `data/hf_cache`. Subsequent starts are instant.

### 3. Local Development (no Docker)

**Backend:**
```bash
cd rag-backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in your keys
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd rag-frontend
npm install
npm run dev
# → http://localhost:5173
```

### 4. Using the App

1. **Sign up** — create an account on the auth screen
2. **Upload documents** — drag & drop or select PDF, TXT, CSV, or XLSX files via the sidebar
3. **Wait for indexing** — progress is polled and displayed in the sidebar
4. **Chat** — ask questions about your documents in natural language
5. **Pin a file** — click 📌 next to any indexed file to restrict retrieval to that source only
6. **Delete files** — click 🗑 to remove a file from all indexes (vector + BM25)
7. **Clear chat** — reset conversation memory while keeping the knowledge base intact

---

## 🔐 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | `""` | Groq cloud API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model identifier |
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model for both LLM and embeddings |
| `EMBEDDER` | `huggingface` | `huggingface` or `ollama` |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace model path |
| `EMBEDDING_DIM` | `768` | Embedding dimensions (must match model) |
| `VECTOR_STORE` | `qdrant` | `qdrant` or `pinecone` |
| `PINECONE_API_KEY` | `""` | Pinecone cloud API key |
| `PINECONE_INDEX` | `rag-index` | Pinecone index name |
| `PINECONE_CLOUD` | `aws` | Pinecone cloud provider |
| `PINECONE_REGION` | `us-east-1` | Pinecone region |
| `CHUNKER` | `hierarchical` | `hierarchical`, `recursive`, or `fixed` |
| `CHUNK_SIZE` | `500` | Flat chunk size in tokens |
| `CHUNK_OVERLAP` | `50` | Flat chunk overlap |
| `CHILD_CHUNK_SIZE` | `300` | Child chunk size (hierarchical) |
| `PARENT_CHUNK_SIZE` | `1200` | Parent chunk size (hierarchical) |
| `TOP_K` | `20` | Number of retrieval candidates before reranking |
| `MIN_RERANK_SCORE` | `0.1` | Minimum reranker score before general-knowledge fallback |
| `JWT_SECRET_KEY` | `change-me` | **Must be changed in production** — used to sign all tokens |
| `MAX_TURNS` | `20` | Sliding window chat history turn count |
| `SESSION_TTL` | `3600` | Chat session time-to-live in seconds |
| `SESSION_MAX` | `100` | Maximum concurrent sessions |
| `HF_TOKEN` | `""` | HuggingFace Hub token (for private/gated models) |

---

## 🖥 Frontend UI

The React frontend is built with a dark theme, monospace design language, and zero external UI library dependencies — all styling uses hand-crafted CSS variables with CSS animations.

### Component Overview

**`AuthPage`** — Toggle between login and signup with inline error display. On success, JWT is stored in `localStorage` and the user is transitioned to the chat view.

**`Sidebar`** — Collapsible panel with:
- File upload zone (drag-and-drop or click)
- Animated KB status indicator (pulsing dot when KB is ready)
- KB stats panel: vector count, BM25 doc count, parent chunk count, embedding model, LLM model
- Indexed files list with per-file **pin** (📌) and **delete** (🗑) action buttons
- Sign-out button with current user email display

**`ChatWindow`** — Auto-scrolling message list with:
- Status text animation ("Searching documents…") during retrieval
- `Shift+Enter` for newlines, `Enter` to send
- Abort support via `abortRef`

**`MessageBubble`** — Full GFM Markdown rendering (tables, code blocks, numbered lists) via react-markdown + remark-gfm, inline citation badges with source/page info, image thumbnails for extracted PDF figures, and a token usage footer per assistant message.

**`useChat` hook** — Async generator that consumes the SSE stream and drives a message state machine:
```
{ streaming: true, content: "" }
  → token events: { streaming: true, content: "The answer is..." }
  → done event:   { streaming: false, citations: [...], query_type, usage }
  → error event:  { streaming: false, content: "⚠️ ...", isError: true }
```

---

<p align="center">
  Built with ♥ using FastAPI · React · Qdrant · Groq · BGE · Docker
</p>
