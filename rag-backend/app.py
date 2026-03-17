# app.py
#
# CHANGES vs original:
#   - Ingestion now uses HierarchicalChunker (chunk_hierarchical)
#     and saves parent_store to disk (pickle) across sessions
#   - BM25 loaded from disk via BM25Store — no more silent degradation
#   - parent_store passed to HybridRetriever + NaiveRetriever
#   - build_chain() passes parent_store into RAGChain
#   - QueryRouter active — chitchat gets instant responses
#   - Pinecone + Ollama removed from UI (still in codebase, not shown)
#   - "Index data/ folder" and "Load existing KB" modes still work
#   - All styling unchanged

import os
import sys
import time
import pickle
import tempfile
from pathlib import Path

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from vectorstore.qdrant_store   import QdrantVectorStore
from embeddings.embedder        import EmbedderFactory
from generation.groq_llm        import LLMFactory
from retrieval.naive_retriever  import NaiveRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.bm25_store       import BM25Store
from retrieval.reranker         import Reranker
from chains.rag_chain           import RAGChain
from ingestion.pdf_loader       import PDFLoader
from ingestion.csv_loader       import CSVLoader
from ingestion.xlsx_loader      import XLSXLoader
from ingestion.text_loader      import TextLoader
from ingestion.chunker          import ChunkerFactory, HierarchicalChunker
from config import (
    QDRANT_PATH, QDRANT_COLLECTION,
    BM25_PATH, PARENT_STORE_PATH,
)


# ─────────────────────────────────────────────────────────
# STYLING  (unchanged from original)
# ─────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e6e3; }

section[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #a09fbb !important;
    font-size: 0.82rem;
}
.rag-logo {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.6rem;
    background: linear-gradient(135deg, #c9a96e 0%, #f0d5a0 50%, #c9a96e 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.rag-subtitle {
    font-size: 0.72rem; color: #504f6a; letter-spacing: 0.12em;
    text-transform: uppercase; margin-top: 2px; margin-bottom: 20px;
}
.sidebar-section {
    font-family: 'Syne', sans-serif; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase; color: #c9a96e;
    margin: 18px 0 8px 0; padding-bottom: 4px; border-bottom: 1px solid #1e1e2e;
}
.msg-user {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2a2a4a; border-radius: 16px 16px 4px 16px;
    padding: 14px 18px; margin-left: 15%; color: #e8e6e3;
    font-size: 0.94rem; line-height: 1.6; margin-bottom: 12px;
}
.msg-assistant {
    background: linear-gradient(135deg, #111118, #13131f);
    border: 1px solid #1e1e30; border-left: 3px solid #c9a96e;
    border-radius: 4px 16px 16px 16px; padding: 14px 18px;
    margin-right: 15%; color: #d4d2cf; font-size: 0.94rem;
    line-height: 1.7; margin-bottom: 12px;
}
.msg-label {
    font-family: 'Syne', sans-serif; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 6px;
}
.msg-label-user { color: #5c5b7a; }
.msg-label-assistant { color: #c9a96e; }
.citations-box {
    margin-top: 12px; padding: 10px 14px; background: #0d0d16;
    border: 1px solid #1a1a28; border-radius: 8px; font-size: 0.78rem; color: #6b6a88;
}
.citations-box strong {
    color: #c9a96e; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
}
.cite-item {
    display: inline-block; background: #161624; border: 1px solid #252538;
    border-radius: 4px; padding: 2px 8px; margin: 2px 3px;
    font-size: 0.75rem; color: #8a89a8;
}
.cite-heading { font-size: 0.68rem; color: #504f6a; font-style: italic; }
.welcome-title {
    font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #c9a96e, #f0d5a0, #c9a96e);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.feature-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 20px; }
.feature-card { background: #0f0f18; border: 1px solid #1e1e2e; border-radius: 10px; padding: 14px 16px; }
.feature-title { font-family: 'Syne', sans-serif; font-size: 0.78rem; font-weight: 700; color: #c9a96e; }
.feature-desc  { font-size: 0.78rem; color: #504f6a; }
.status-badge  { display: inline-flex; align-items: center; gap: 5px; font-size: 0.72rem; padding: 3px 10px; border-radius: 20px; }
.status-ready  { background:#0d2010; border:1px solid #1a4020; color:#4caf70; }
.status-loading{ background:#1a1000; border:1px solid #3a2800; color:#c9a96e; }
.kb-badge { background:#0d1520; border:1px solid #1a3040; color:#5aaacc; padding:3px 10px; border-radius:20px; }
.metric-pill { background:#0f0f18; border:1px solid #1e1e2e; border-radius:6px; padding:6px 12px; font-size:0.75rem; color:#6b6a88; }
.metric-pill span { color:#c9a96e; font-weight:600; }
.stTextInput input { background:#0f0f18 !important; border:1px solid #1e1e2e !important; border-radius:10px !important; color:#e8e6e3 !important; }
.stButton > button { background:linear-gradient(135deg,#c9a96e,#a8854a) !important; color:#0a0a0f !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; border-radius:8px !important; border:none !important; }
.image-result-label { font-family:'Syne',sans-serif; font-size:0.65rem; font-weight:700; color:#5aaacc; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
footer { visibility: hidden; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "chain"        : None,
        "messages"     : [],
        "indexed_files": [],
        "total_chunks" : 0,
        "chain_ready"  : False,
        "store"        : None,
        "parent_store" : {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────
# CACHED RESOURCES  (loaded once per session)
# ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embedder():
    return EmbedderFactory.get("huggingface")

@st.cache_resource(show_spinner=False)
def get_reranker():
    return Reranker()

@st.cache_resource(show_spinner=False)
def get_chunker():
    return ChunkerFactory.get("hierarchical")


# ─────────────────────────────────────────────────────────
# PARENT STORE PERSISTENCE
# ─────────────────────────────────────────────────────────

def load_parent_store() -> dict:
    """Load parent store from disk. Returns empty dict if not found."""
    if Path(PARENT_STORE_PATH).exists():
        try:
            with open(PARENT_STORE_PATH, "rb") as f:
                data = pickle.load(f)
            print(f"  [APP] Loaded {len(data)} parents from disk")
            return data
        except Exception as e:
            print(f"  [APP] Parent store load failed: {e}")
    return {}


def save_parent_store(store: dict) -> None:
    """Save parent store to disk."""
    try:
        with open(PARENT_STORE_PATH, "wb") as f:
            pickle.dump(store, f)
    except Exception as e:
        print(f"  [APP] Parent store save failed: {e}")


def wipe_parent_store() -> None:
    """Delete parent store file from disk."""
    if Path(PARENT_STORE_PATH).exists():
        Path(PARENT_STORE_PATH).unlink()


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def ingest_file(
    file,
    chunker: HierarchicalChunker,
) -> tuple[list[dict], dict]:
    """
    Load, parse, and chunk a single uploaded file.
    Returns (children, parents) for hierarchical chunker.
    Falls back to flat chunking for non-hierarchical chunkers.
    """
    ext = os.path.splitext(file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        loaders = {
            ".pdf" : PDFLoader,
            ".csv" : CSVLoader,
            ".xlsx": XLSXLoader,
            ".txt" : TextLoader,
        }
        if ext not in loaders:
            return [], {}

        docs   = loaders[ext](tmp_path).load()
        # Fix source name to uploaded filename
        for d in docs:
            d["source"] = file.name

        if isinstance(chunker, HierarchicalChunker):
            children, parents = chunker.chunk_hierarchical(docs)
        else:
            children = chunker.chunk_documents(docs)
            parents  = {}

        for c in children:
            c["source"] = file.name

        return children, parents

    finally:
        os.unlink(tmp_path)


def ingest_folder(
    folder_path: str,
    chunker: HierarchicalChunker,
) -> tuple[list[dict], dict]:
    """Ingest all supported files from a local folder."""
    ext_map = {
        ".pdf" : PDFLoader,
        ".csv" : CSVLoader,
        ".xlsx": XLSXLoader,
        ".txt" : TextLoader,
    }
    all_children: list[dict] = []
    all_parents:  dict       = {}

    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[-1].lower() in ext_map
    ]

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        ext   = os.path.splitext(fname)[-1].lower()
        try:
            docs = ext_map[ext](fpath).load()
            for d in docs:
                d["source"] = fname

            if isinstance(chunker, HierarchicalChunker):
                children, parents = chunker.chunk_hierarchical(docs)
            else:
                children = chunker.chunk_documents(docs)
                parents  = {}

            for c in children:
                c["source"] = fname

            all_children.extend(children)
            all_parents.update(parents)

        except Exception as e:
            st.warning(f"Could not load {fname}: {e}")

    return all_children, all_parents


def check_existing_kb() -> int:
    """Return vector count in existing Qdrant KB, 0 if empty/missing."""
    try:
        store = QdrantVectorStore(embedder=get_embedder())
        return store.count()
    except Exception:
        return 0


def build_chain(
    store,
    llm_provider     : str,
    retriever_choice : str,
    use_reranker     : bool,
    retrieve_k       : int,
    rerank_k         : int,
    cite_sources     : bool,
    parent_store     : dict,
) -> RAGChain:
    embedder = get_embedder()
    reranker = get_reranker() if use_reranker else None

    if "Hybrid" in retriever_choice:
        retriever = HybridRetriever(
            vector_store = store,
            embedder     = embedder,
            top_k        = retrieve_k,
            parent_store = parent_store,  # ← NEW
        )
    else:
        retriever = NaiveRetriever(
            vector_store = store,
            embedder     = embedder,
            top_k        = retrieve_k,
            parent_store = parent_store,  # ← NEW
        )

    llm = LLMFactory.get(llm_provider)

    chain = RAGChain(
        llm           = llm,
        vector_store  = store,
        retriever     = retriever,
        reranker      = reranker,
        use_reranker  = use_reranker,
        retrieve_top_k= retrieve_k,
        rerank_top_k  = rerank_k,
        cite_sources  = cite_sources,
        parent_store  = parent_store,  # ← NEW
    )
    return chain


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="rag-logo">RAG Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-subtitle">Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

    # Status
    existing_kb = check_existing_kb()
    if st.session_state.chain_ready:
        st.markdown('<span class="status-badge status-ready">● Pipeline Ready</span>', unsafe_allow_html=True)
    elif existing_kb > 0:
        st.markdown(f'<span class="kb-badge">💾 Saved KB: {existing_kb} chunks</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-loading">○ No Knowledge Base</span>', unsafe_allow_html=True)

    # ── Pipeline config ───────────────────────────────────
    st.markdown('<div class="sidebar-section">⚙️ Pipeline Config</div>', unsafe_allow_html=True)

    llm_provider     = st.selectbox("LLM",       ["groq"])
    retriever_choice = st.selectbox("Retriever",  ["Hybrid (BM25 + Dense)", "Naive (Dense only)"])
    use_reranker     = st.toggle("Cross-Encoder Reranker", value=True)
    retrieve_k       = st.slider("Fetch top-k",       5, 50, 20)
    rerank_k         = st.slider("Keep after rerank", 1, 10, 5)
    cite_sources     = st.toggle("Source citations",  value=True)

    # ── Knowledge Base modes ──────────────────────────────
    st.markdown('<div class="sidebar-section">📚 Knowledge Base</div>', unsafe_allow_html=True)

    kb_mode = st.radio(
        "Mode",
        ["🔄 Load existing KB", "📁 Index data/ folder",
         "📤 Upload new files", "🗑 Wipe & rebuild"],
        help=(
            "Load existing KB → resume from last session\n"
            "Index data/ folder → build KB from your data folder once\n"
            "Upload new files → add to existing KB without wiping\n"
            "Wipe & rebuild → start fresh"
        )
    )

    uploaded_files = None
    if kb_mode in ["📤 Upload new files", "🗑 Wipe & rebuild"]:
        uploaded_files = st.file_uploader(
            "Files", type=["pdf", "csv", "xlsx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    launch_btn = st.button("Launch", use_container_width=True)

    # ── Memory / reset ────────────────────────────────────
    if st.session_state.chain_ready:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chain.reset_memory()
                st.rerun()
        with c2:
            if st.button("⚙️ Reconfigure", use_container_width=True):
                st.session_state.chain_ready = False
                st.session_state.chain       = None
                st.rerun()

        if st.button("🧠 Clear Chat (keep summary)", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chain.llm.history.clear_turns_only()
            st.rerun()

    # ── Stats ─────────────────────────────────────────────
    if st.session_state.chain_ready:
        st.markdown('<div class="sidebar-section">📊 Stats</div>', unsafe_allow_html=True)
        info = st.session_state.chain.get_info()
        for label, val in [
            ("KB chunks",    st.session_state.total_chunks),
            ("Files loaded", len(st.session_state.indexed_files)),
            ("Parents",      info.get("parent_store", 0)),
            ("Chat turns",   info["history_turns"]),
            ("Model",        info["llm"]["model"].split("-")[0] + "…"),
        ]:
            st.markdown(
                f'<div class="metric-pill">{label} <span>{val}</span></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────
# LAUNCH HANDLER
# ─────────────────────────────────────────────────────────

if launch_btn:
    embedder = get_embedder()
    chunker  = get_chunker()

    with st.spinner("Building pipeline…"):
        try:
            # ── Vector store ──────────────────────────────
            store = QdrantVectorStore(embedder=embedder)
            if kb_mode == "🗑 Wipe & rebuild":
                store.reset_collection()
                BM25Store(path=BM25_PATH).reset()
                wipe_parent_store()

            new_children: list[dict] = []
            new_parents:  dict       = {}
            indexed_files             = list(st.session_state.indexed_files)

            # ── Ingest based on mode ──────────────────────
            if kb_mode == "📁 Index data/ folder":
                data_path = os.path.join(os.path.dirname(__file__), "data")
                if os.path.exists(data_path):
                    new_children, new_parents = ingest_folder(data_path, chunker)
                    indexed_files = os.listdir(data_path)
                    st.info(f"Found {len(new_children)} chunks in data/ folder.")
                else:
                    st.error("No data/ folder found at project root.")
                    st.stop()

            elif kb_mode in ["📤 Upload new files", "🗑 Wipe & rebuild"]:
                if uploaded_files:
                    progress = st.progress(0)
                    for i, f in enumerate(uploaded_files):
                        progress.progress(i / len(uploaded_files), text=f"Loading {f.name}…")
                        children, parents = ingest_file(f, chunker)
                        new_children.extend(children)
                        new_parents.update(parents)
                        if f.name not in indexed_files:
                            indexed_files.append(f.name)
                    progress.empty()
                elif kb_mode == "📤 Upload new files":
                    st.warning("Please upload at least one file.")
                    st.stop()

            # ── Load parent store from disk ───────────────
            parent_store = load_parent_store()
            parent_store.update(new_parents)

            # ── Index new chunks ──────────────────────────
            if new_children:
                store.add_documents(new_children)
                # Persistent BM25 — add() appends + saves
                bm25 = BM25Store(path=BM25_PATH)
                bm25.add(new_children)
                save_parent_store(parent_store)

            # ── Total KB size ─────────────────────────────
            total = store.count()

            # ── Build chain ───────────────────────────────
            chain = build_chain(
                store            = store,
                llm_provider     = llm_provider,
                retriever_choice = retriever_choice,
                use_reranker     = use_reranker,
                retrieve_k       = retrieve_k,
                rerank_k         = rerank_k,
                cite_sources     = cite_sources,
                parent_store     = parent_store,
            )

            # Index BM25 on retriever too (for "Load existing KB" flow)
            if new_children and hasattr(chain.retriever, "index_chunks"):
                chain.retriever.index_chunks(new_children)

            # ── Save state ────────────────────────────────
            st.session_state.chain         = chain
            st.session_state.store         = store
            st.session_state.chain_ready   = True
            st.session_state.total_chunks  = total
            st.session_state.indexed_files = indexed_files
            st.session_state.parent_store  = parent_store

            if kb_mode == "🗑 Wipe & rebuild":
                st.session_state.messages = []

            st.success(
                f"✅ Ready! KB has **{total} chunks** | "
                f"**{len(parent_store)} parents** | "
                f"**{len(indexed_files)} file(s)**."
            )
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ {e}")
            raise e


# ─────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────

st.markdown(
    '<h1 style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:2rem;'
    'letter-spacing:-0.03em;color:#e8e6e3;margin-bottom:2px;">Document Intelligence</h1>'
    '<p style="color:#504f6a;font-size:0.85rem;margin-top:0;">Ask questions about your knowledge base</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── Welcome screen ────────────────────────────────────────
if not st.session_state.chain_ready:
    existing_kb = check_existing_kb()
    kb_hint = (
        f'<strong style="color:#5aaacc">💾 You have a saved knowledge base with '
        f'{existing_kb} chunks.</strong><br>'
        f'Select <em>Load existing KB</em> in the sidebar to resume instantly.'
        if existing_kb > 0
        else "Upload files, index your data/ folder, or load a previous session from the sidebar."
    )
    st.markdown(f"""
    <div style="text-align:center;padding:40px 40px;max-width:640px;margin:0 auto;">
        <div class="welcome-title">Your Document Brain</div>
        <p style="color:#504f6a;font-size:0.9rem;line-height:1.7;margin:12px 0 28px;">{kb_hint}</p>
        <div class="feature-grid">
            <div class="feature-card"><div class="feature-title">🧩 Hierarchical chunks</div><div class="feature-desc">Small for retrieval, large for LLM context. Best quality answers.</div></div>
            <div class="feature-card"><div class="feature-title">🔍 Hybrid retrieval</div><div class="feature-desc">BGE + persistent BM25 fused with RRF, reranked by cross-encoder.</div></div>
            <div class="feature-card"><div class="feature-title">🤖 Smart routing</div><div class="feature-desc">Greetings answered instantly. Doc queries go through full retrieval.</div></div>
            <div class="feature-card"><div class="feature-title">💾 Persistent KB</div><div class="feature-desc">Index once, resume instantly. BM25 + parents saved across sessions.</div></div>
            <div class="feature-card"><div class="feature-title">🧠 Memory</div><div class="feature-desc">Sliding window keeps recent turns. Optional rolling summary.</div></div>
            <div class="feature-card"><div class="feature-title">🖼️ Image responses</div><div class="feature-desc">Retrieved images displayed alongside answers.</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat history ──────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user"><div class="msg-label msg-label-user">You</div>'
                f'{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            citations_html = ""
            if msg.get("citations"):
                items = ""
                for c in msg["citations"]:
                    icon = "🖼️" if c.get("type") == "image" else ("📊" if c.get("type") == "table" else "📄")
                    section = c.get("section_path") or c.get("heading") or ""
                    section_part = (
                        f' <span class="cite-heading">· {section}</span>'
                        if section else ""
                    )
                    items += (
                        f'<span class="cite-item">'
                        f'{icon} {c["source"]} p.{c["page"]}'
                        f'{section_part}'
                        f'</span>'
                    )
                citations_html = f'<div class="citations-box"><strong>Sources</strong><br>{items}</div>'

            usage = msg.get("usage", {})
            tokens_html = (
                f'<div style="margin-top:6px;font-size:0.7rem;color:#2e2e46;">'
                f'{usage["total_tokens"]} tokens</div>'
            ) if usage.get("total_tokens") else ""

            st.markdown(
                f'<div class="msg-assistant"><div class="msg-label msg-label-assistant">Assistant</div>'
                f'{msg["content"]}{citations_html}{tokens_html}</div>',
                unsafe_allow_html=True
            )

            if msg.get("image_paths"):
                for img_path in msg["image_paths"]:
                    if os.path.exists(img_path):
                        img_col, _ = st.columns([3, 2])
                        with img_col:
                            st.markdown(
                                '<div class="image-result-label">🖼️ Referenced Image</div>',
                                unsafe_allow_html=True
                            )
                            st.image(
                                img_path,
                                caption=f"Source: {os.path.basename(img_path)}",
                                width=460,
                            )

    # ── Input ─────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_input = st.text_input(
                "Q", placeholder="Ask anything about your documents…",
                label_visibility="collapsed"
            )
        with col_btn:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_input.strip():
        question = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": question})

        has_kb = st.session_state.total_chunks > 0

        meta      = {"citations": [], "image_paths": [], "usage": {}}
        chain_gen = st.session_state.chain.stream(question, has_kb=has_kb)

        def token_generator():
            for item in chain_gen:
                if isinstance(item, str):
                    yield item
                else:
                    meta["citations"]   = item.get_citations()
                    meta["image_paths"] = item.get_images()
                    meta["usage"]       = item.usage

        # Show a status indicator during retrieval phase, then stream
        status_box = st.empty()
        status_box.markdown(
            '<div style="color:#504f6a;font-size:0.82rem;padding:6px 0;">🔍 Searching documents…</div>',
            unsafe_allow_html=True,
        )

        with st.chat_message("assistant"):
            full_answer = st.write_stream(token_generator())

        status_box.empty()

        st.session_state.messages.append({
            "role"       : "assistant",
            "content"    : full_answer,
            "citations"  : meta["citations"],
            "image_paths": meta["image_paths"],
            "usage"      : meta["usage"],
        })
        st.rerun()